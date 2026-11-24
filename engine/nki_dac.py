"""Fused NKI kernels of the codec's DAC decoder: one block's residual stack, and the whole DAC tail.

A DAC decoder block ends with three residual units, each
    y = x + conv2(snake2(conv1(snake1(x))))        conv1: k=7, dilation d in (1, 3, 9), causal
                                                   conv2: k=1
with snake(v) = v + ib * sin(a * v)^2 per channel. In the XLA graph (CodecStep._dac) every one of
those ops round-trips the whole [B, T, C] FP32 activation through HBM. `resstack_kernel` runs the
three units for one stream at a time, channels-first ([C, T], C on the partitions), keeping x,
the BF16 conv inputs and every intermediate in SBUF; the convs are BF16 matmuls accumulated in
FP32 PSUM (taps x input-channel tiles), T in chunks of <= 512 columns (one PSUM bank).

Numerics: as the XLA path (BF16 weights and conv inputs, FP32 accumulation, FP32 bias, snake and
residual, BF16 carried contexts). sin on the Scalar Engine is valid on [-pi, pi] only, so the
argument is range-reduced first: k = int(a v / pi) (any integer rounding mode gives
|a v / pi - k| < 1), r = v - k pi / a, sin(a r) = +-sin(a v), and sin^2 is unchanged.

Layout of the inputs (prepared once by `pack_block`):
    x    [B, C, T] FP32            the block's activation after its transposed conv
    c1..c3 [B, C, 6 d] BF16        the conv1 contexts (last 6 d conv1 inputs of the previous frame)
    w1   [C, 3 * 7 * C] BF16       [cin, (unit, tap, cout)]
    w2   [C, 3 * C] BF16           [cin, (unit, cout)]
    vec  [C, 3 * NV] FP32          per-channel constants of each unit (see V_* below)
Returns y [B, C, T] FP32 and the three new contexts [B, C, 6 d] BF16.
Launched with a grid of 2 (LNC=2): each physical core takes half of the streams.

`tail_kernel` (built by the same `_build_kernel`) is the serving form: it runs the whole DAC tail
of each stream (decoder.3 and decoder.4, each a transposed conv plus the residual stack above,
then the final snake and conv_out) with every intermediate in SBUF, and returns the PCM and the
tail's nine carried contexts (channels-first). `tail_ref` is its FP32 CPU reference, used by the
startup self-test (DeviceCodec.nki_selftest).
"""

from __future__ import annotations

import math

DIL = (1, 3, 9)
KT = 7  # conv1 taps
FCH = 512  # PSUM bank: FP32 columns per matmul output

# per-unit constant columns of `vec`
V_A1PI, V_NPIA1, V_A1, V_SIB1, V_B1, V_A2PI, V_NPIA2, V_A2, V_SIB2, V_A2B1, V_B2 = range(11)
NV = 11
# tail kernel (decoder.3 .. conv_out): extra columns of vec3 / vec4 after the 3 * NV unit columns
T_B = 3 * NV  # the transposed conv's bias (vec3: decoder.3's, vec4: decoder.4's)
T_A = 3 * NV + 1  # 4 snake columns a / pi, -pi / a, a, sqrt(1 / b): vec3 decoder.4's input snake, vec4 the final snake
T_OB = 3 * NV + 5  # vec4: conv_out's bias
NV3, NV4 = 3 * NV + 5, 3 * NV + 6

_KERNEL = None


def _chunks(T: int, fmax: int = FCH):
    n = -(-T // fmax)
    base, rem = divmod(T, n)
    out, t0 = [], 0
    for i in range(n):
        w = base + (1 if i < rem else 0)
        out.append((t0, w))
        t0 += w
    return out


def _build_kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl

    def snake(dst, src, vs, u, first: bool, P: int, n: int):
        """dst (BF16) = snake(src) (+ b1 first when not `first`: src is conv1's PSUM)."""
        o = u * NV
        col = lambda j: vs[0:P, o + j : o + j + 1]  # noqa: E731
        if first:
            api_, npia, a, sib = V_A1PI, V_NPIA1, V_A1, V_SIB1
        else:
            api_, npia, a, sib = V_A2PI, V_NPIA2, V_A2, V_SIB2
        k = nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)
        if first:
            nisa.tensor_scalar(dst=k, data=src, op0=nl.multiply, operand0=col(api_))
        else:
            nisa.tensor_scalar(dst=k, data=src, op0=nl.add, operand0=col(V_B1), op1=nl.multiply, operand1=col(api_))
        r = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=r, data=k, op0=nl.multiply, operand0=col(npia), op1=nl.add, operand1=src)
        s = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        if first:
            nisa.activation(dst=s, op=nl.sin, data=r, scale=col(a))
        else:
            nisa.activation(dst=s, op=nl.sin, data=r, scale=col(a), bias=col(V_A2B1))
        t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=t, op=nl.square, data=s, scale=col(sib))
        if first:
            nisa.tensor_tensor(dst=dst, data1=src, data2=t, op=nl.add)
        else:
            nisa.scalar_tensor_tensor(dst=dst, data=src, op0=nl.add, operand0=col(V_B1), op1=nl.add, operand1=t)

    def snake_in(dst, src, vs, o: int, P: int, n: int):
        """dst (BF16) = snake(src) for a block-input or final snake: vs columns o .. o + 3 hold
        a / pi, -pi / a, a, sqrt(1 / b) (SN_* below); src is FP32 in SBUF."""
        col = lambda j: vs[0:P, o + j : o + j + 1]  # noqa: E731
        k = nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=k, data=src, op0=nl.multiply, operand0=col(0))
        r = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=r, data=k, op0=nl.multiply, operand0=col(1), op1=nl.add, operand1=src)
        s = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=s, op=nl.sin, data=r, scale=col(2))
        t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=t, op=nl.square, data=s, scale=col(3))
        nisa.tensor_tensor(dst=dst, data1=src, data2=t, op=nl.add)

    def load_w(src, nct: int, P: int, dtype):
        """src [C, N] in HBM -> nct SBUF tiles [P, N] (C = nct * P)."""
        N = src.shape[1]
        out = []
        for ci in range(nct):
            t = nl.ndarray((P, N), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=src[ci * P : (ci + 1) * P, 0:N])
            out.append(t)
        return out

    def stack(b, xs, cin, cout, w1s, w2s, vss, C: int, nct: int, P: int, T: int):
        """The three residual units of one stream b, in place on xs (nct tiles [P, T] FP32);
        cin/cout the three conv1 contexts [B, C, 6 d] BF16 in HBM (read / written)."""
        chunks = _chunks(T)
        for u in range(3):
            d = DIL[u]
            L = (KT - 1) * d
            bufs = []
            for ci in range(nct):
                bt = nl.ndarray((P, L + T), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=bt[0:P, 0:L], src=cin[u][b, ci * P : (ci + 1) * P, 0:L])
                for t0, n in chunks:
                    snake(bt[0:P, L + t0 : L + t0 + n], xs[ci][0:P, t0 : t0 + n], vss[ci], u, True, P, n)
                nisa.dma_copy(dst=cout[u][b, ci * P : (ci + 1) * P, 0:L], src=bt[0:P, T : T + L])
                bufs.append(bt)
            for t0, n in chunks:
                h2 = []
                for co in range(nct):
                    ps = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)
                    i = 0
                    for ci in range(nct):
                        for k in range(KT):
                            off = (u * KT + k) * C + co * P
                            nisa.nc_matmul(dst=ps, stationary=w1s[ci][0:P, off : off + P],
                                           moving=bufs[ci][0:P, k * d + t0 : k * d + t0 + n], accumulate=(i > 0))
                            i += 1
                    hb = nl.ndarray((P, n), dtype=nl.bfloat16, buffer=nl.sbuf)
                    snake(hb, ps, vss[co], u, False, P, n)
                    h2.append(hb)
                for co in range(nct):
                    ps2 = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)
                    for ci in range(nct):
                        off = u * C + co * P
                        nisa.nc_matmul(dst=ps2, stationary=w2s[ci][0:P, off : off + P], moving=h2[ci], accumulate=(ci > 0))
                    xv = xs[co][0:P, t0 : t0 + n]
                    o = u * NV
                    nisa.scalar_tensor_tensor(dst=xv, data=ps2, op0=nl.add, operand0=vss[co][0:P, o + V_B2 : o + V_B2 + 1],
                                              op1=nl.add, operand1=xv)

    def tconv(b, xs, prev, nprev, wts, vin, acol: int, vout, bcol: int, Cin: int, Cout: int, K: int, n: int):
        """A decoder block's input snake and transposed conv (stride r = K / 2) for stream b:
        xs nci tiles [Pi, n] FP32 (the block input) -> nco tiles [Po, n r] FP32. prev/nprev
        [B, Cin, 1] BF16 in HBM: the previous frame's last snake column (read / written).
        y[:, t r + j] = W_j^T x[:, t] + W_{r+j}^T x[:, t - 1] + bias, both taps accumulated in one
        PSUM tile (x[:, -1] = prev); vin[ci] columns acol .. acol + 3 the snake, vout[co] column bcol
        the bias."""
        r = K // 2
        nci, nco = (Cin + 127) // 128, (Cout + 127) // 128
        Pi, Po = Cin // nci, Cout // nco
        bufs = []
        for ci in range(nci):
            bt = nl.ndarray((Pi, n + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bt[0:Pi, 0:1], src=prev[b, ci * Pi : (ci + 1) * Pi, 0:1])
            for t0, m in _chunks(n):
                snake_in(bt[0:Pi, 1 + t0 : 1 + t0 + m], xs[ci][0:Pi, t0 : t0 + m], vin[ci], acol, Pi, m)
            nisa.dma_copy(dst=nprev[b, ci * Pi : (ci + 1) * Pi, 0:1], src=bt[0:Pi, n : n + 1])
            bufs.append(bt)
        ys = []
        for co in range(nco):
            y = nl.ndarray((Po, n * r), dtype=nl.float32, buffer=nl.sbuf)
            ys.append(y)
        for co in range(nco):
            for j in range(r):
                for t0, m in _chunks(n):
                    ps = nl.ndarray((Po, m), dtype=nl.float32, buffer=nl.psum)
                    i = 0
                    for ci in range(nci):
                        o1 = j * Cout + co * Po
                        o2 = (r + j) * Cout + co * Po
                        nisa.nc_matmul(dst=ps, stationary=wts[ci][0:Pi, o1 : o1 + Po], moving=bufs[ci][0:Pi, 1 + t0 : 1 + t0 + m],
                                       accumulate=(i > 0))
                        nisa.nc_matmul(dst=ps, stationary=wts[ci][0:Pi, o2 : o2 + Po], moving=bufs[ci][0:Pi, t0 : t0 + m],
                                       accumulate=True)
                        i += 2
                    nisa.tensor_scalar(dst=ys[co][0:Po, r * t0 + j : r * (t0 + m) : r], data=ps, op0=nl.add,
                                       operand0=vout[co][0:Po, bcol : bcol + 1])
        return ys

    def grid_range(B):
        grid_ndim = nl.program_ndim()
        nprog, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
        bper = (B + nprog - 1) // nprog
        b_lo = pid * bper
        return b_lo, min(B, b_lo + bper)

    @nki.jit
    def resstack_kernel(x, c1, c2, c3, w1, w2, vec):
        B, C, T = x.shape
        nct = (C + 127) // 128
        P = C // nct
        cin = (c1, c2, c3)
        y = nl.ndarray((B, C, T), dtype=nl.float32, buffer=nl.shared_hbm)
        n1 = nl.ndarray(c1.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        n2 = nl.ndarray(c2.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        n3 = nl.ndarray(c3.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        cout = (n1, n2, n3)
        b_lo, b_hi = grid_range(B)
        w1s = load_w(w1, nct, P, nl.bfloat16)
        w2s = load_w(w2, nct, P, nl.bfloat16)
        vss = load_w(vec, nct, P, nl.float32)
        for b in range(b_lo, b_hi):
            xs = []
            for ci in range(nct):
                t = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=t, src=x[b, ci * P : (ci + 1) * P, 0:T])
                xs.append(t)
            stack(b, xs, cin, cout, w1s, w2s, vss, C, nct, P, T)
            for ci in range(nct):
                nisa.dma_copy(dst=y[b, ci * P : (ci + 1) * P, 0:T], src=xs[ci])
        return y, n1, n2, n3

    @nki.jit
    def tail_kernel(h2, p3, c31, c32, c33, p4, c41, c42, c43, co, wt3, vin3, w13, w23, vec3, wt4, w14, w24, vec4, wo):
        """The DAC tail: decoder.3 + decoder.4 + final snake + conv_out of each stream, channels-first,
        every intermediate in SBUF. h2 [B, C3, n3] FP32 (decoder.2's output) -> pcm [B, n3 r3 r4]
        FP32 in [-1, 1] and the nine new contexts (same shapes as p3 .. co)."""
        B, C3, n3 = h2.shape
        C4 = c31.shape[1]
        C5 = c41.shape[1]
        K3, K4 = wt3.shape[1] // C4, wt4.shape[1] // C5
        T3, T4 = n3 * (K3 // 2), n3 * (K3 // 2) * (K4 // 2)
        ins = (p3, c31, c32, c33, p4, c41, c42, c43, co)
        outs = [nl.ndarray(t.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm) for t in ins]
        pcm = nl.ndarray((B, T4), dtype=nl.float32, buffer=nl.shared_hbm)
        b_lo, b_hi = grid_range(B)
        n3t, n4t = (C3 + 127) // 128, (C4 + 127) // 128
        P3, P4, P5 = C3 // n3t, C4 // n4t, C5
        wt3s = load_w(wt3, n3t, P3, nl.bfloat16)
        vin3s = load_w(vin3, n3t, P3, nl.float32)
        w13s = load_w(w13, n4t, P4, nl.bfloat16)
        w23s = load_w(w23, n4t, P4, nl.bfloat16)
        vec3s = load_w(vec3, n4t, P4, nl.float32)
        wt4s = load_w(wt4, n4t, P4, nl.bfloat16)
        w14s = load_w(w14, 1, P5, nl.bfloat16)
        w24s = load_w(w24, 1, P5, nl.bfloat16)
        vec4s = load_w(vec4, 1, P5, nl.float32)
        wos = load_w(wo, 1, P5, nl.bfloat16)
        LO = co.shape[2]
        for b in range(b_lo, b_hi):
            xin = []
            for ci in range(n3t):
                t = nl.ndarray((P3, n3), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=t, src=h2[b, ci * P3 : (ci + 1) * P3, 0:n3])
                xin.append(t)
            # decoder.3: snake (vin3 cols 0..3), tconv (bias: vec3 col T_B), residual stack
            xs3 = tconv(b, xin, p3, outs[0], wt3s, vin3s, 0, vec3s, T_B, C3, C4, K3, n3)
            stack(b, xs3, (c31, c32, c33), outs[1:4], w13s, w23s, vec3s, C4, n4t, P4, T3)
            # decoder.4: snake (vec3 cols T_A..), tconv (bias vec4 col T_B), residual stack
            xs4 = tconv(b, xs3, p4, outs[4], wt4s, vec3s, T_A, vec4s, T_B, C4, C5, K4, T3)
            stack(b, xs4, (c41, c42, c43), outs[5:8], w14s, w24s, vec4s, C5, 1, P5, T4)
            # final snake (vec4 cols T_A..) and conv_out (k = LO + 1, Cout = 1), clamp
            bo = nl.ndarray((P5, LO + T4), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=bo[0:P5, 0:LO], src=co[b, 0:P5, 0:LO])
            for t0, m in _chunks(T4):
                snake_in(bo[0:P5, LO + t0 : LO + t0 + m], xs4[0][0:P5, t0 : t0 + m], vec4s[0], T_A, P5, m)
            nisa.dma_copy(dst=outs[8][b, 0:P5, 0:LO], src=bo[0:P5, T4 : T4 + LO])
            ys = nl.ndarray((1, T4), dtype=nl.float32, buffer=nl.sbuf)
            for t0, m in _chunks(T4):
                ps = nl.ndarray((1, m), dtype=nl.float32, buffer=nl.psum)
                for k in range(LO + 1):
                    nisa.nc_matmul(dst=ps, stationary=wos[0][0:P5, k : k + 1], moving=bo[0:P5, k + t0 : k + t0 + m], accumulate=(k > 0))
                yv = ys[0:1, t0 : t0 + m]
                nisa.tensor_scalar(dst=yv, data=ps, op0=nl.add, operand0=vec4s[0][0:1, T_OB : T_OB + 1], op1=nl.maximum, operand1=-1.0)
                nisa.tensor_scalar(dst=yv, data=yv, op0=nl.minimum, operand0=1.0)
            nisa.dma_copy(dst=pcm[b : b + 1, 0:T4], src=ys)
        return (pcm, *outs)

    return resstack_kernel, tail_kernel


def kernels():
    """(resstack_kernel, tail_kernel), the @nki.jit kernels (built on first use: imports nki)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def kernel():
    """The residual-stack kernel (one DAC block's three residual units)."""
    return kernels()[0]


def tail_kernel():
    """The DAC tail kernel (decoder.3 .. conv_out, the serving path on a NeuronCore)."""
    return kernels()[1]


_WRAPPED = {}


def wrapped(grid: int = 2, name: str = "resstack"):
    """A kernel wrapped for torch (torch.compile graphs and eager Neuron tensors), launched on
    `grid` programs (2 = both physical cores of an LNC=2 logical core); name resstack or tail."""
    if name not in _WRAPPED:
        from torch_neuronx import wrap_nki

        _WRAPPED[name] = wrap_nki(kernels()[0 if name == "resstack" else 1])
    w = _WRAPPED[name]
    return w[grid] if grid > 1 else w


# ---------------------------------------------------------------------------- host side
def pack_block(block):
    """(w1, w2, vec) of a CodecStep block dict (its `units`), as the kernel takes them; on the
    device of the block's parameters."""
    import torch

    BF = torch.bfloat16
    us = block["units"]
    W1 = torch.stack([u["c1"][0].float() for u in us])  # [3, 7, Cin, Cout]
    C = W1.shape[-1]
    w1 = W1.permute(2, 0, 1, 3).reshape(C, 3 * KT * C).to(BF).contiguous()
    W2 = torch.stack([u["c2"][0][0].float() for u in us])  # [3, Cin, Cout]
    w2 = W2.permute(1, 0, 2).reshape(C, 3 * C).to(BF).contiguous()
    cols = []
    for u in us:
        a1, ib1 = (t.float() for t in u["a1"])
        a2, ib2 = (t.float() for t in u["a2"])
        b1, b2 = u["c1"][1].float(), u["c2"][1].float()
        cols.append(torch.stack([a1 / math.pi, -math.pi / a1, a1, ib1.sqrt(), b1,
                                 a2 / math.pi, -math.pi / a2, a2, ib2.sqrt(), a2 * b1, b2], 1))  # [C, NV]
    vec = torch.cat(cols, 1).float().contiguous()
    return w1, w2, vec


def resstack_nki(x, sts, packed, grid: int = 2):
    """The block's three units through the kernel: x [B, T, C] FP32 (channels-last, as CodecStep
    holds it), sts the three conv1 contexts [B, 6 d, C] BF16 -> (y [B, T, C] FP32, new contexts)."""
    w1, w2, vec = packed
    xc = x.transpose(1, 2).contiguous()
    cs = [s.transpose(1, 2).contiguous() for s in sts]
    y, n1, n2, n3 = wrapped(grid)(xc, cs[0], cs[1], cs[2], w1, w2, vec)
    return y.transpose(1, 2), [n.transpose(1, 2) for n in (n1, n2, n3)]


def resstack_xla(step, block, x, sts):
    """The same three units as CodecStep._dac runs them (the reference path)."""
    new = []
    h = x
    for u, st in zip(block["units"], sts):
        r = h
        y, n = step._conv(step._snake(h, u["a1"]), st, u["c1"])
        new.append(n)
        y, _ = step._conv(step._snake(y, u["a2"]), None, u["c2"])
        h = y + r
    return h, new


def resstack_ref(cw, blk: int, x_cf, ctx_cf):
    """FP32 CPU reference (original FP32 weights), channels-first: x [B, C, T], ctx 3 x [B, C, 6 d]
    -> (y [B, C, T], 3 new contexts [B, C, 6 d] FP32). blk is 1-based (decoder.<blk>)."""
    import torch
    import torch.nn.functional as F

    w = cw.w
    p = f"decoder.{blk}.block."

    def snake(v, q):
        a = torch.exp(w[q + "alpha"]).float()[:, None]
        ib = (1.0 / (torch.exp(w[q + "beta"]) + 1e-9)).float()[:, None]
        return v + ib * torch.sin(v * a).pow(2)

    h, new = x_cf.float(), []
    for j, d in enumerate(DIL):
        q = f"{p}{j + 2}."
        full = torch.cat([ctx_cf[j].float(), snake(h, q + "act1.")], -1)
        new.append(full[..., -ctx_cf[j].shape[-1] :])
        y = F.conv1d(full, w[q + "conv1.conv.weight"].float(), w[q + "conv1.conv.bias"].float(), dilation=d)
        y = F.conv1d(snake(y, q + "act2."), w[q + "conv2.conv.weight"].float(), w[q + "conv2.conv.bias"].float())
        h = y + h
    return h, new


# ---------------------------------------------------------------------------- the DAC tail kernel
def _snake_cols(prm):
    """[C, 4] FP32: a / pi, -pi / a, a, sqrt(1 / b) of a CodecStep snake (a, ib)."""
    import torch

    a, ib = (t.float() for t in prm)
    return torch.stack([a / math.pi, -math.pi / a, a, ib.sqrt()], 1)


def pack_tail(step):
    """The tail kernel's weights from a CodecStep (decoder.3, decoder.4, final snake, conv_out):
    (wt3, vin3, w13, w23, vec3, wt4, w14, w24, vec4, wo), on the device of its parameters."""
    import torch

    BF = torch.bfloat16
    b3, b4 = step.blocks[2], step.blocks[3]
    w13, w23, v3 = pack_block(b3)
    w14, w24, v4 = pack_block(b4)
    wt3 = b3["t"][0].to(BF).contiguous()  # [Cin, K * Cout]
    wt4 = b4["t"][0].to(BF).contiguous()
    vin3 = _snake_cols(b3["act"]).contiguous()  # [384, 4]
    vec3 = torch.cat([v3, b3["t"][1].float()[:, None], _snake_cols(b4["act"])], 1).contiguous()  # [192, NV3]
    Wo, bo, _ = step.conv_out  # [7, 96, 1], [1]
    C5 = Wo.shape[1]
    vec4 = torch.cat([v4, b4["t"][1].float()[:, None], _snake_cols(step.final_act), bo.float().reshape(1, 1).expand(C5, 1)], 1).contiguous()
    wo = Wo[:, :, 0].T.to(BF).contiguous()  # [96, 7]
    assert vec3.shape[1] == NV3 and vec4.shape[1] == NV4
    return wt3, vin3, w13, w23, vec3, wt4, w14, w24, vec4, wo


def tail_ctx_shapes(step):
    """[(C, L)] of the tail's nine carried contexts, in the kernel's (and CodecStep.ctx's) order:
    decoder.3 tconv prev, its 3 conv1 contexts, decoder.4 tconv prev, its 3, conv_out's."""
    return [(c, k) for c, k in step.ctx[step.n_xf_ctx + step.n_head_ctx :] if k]


def tail_nki(h2_cf, sts_cf, packed, grid: int = 2):
    """h2 [B, C3, n] FP32 channels-first, sts the nine contexts [B, C, L] BF16 channels-first ->
    (pcm [B, 1920] FP32, the nine new contexts)."""
    out = wrapped(grid, "tail")(h2_cf, *sts_cf, *packed)
    return out[0], list(out[1:])


def tail_ref(cw, h2_cf, ctx_cf):
    """FP32 CPU reference (original FP32 weights) of decoder.3 .. conv_out, channels-first: h2
    [B, C3, n], ctx the nine contexts [B, C, L] -> (pcm [B, T] clamped, nine new contexts FP32,
    max |pre-clamp output|, max |decoder.4 output|)."""
    import torch
    import torch.nn.functional as F

    w = cw.w

    def snake(v, q):
        a = torch.exp(w[q + "alpha"]).float()[:, None]
        ib = (1.0 / (torch.exp(w[q + "beta"]) + 1e-9)).float()[:, None]
        return v + ib * torch.sin(v * a).pow(2)

    h, new, it = h2_cf.float(), [], iter(ctx_cf)
    for blk in (3, 4):
        p = f"decoder.{blk}.block."
        prev = next(it).float()
        full = torch.cat([prev, snake(h, p + "0.")], -1)  # [B, Cin, n + 1]
        new.append(full[..., -1:])
        W = w[p + "1.conv.weight"].float()  # [Cin, Cout, K]
        K = W.shape[-1]
        r = K // 2
        z = torch.einsum("bct,cok->bkot", full, W)  # [B, K, Cout, n + 1]
        y = torch.stack([z[:, j, :, 1:] + z[:, r + j, :, :-1] for j in range(r)], -1)  # [B, Cout, n, r]
        h = y.reshape(y.shape[0], y.shape[1], -1) + w[p + "1.conv.bias"].float()[:, None]
        h, ns = resstack_ref(cw, blk, h, [next(it) for _ in DIL])
        new += ns
    h4 = h
    co = next(it).float()
    full = torch.cat([co, snake(h, "decoder.5.")], -1)
    new.append(full[..., -co.shape[-1] :])
    y = F.conv1d(full, w["decoder.6.conv.weight"].float(), w["decoder.6.conv.bias"].float())[:, 0]
    return y.clamp(-1.0, 1.0), new, float(y.abs().max()), float(h4.abs().max())
