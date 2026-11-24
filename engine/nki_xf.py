"""NKI ring attention of the codec's xf graph (CodecStep.forward_xf, QWEN3TTS_CODEC_XF_ATTN=nki).

CodecStep.forward_xf attends each row's new query over its slot's 72-column key/value ring, per
head and layer: in XLA that is two tiny FP32 matmuls per (row, head) ([1, 64] x [64, 72] and
[1, 72] x [72, 64]), ~1,900 PE instructions per row, and 5.35 of the 6.73 ms of the B=64 xf graph
(profiled on trn2). Here (row, head) pairs sit on the partitions, 128 per tile, and the two
contractions are VectorE multiplies plus free-axis reductions over the whole tile at once:

    s[p, j]  = sum_d q[p, d] K[p, j, d]          (FP32; K BF16 from the ring)
    s        = s * scale + kbias[p, j]           columns 0..71: the ring; column 72: s_new
    w        = softmax(s)                        (FP32; exp on ScalarE, sum, reciprocal, multiply)
    o[p, d]  = sum_j w[p, j] V[p, j, d] + w[p, 72] vnew[p, d]

The ring rows are gathered straight from the K / V tables by an indirect DMA (one index per
partition: the table row of (layer, slot, head)), so no gathered copy goes through HBM.

Inputs (N = B * H, a multiple of 128):
    q     [N, hd]  FP32   the post-RoPE query
    snew  [N, 1]   FP32   (q . k_new) * scale, the score of the row's own new key
    kbias [N, R]   FP32   0 on ring columns holding one of the previous 71 frames, else -FLT_MAX
    vnew  [N, hd]  FP32   the new value (BF16-rounded, as the ring holds it)
    K, V  [rows, R * hd] BF16  the ring tables, one row per (layer, slot, head)
    idx   [N, 1]   int32  the table row of each (row, head) for this layer
Returns o [N, hd] FP32. Launched on a grid of 2 (LNC=2): each physical core takes half the tiles.
"""

from __future__ import annotations

import os

P = 128
_KERNELS = {}
_WRAPPED = {}


def _build(mul_engine: str):
    import nki
    import nki.isa as nisa
    import nki.language as nl

    # the engine of the two big multiplies (QK, PV): vector, gpsimd, or split (QK on VectorE, PV on
    # GpSimd, so consecutive tiles overlap); the reductions and the softmax run on VectorE/ScalarE.
    eng_qk = nisa.engine.gpsimd if mul_engine == "gpsimd" else nisa.engine.vector
    eng_pv = nisa.engine.gpsimd if mul_engine in ("gpsimd", "split") else nisa.engine.vector

    @nki.jit
    def ring_attn_kernel(q, snew, kbias, vnew, K, V, idx):
        N, hd = q.shape
        R = kbias.shape[1]
        scale = float(hd) ** -0.5
        out = nl.ndarray((N, hd), dtype=nl.float32, buffer=nl.shared_hbm)
        nt = N // P
        grid_ndim = nl.program_ndim()
        nprog, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
        per = (nt + nprog - 1) // nprog
        t_lo, t_hi = pid * per, min(nt, pid * per + per)
        for t in range(t_lo, t_hi):
            r0 = t * P
            it = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=it, src=idx[r0 : r0 + P, 0:1])
            kt = nl.ndarray((P, R, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kt, src=K.ap(pattern=[[R * hd, P], [1, R * hd]], vector_offset=it, indirect_dim=0))
            vt = nl.ndarray((P, R, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt, src=V.ap(pattern=[[R * hd, P], [1, R * hd]], vector_offset=it, indirect_dim=0))
            qt = nl.ndarray((P, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=qt, src=q[r0 : r0 + P, 0:hd])
            bt = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bt, src=kbias[r0 : r0 + P, 0:R])
            vn = nl.ndarray((P, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vn, src=vnew[r0 : r0 + P, 0:hd])
            s = nl.ndarray((P, R + 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s[0:P, R : R + 1], src=snew[r0 : r0 + P, 0:1])
            # scores over the ring: K * q (q broadcast over the ring axis), summed over hd
            prod = nl.ndarray((P, R, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=kt, data2=qt.expand_dim(1).broadcast(1, R), op=nl.multiply, engine=eng_qk)
            nisa.tensor_reduce(dst=s[0:P, 0:R], op=nl.add, data=prod, axis=(2,))
            nisa.scalar_tensor_tensor(dst=s[0:P, 0:R], data=s[0:P, 0:R], op0=nl.multiply, operand0=scale, op1=nl.add, operand1=bt)
            # softmax over the R + 1 columns, FP32
            mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=s, axis=(1,))
            nmx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
            e = nl.ndarray((P, R + 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=s, bias=nmx, scale=1.0)
            den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=den, op=nl.add, data=e, axis=(1,))
            rden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rden, data=den)
            w = nl.ndarray((P, R + 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=w, data=e, op0=nl.multiply, operand0=rden)
            # weighted sum of the ring values: V * w (w broadcast over hd), summed over the ring axis
            pv = nl.ndarray((P, R, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pv, data1=vt, data2=w[0:P, 0:R].expand_dim(2).broadcast(2, hd), op=nl.multiply, engine=eng_pv)
            o = nl.ndarray((P, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=o, op=nl.add, data=pv.permute((0, 2, 1)), axis=(2,))
            # + w_new * v_new
            nisa.scalar_tensor_tensor(dst=o, data=vn, op0=nl.multiply, operand0=w[0:P, R : R + 1], op1=nl.add, operand1=o)
            nisa.dma_copy(dst=out[r0 : r0 + P, 0:hd], src=o)
        return out

    return ring_attn_kernel


ENGINES = ("vector", "gpsimd", "split")


def mul_engine() -> str:
    e = os.environ.get("QWEN3TTS_XF_NKI_ENGINE", "vector")
    if e not in ENGINES:
        raise SystemExit(f"QWEN3TTS_XF_NKI_ENGINE={e!r}: choose one of {ENGINES}")
    return e


def kernel(engine: str | None = None):
    e = engine or mul_engine()
    if e not in _KERNELS:
        _KERNELS[e] = _build(e)
    return _KERNELS[e]


def wrapped(grid: int = 2, engine: str | None = None):
    """The kernel wrapped for torch (inside torch.compile graphs and on eager Neuron tensors)."""
    e = engine or mul_engine()
    if e not in _WRAPPED:
        from torch_neuronx import wrap_nki

        _WRAPPED[e] = wrap_nki(kernel(e))
    w = _WRAPPED[e]
    return w[grid] if grid > 1 else w


def ring_attn_nki(q, snew, kbias, vnew, K2, V2, idx, grid: int = 2):
    return wrapped(grid)(q, snew, kbias, vnew, K2, V2, idx)


def ring_attn_ref(q, snew, kbias, vnew, K2, V2, idx):
    """The same function in FP32 torch, the per-(row, head) matmul form of CodecStep's `mm` path."""
    import torch

    N, hd = q.shape
    R = kbias.shape[1]
    Ko = K2[idx[:, 0].long()].float().view(N, R, hd)
    Vo = V2[idx[:, 0].long()].float().view(N, R, hd)
    s_old = torch.matmul(q[:, None, :], Ko.transpose(1, 2))[:, 0] * hd**-0.5 + kbias
    p = torch.softmax(torch.cat([s_old, snew], -1), -1)
    return torch.matmul(p[:, None, :-1], Vo)[:, 0] + p[:, -1:] * vnew
