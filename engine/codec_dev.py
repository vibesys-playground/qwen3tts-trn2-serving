"""Batched streaming decode of the Qwen3-TTS-Tokenizer-12Hz codec on a NeuronCore.

Two compiled graphs decode the next frame of up to B streams at once. Each stream lives in a
*slot*; slots come in groups of GROUP, and a call takes G whole groups (B = G * GROUP rows).
Per-slot masks say which rows decode a frame this call (`active`) and which start a new stream
(`fresh`, state read as zeros); rows that are not active keep their state unchanged.

- The xf graph (pre_conv, the 8-layer transformer, the upsamplers) keeps its state on the device
  and updates it in place, indexed by slot: each slot's key/value window is a 72-column ring in
  one K and one V table (frame p's post-RoPE key/value is written at column p % 72 by one
  `scatter_`; the host's key bias masks the columns that do not hold one of the previous 71
  frames), and its three conv contexts are [slots + 1, ctx, C] tables read with `index_select`
  and written with `index_copy_`. Rows with no frame point at a scratch slot.
- The DAC runs as two graphs. The head graph (decoder.0 .. decoder.2) takes each
  group's head conv contexts as one flat [GROUP, N] tensor and returns the group's updated tensor
  (`where(active)`), as the one-graph CodecStep.forward did for the whole state; it returns h2
  [B, 160, 384] FP32. The tail graph (decoder.3 .. conv_out) does the same with its own group
  tensors; on a NeuronCore it is one fused NKI kernel (engine.nki_dac.tail_kernel) and its
  contexts are stored channels-first (h2 is then transposed at the end of the head graph).
  QWEN3TTS_CODEC_NKI=0 runs the tail as XLA code, channels-last. Only h [B, 4, 1024] and h2
  FP32 pass between the graphs.

Host side: CodecService (one codec thread) picks, per device call, the pending frames of up to
`max_groups` slot groups earliest-deadline-first (CodecStream.deadline: the TTFA budget before the
first frame, then the moment the client's player would run dry), runs one call, converts the PCM
and hands it to a FIFO delivery thread that puts it on each stream's queue (or, for async
streams, one event-loop callback per call). The startup self-tests (DeviceCodec.nki_selftest,
scheduler_selftest, and the server's codec self-test against the FP32 host decoder) exit the
process if the device path does not reproduce the reference decode.

The numerics are the host decoder's (engine.codec, a port of qwen-tts 0.1.1) in BF16: weights,
matmul inputs and the carried state in BF16, the residual streams, norms, softmax, snake and the
output clamp in FP32. Convolutions run channels-last as sums of shifted matmuls; transposed
convolutions as one matmul plus an overlap-add of neighbouring frames. The quantizer (a codebook
lookup and a linear projection, ~8 kFLOP a frame) runs on the host in FP32 and the graph starts
from its 512-wide output.
"""

from __future__ import annotations

import asyncio
import collections
import math
import os
import queue
import sys
import threading
import time
import traceback
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .codec import SAMPLES_PER_FRAME, CodecWeights, to_pcm16

BF = torch.bfloat16
GROUP = 8  # slots per state group
DIL = (1, 3, 9)
NKI_SELFTEST_DB = float(os.environ.get("QWEN3TTS_NKI_SELFTEST_DB", "35"))  # kernel vs FP32 CPU, startup
NKI_BLOCKS = (2, 3)  # 0-based decoder blocks (decoder.3, decoder.4) with the fused residual-stack kernel
HEAD_BLOCKS = 2  # the DAC head graph runs decoder.0 .. decoder.2, the tail graph the rest
# A head call of fewer than HEAD_MIN groups runs the head graph of the smallest bucket that holds
# HEAD_MIN groups (at B=16 the head graph spills to HBM; measured on trn2: 6.38 ms at B=16 vs
# 3.82 ms at B=32); a larger call runs its own.
HEAD_MIN = int(os.environ.get("QWEN3TTS_CODEC_HEAD_MIN", "4"))
# Experimental: groups per head launch (0: one launch per call, the default). With 4, an 8-group call runs the head as
# two B=32 launches of the same graph code (rows sliced inside the graph) and one B=64 tail.
HEAD_MAX = int(os.environ.get("QWEN3TTS_CODEC_HEAD_MAX", "0"))
XF_ATTN = ("mm", "vec", "vecf", "nki", "none")  # forward_xf ring-attention forms (see CodecStep)
XF_NKI_SELFTEST_DB = float(os.environ.get("QWEN3TTS_XF_NKI_SELFTEST_DB", "50"))  # ring-attention kernel vs FP32 CPU, startup


def _p(t: torch.Tensor, dtype=BF) -> nn.Parameter:
    return nn.Parameter(t.detach().to(dtype).contiguous(), requires_grad=False)


class CodecTables:
    """Host-side pieces: the quantizer table and the rotary/window inputs of a call."""

    def __init__(self, cw: CodecWeights):
        self.cw = cw
        # table[i, code] = codebook_i[code] @ out_proj_i^T, so the quantizer output is a sum of rows.
        t = [cw.books[0] @ cw.first_out[:, :, 0].T] + [b @ cw.rest_out[:, :, 0].T for b in cw.books[1:]]
        self.table = torch.stack(t).contiguous()  # [16, 2048, 512] FP32
        self.nq = self.table.shape[0]
        self.window = cw.cfg["sliding_window"]
        self.inv_freq = cw.inv_freq

    def quantize(self, codes: np.ndarray) -> torch.Tensor:
        """codes [B, 16] ints -> [B, 512] FP32."""
        c = torch.as_tensor(np.asarray(codes), dtype=torch.long).clamp(min=0)
        return self.table[torch.arange(self.nq)[None, :], c].sum(1)

    def rope(self, pos: np.ndarray):
        """cos, sin [B, head_dim] FP32 and the key bias [B, window - 1] for absolute frame positions."""
        p = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
        f = p[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), -1)
        w = self.window - 1
        kpos = p[:, None] - w + torch.arange(w, dtype=torch.float32)[None, :]
        bias = torch.where(kpos >= 0, 0.0, torch.finfo(torch.float32).min)
        return emb.cos(), emb.sin(), bias


class CodecStep(nn.Module):
    """One frame of the decoder for B = G * GROUP slots.

    forward(x [B, 512] FP32, cos, sin [B, 64] FP32, kbias [B, 71] FP32, active [B] FP32 0/1,
            fresh [B] FP32 0/1, *state: G x (kv [GROUP, L, 2, H, 71, hd] BF16, conv [GROUP, N] BF16))
        -> (pcm [B, 1920] FP32 in [-1, 1], *new state)
    """

    def __init__(self, cw: CodecWeights):
        super().__init__()
        w, cfg = cw.w, cw.cfg
        self.L = cfg["num_hidden_layers"]
        self.nh, self.hd = cfg["num_attention_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.win = cfg["sliding_window"] - 1
        # How forward_xf computes the 72-column ring attention (QWEN3TTS_CODEC_XF_ATTN):
        # nki (default): the NKI ring-attention kernel (engine.nki_xf), true FP32 (124 dB vs FP32
        # CPU at B=64; on a NeuronCore only, the mm math elsewhere); the xf graph at B=64 takes
        # 2.52 ms vs 6.73 ms with mm (measured on trn2).
        # mm: per-(row, head) matmuls (on the device its PE passes see BF16-rounded q and p:
        # 51-54 dB vs FP32); vec: broadcast multiply + reduce of the BF16 ring against FP32 q / p
        # (promoted to FP32); vecf: the same with an explicit upcast of the ring (XLA lowers both
        # back to mm's dot_general); none: timing floor only (o = v), refused unless
        # QWEN3TTS_CODEC_XF_ATTN_TIMING_ONLY=1 (benchmarks).
        self.xf_attn = os.environ.get("QWEN3TTS_CODEC_XF_ATTN", "nki")
        if self.xf_attn not in XF_ATTN or (self.xf_attn == "none" and os.environ.get("QWEN3TTS_CODEC_XF_ATTN_TIMING_ONLY") != "1"):
            raise SystemExit(f"QWEN3TTS_CODEC_XF_ATTN={self.xf_attn!r} is not servable (choose one of mm, vec, vecf, nki)")
        self.ctx = []  # (C, ctx) of every carried context, in call order

        def conv(name, dil=1):  # Conv1d weight [Cout, Cin, k] -> [k, Cin, Cout]
            W = w[name + ".weight"]
            k, cin = W.shape[-1], W.shape[1]
            self.ctx.append((cin, (k - 1) * dil))
            return (_p(W.permute(2, 1, 0)), _p(w[name + ".bias"], torch.float32), dil)

        def tconv(name):  # ConvTranspose1d weight [Cin, Cout, K] -> [Cin, K * Cout]
            W = w[name + ".weight"]
            cin, cout, k = W.shape
            return (_p(W.permute(0, 2, 1).reshape(cin, k * cout)), _p(w[name + ".bias"], torch.float32), k, cout)

        def snake(prefix):
            a = torch.exp(w[prefix + "alpha"])
            ib = 1.0 / (torch.exp(w[prefix + "beta"]) + 1e-9)
            return (_p(a, torch.float32), _p(ib, torch.float32))

        self.pre_conv = conv("pre_conv.conv")
        tw = lambda n: _p(torch.stack([w[f"pre_transformer.layers.{i}.{n}"] for i in range(self.L)]))  # noqa: E731
        tf = lambda n: _p(torch.stack([w[f"pre_transformer.layers.{i}.{n}"] for i in range(self.L)]), torch.float32)  # noqa: E731
        self.in_w, self.in_b = _p(w["pre_transformer.input_proj.weight"]), _p(w["pre_transformer.input_proj.bias"], torch.float32)
        self.wq, self.wk, self.wv, self.wo = (tw(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.wg, self.wu, self.wd = (tw(f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.ln1, self.ln2 = tf("input_layernorm.weight"), tf("post_attention_layernorm.weight")
        self.s_attn, self.s_mlp = tf("self_attn_layer_scale.scale"), tf("mlp_layer_scale.scale")
        self.norm = _p(w["pre_transformer.norm.weight"], torch.float32)
        self.out_w, self.out_b = _p(w["pre_transformer.output_proj.weight"]), _p(w["pre_transformer.output_proj.bias"], torch.float32)
        self.up = []
        for i in range(len(cfg["upsampling_ratios"])):
            p = f"upsample.{i}."
            dw = w[p + "1.dwconv.conv.weight"]  # [C, 1, 7]
            self.ctx.append((dw.shape[0], dw.shape[-1] - 1))
            self.up.append(
                dict(
                    t=tconv(p + "0.conv"),
                    dw=(_p(dw[:, 0, :].T, torch.float32), _p(w[p + "1.dwconv.conv.bias"], torch.float32)),
                    ln=(_p(w[p + "1.norm.weight"], torch.float32), _p(w[p + "1.norm.bias"], torch.float32)),
                    pw1=(_p(w[p + "1.pwconv1.weight"]), _p(w[p + "1.pwconv1.bias"], torch.float32)),
                    pw2=(_p(w[p + "1.pwconv2.weight"]), _p(w[p + "1.pwconv2.bias"], torch.float32)),
                    gamma=_p(w[p + "1.gamma"], torch.float32),
                )
            )
        self.conv_in = conv("decoder.0.conv")
        self.blocks = []
        for i in range(len(cfg["upsample_rates"])):
            p = f"decoder.{i + 1}.block."
            b = dict(act=snake(p + "0."), units=[])
            cin = w[p + "1.conv.weight"].shape[0]
            self.ctx.append((cin, 1))  # the transposed conv's previous input frame
            b["t"] = tconv(p + "1.conv")
            for j, dil in enumerate(DIL):
                q = f"{p}{j + 2}."
                b["units"].append(dict(a1=snake(q + "act1."), c1=conv(q + "conv1.conv", dil), a2=snake(q + "act2."), c2=conv(q + "conv2.conv")))
            self.blocks.append(b)
        n = len(cfg["upsample_rates"]) + 1
        self.final_act = snake(f"decoder.{n}.")
        self.conv_out = conv(f"decoder.{n + 1}.conv")
        # Blocks whose residual stack runs as one fused NKI kernel in the serving DAC graph
        # (decoder blocks 3 and 4; QWEN3TTS_CODEC_NKI=0 keeps the XLA code for comparison).
        self.nki_blocks = NKI_BLOCKS if os.environ.get("QWEN3TTS_CODEC_NKI", "1") != "0" else ()
        from .nki_dac import pack_block

        self.rs = [tuple(_p(t, t.dtype) for t in pack_block(self.blocks[i])) for i in self.nki_blocks]
        self.conv_state = sum(c * k for c, k in self.ctx)
        self.n_xf_ctx = 1 + len(self.up)  # pre_conv and the upsamplers: the xf graph's contexts
        assert all(k for _, k in self.ctx[: self.n_xf_ctx])
        self.dac_state = sum(c * k for c, k in self.ctx[self.n_xf_ctx :])
        # The DAC as two graphs. Head: decoder.0 .. decoder.2 (conv_in, blocks 1-2), its
        # contexts flat channels-last as before. Tail: decoder.3 .. conv_out, on a NeuronCore one
        # fused NKI kernel (engine.nki_dac.tail_kernel) whose contexts are stored channels-first.
        self.n_head_ctx = 1 + HEAD_BLOCKS * (1 + 2 * len(DIL))
        self.head_state = sum(c * k for c, k in self.ctx[self.n_xf_ctx : self.n_xf_ctx + self.n_head_ctx])
        self.tail_state = self.dac_state - self.head_state
        self.nki_tail = bool(self.nki_blocks)
        from .nki_dac import pack_tail

        self.tp = tuple(_p(t, t.dtype) for t in pack_tail(self)) if self.nki_tail else ()
        self.ring = self.win + 1  # KV ring columns: the 71-frame window plus the column being written
        # Register everything held in the dicts/tuples as parameters so .to(device) moves them.
        self._params = nn.ParameterList([t for t in self._walk((self.pre_conv, self.up, self.conv_in, self.blocks, self.final_act, self.conv_out, self.rs, self.tp))])

    @staticmethod
    def _walk(o):
        if isinstance(o, nn.Parameter):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from CodecStep._walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from CodecStep._walk(v)

    def _apply(self, fn, recurse=True):
        # Keep the dict/tuple views pointing at the moved parameters.
        super()._apply(fn, recurse)
        moved = iter(self._params)

        def rebuild(o):
            if isinstance(o, nn.Parameter):
                return next(moved)
            if isinstance(o, dict):
                return {k: rebuild(v) for k, v in o.items()}
            if isinstance(o, tuple):
                return tuple(rebuild(v) for v in o)
            if isinstance(o, list):
                return [rebuild(v) for v in o]
            return o

        self.pre_conv, self.up, self.conv_in, self.blocks, self.final_act, self.conv_out, self.rs, self.tp = rebuild(
            (self.pre_conv, self.up, self.conv_in, self.blocks, self.final_act, self.conv_out, self.rs, self.tp)
        )
        return self

    def state_shapes(self):
        return (GROUP, self.L, 2, self.nh, self.win, self.hd), (GROUP, self.conv_state)

    def new_group(self, device="cpu"):
        kv, cv = self.state_shapes()
        return torch.zeros(kv, dtype=BF, device=device), torch.zeros(cv, dtype=BF, device=device)

    # ------------------------------------------------------------------ pieces (channels-last)
    @staticmethod
    def _conv(x, st, prm):
        """Causal conv: x [B, n, Cin] FP32, st [B, ctx, Cin] BF16 -> (y [B, n, Cout] FP32, new st)."""
        W, b, d = prm
        n = x.shape[1]
        xb = x.to(BF)
        if st is None:
            return torch.matmul(xb, W[0]).float() + b, None
        full = torch.cat([st, xb], 1)
        y = None
        for j in range(W.shape[0]):
            t = torch.matmul(full[:, j * d : j * d + n], W[j]).float()
            y = t if y is None else y + t
        return y + b, full[:, -st.shape[1] :]

    @staticmethod
    def _tconv(x, prev, prm):
        """Transposed conv (stride r, kernel r or 2r, trimmed causally): x [B, n, Cin] FP32."""
        W, b, k, cout = prm
        B, n, _ = x.shape
        xb = x.to(BF)
        if prev is None:  # kernel == stride
            z = torch.matmul(xb, W).float().view(B, n * k, cout)
            return z + b, None
        r = k // 2
        full = torch.cat([prev, xb], 1)  # [B, n + 1, Cin]
        z = torch.matmul(full, W).float().view(B, n + 1, k, cout)
        y = z[:, 1:, :r] + z[:, :-1, r:]
        return y.reshape(B, n * r, cout) + b, full[:, -1:]

    @staticmethod
    def _snake(x, prm):
        a, ib = prm
        return x + ib * torch.sin(x * a).pow(2)

    def _rms(self, x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

    def _ups(self, h, sts):
        """The upsamplers: h [B, 1, 1024] FP32 -> [B, 4, 1024]; sts their depthwise-conv contexts."""
        new = []
        for u, st in zip(self.up, sts):
            h, _ = self._tconv(h, None, u["t"])
            res = h
            dw, db = u["dw"]
            full = torch.cat([st.float(), h], 1)
            n = h.shape[1]
            y = db + sum(full[:, j : j + n] * dw[j] for j in range(dw.shape[0]))
            new.append(full[:, -st.shape[1] :].to(BF))
            y = F.layer_norm(y, (y.shape[-1],), u["ln"][0], u["ln"][1], eps=1e-6)
            y = F.linear(y.to(BF), u["pw1"][0]).float() + u["pw1"][1]
            y = F.gelu(y)
            y = F.linear(y.to(BF), u["pw2"][0]).float() + u["pw2"][1]
            h = res + u["gamma"] * y
        return h, new

    def _dac(self, h, sts, nki: bool = False, part: str = "all"):
        """decoder.0 .. conv_out: h [B, 4, 1024] FP32 -> (pcm [B, 1920] in [-1, 1], new contexts);
        sts the DAC's contexts in `ctx` order (None where a conv has none). With `nki` (the serving
        DAC graph on a NeuronCore) the residual stacks of `nki_blocks` run as the fused kernel.
        part "head": decoder.0 .. decoder.2 only (returns h2 [B, 160, 384] FP32 and the head's
        contexts); "tail": decoder.3 .. conv_out from h2 with the tail's contexts."""
        it = iter(sts)
        new = []

        def conv(x, prm):
            y, n = self._conv(x, next(it), prm)
            new.append(n)
            return y

        if part != "tail":
            h = conv(h, self.conv_in)
        for bi, b in enumerate(self.blocks):
            if (part == "head" and bi >= HEAD_BLOCKS) or (part == "tail" and bi < HEAD_BLOCKS):
                continue
            h = self._snake(h, b["act"])
            h, n = self._tconv(h, next(it), b["t"])
            new.append(n)
            if nki and bi in self.nki_blocks and h.device.type == "neuron":
                from .nki_dac import resstack_nki

                c1s = []
                for _ in b["units"]:
                    c1s.append(next(it))
                    assert next(it) is None  # conv2 (k=1) carries no context
                h, ns = resstack_nki(h, c1s, self.rs[self.nki_blocks.index(bi)])
                for n in ns:
                    new += [n, None]
                continue
            for u in b["units"]:
                r = h
                y = conv(self._snake(h, u["a1"]), u["c1"])
                y = conv(self._snake(y, u["a2"]), u["c2"])
                h = y + r
        if part == "head":
            return h, new
        h = conv(self._snake(h, self.final_act), self.conv_out)
        return h[:, :, 0].clamp(min=-1.0, max=1.0), new

    # ------------------------------------------------------------------ the two serving graphs
    def forward_xf(self, x, cos, sin, kbias, kidx, slots, fresh, K, V, *xctx):
        """pre_conv + transformer + upsamplers of B rows, the state updated IN PLACE, slot-indexed.

        x [B, 512], cos/sin [B, hd] FP32; kbias [B, RING] FP32 (0 on ring columns holding one of
        the row's previous 71 frames, else -inf); kidx [B * H] int64, the flat row of K/V that this
        frame's key/value goes to (slot, head, column pos % RING); slots [B] int64 (the scratch slot
        for rows with no frame); fresh [B] FP32 0/1. K, V [L, (NS + 1) * H * RING, hd] BF16: the
        key/value ring of every slot, keys post-RoPE. xctx: the pre_conv and upsampler contexts,
        each [NS + 1, ctx, C] BF16. Returns h [B, 4, 1024] FP32."""
        B = x.shape[0]
        nh, hd, L, W = self.nh, self.hd, self.L, self.ring
        keep = (1.0 - fresh).to(BF)[:, None, None]
        sts = [c.index_select(0, slots) * keep for c in xctx]
        mode = self.xf_attn
        if mode == "nki" and x.device.type != "neuron":
            mode = "mm"  # the kernel's function, off the NeuronCore
        if mode == "nki":  # the kernel gathers each (row, head)'s ring row itself: no gathered copy
            from .nki_xf import ring_attn_nki

            nsh = K.shape[1] // W  # table rows per layer: (NS + 1) * H
            K2, V2 = K.view(L * nsh, W * hd), V.view(L * nsh, W * hd)
            krow = (slots[:, None] * nh + torch.arange(nh, device=x.device)[None, :]).reshape(B * nh, 1).to(torch.int32)
            kbN = kbias[:, None, :].expand(B, nh, W).reshape(B * nh, W)
        else:
            Kr = K.view(L, -1, nh, W, hd).index_select(1, slots)  # [L, B, H, RING, hd]
            Vr = V.view(L, -1, nh, W, hd).index_select(1, slots)
        h, n0 = self._conv(x[:, None, :], sts[0], self.pre_conv)  # [B, 1, 1024]
        h = F.linear(h[:, 0].to(BF), self.in_w).float() + self.in_b  # [B, 512]
        c, s = cos[:, None, :], sin[:, None, :]
        scale = hd**-0.5
        ks, vs = [], []
        for li in range(L):
            r = h
            a = self._rms(h, self.ln1[li]).to(BF)
            q = F.linear(a, self.wq[li]).float().view(B, nh, hd)
            k = F.linear(a, self.wk[li]).float().view(B, nh, hd)
            v = F.linear(a, self.wv[li]).float().view(B, nh, hd)
            q = q * c + torch.cat((-q[..., hd // 2 :], q[..., : hd // 2]), -1) * s
            k = k * c + torch.cat((-k[..., hd // 2 :], k[..., : hd // 2]), -1) * s
            kb, vb = k.to(BF), v.to(BF)
            if mode == "nki":
                s_new = (q * kb.float()).sum(-1, keepdim=True) * scale
                o = ring_attn_nki(q.reshape(B * nh, hd), s_new.reshape(B * nh, 1), kbN, vb.float().reshape(B * nh, hd),
                                  K2, V2, krow + li * nsh).view(B, nh, hd)
            elif mode == "none":  # timing floor (bench only): no attention over the ring
                o = vb.float()
            else:
                Ko, Vo = Kr[li], Vr[li]  # [B, H, RING, hd]
                if mode == "mm":
                    s_old = torch.matmul(q[:, :, None, :], Ko.float().transpose(2, 3))[:, :, 0]
                else:  # vec / vecf: [B, H, 1, hd] * [B, H, RING, hd] summed over hd, in FP32
                    s_old = (q[:, :, None, :] * (Ko.float() if mode == "vecf" else Ko)).sum(-1)
                s_old = s_old * scale + kbias[:, None, :]
                s_new = (q * kb.float()).sum(-1, keepdim=True) * scale
                p = torch.softmax(torch.cat([s_old, s_new], -1), -1)
                if mode == "mm":
                    o = torch.matmul(p[:, :, None, :-1], Vo.float())[:, :, 0]
                else:  # [B, H, RING, 1] * [B, H, RING, hd] summed over RING, in FP32
                    o = (p[:, :, :-1, None] * (Vo.float() if mode == "vecf" else Vo)).sum(2)
                o = o + p[:, :, -1:] * vb.float()
            o = F.linear(o.reshape(B, nh * hd).to(BF), self.wo[li]).float()
            h = r + self.s_attn[li] * o
            r = h
            a = self._rms(h, self.ln2[li]).to(BF)
            m = F.linear((F.silu(F.linear(a, self.wg[li]).float()) * F.linear(a, self.wu[li]).float()).to(BF), self.wd[li]).float()
            h = r + self.s_mlp[li] * m
            ks.append(kb)
            vs.append(vb)
        # One in-place scatter per tensor: column pos % RING of each (row, head), every layer.
        idx = kidx.view(1, B * nh, 1).expand(L, B * nh, hd)
        K.scatter_(1, idx, torch.stack(ks, 0).reshape(L, B * nh, hd))
        V.scatter_(1, idx, torch.stack(vs, 0).reshape(L, B * nh, hd))
        h = self._rms(h, self.norm).to(BF)
        h = (F.linear(h, self.out_w).float() + self.out_b)[:, None, :]  # [B, 1, 1024]
        h, nu = self._ups(h, sts[1:])
        for t, n in zip(xctx, [n0] + nu):
            t.index_copy_(0, slots, n)
        return h

    def forward_dac(self, h, active, fresh, *gs):
        """The DAC of G = len(gs) whole slot groups (B = G * GROUP rows): h [B, 4, 1024] FP32,
        active/fresh [B] FP32 0/1, gs G x [GROUP, dac_state] BF16 (each group's DAC contexts, flat)
        -> (pcm [B, 1920] FP32, *G new group states); rows that are not active keep their state."""
        G = len(gs)
        cs = torch.cat(gs, 0) if G > 1 else gs[0]
        B = cs.shape[0]
        cs = cs * (1.0 - fresh).to(BF)[:, None]
        sts, off = [], 0
        for c, k in self.ctx[self.n_xf_ctx :]:
            sts.append(cs[:, off : off + c * k].view(B, k, c) if k else None)
            off += c * k
        pcm, new = self._dac(h, sts, nki=True)
        cs_new = torch.cat([t.reshape(B, -1) for t in new if t is not None], 1)
        cs_new = torch.where((active > 0)[:, None], cs_new, cs)
        return (pcm, *[cs_new[g * GROUP : (g + 1) * GROUP] for g in range(G)])

    def _unpack(self, cs, ctx, cf: bool = False):
        """A flat [B, N] context tensor -> the contexts of `ctx` ([B, k, C] channels-last, or
        [B, C, k] channels-first with cf; None where k = 0)."""
        B, sts, off = cs.shape[0], [], 0
        for c, k in ctx:
            if not k:
                sts.append(None)
                continue
            t = cs[:, off : off + c * k]
            sts.append(t.reshape(B, c, k) if cf else t.reshape(B, k, c))
            off += c * k
        return sts

    def tail_cf(self, device) -> bool:
        """Whether the tail runs as the NKI kernel (its contexts then stored channels-first)."""
        return self.nki_tail and torch.device(device).type == "neuron"

    def forward_dac_head(self, h, active, fresh, *gs, row0=None):
        """DAC head graph: decoder.0 .. decoder.2 of G whole slot groups: h [B, 4, 1024] FP32,
        gs G x [GROUP, head_state] BF16 -> (h2, *Gh new group states). h2 is [B, 384, 160] FP32
        channels-first when the tail is the NKI kernel, else [B, 160, 384].
        Head padding: when h has fewer rows than the groups hold (Bh = Gh * GROUP < B),
        the graph runs at B rows with h, active and fresh zero-padded (the extra groups are the
        caller's scratch states, never active), and returns h2 and the states of the first Gh
        groups only. A B=16 call thereby runs the B=32 head graph, which does not spill to HBM.
        Head split (experimental, QWEN3TTS_CODEC_HEAD_MAX): with `row0` (a constant of the compiled module) the graph takes rows
        row0 .. row0 + G * GROUP of h, active and fresh (h, active, fresh are the whole call's)."""
        G = len(gs)
        if row0 is not None:
            n = G * GROUP
            h, active, fresh = h[row0 : row0 + n], active[row0 : row0 + n], fresh[row0 : row0 + n]
        cs = torch.cat(gs, 0) if G > 1 else gs[0]
        B, Bh = cs.shape[0], h.shape[0]
        if Bh < B:
            z = torch.zeros(B - Bh, dtype=active.dtype, device=active.device)
            h = torch.cat([h, h.new_zeros((B - Bh,) + tuple(h.shape[1:]))], 0)
            active, fresh = torch.cat([active, z], 0), torch.cat([fresh, z], 0)
        cs = cs * (1.0 - fresh).to(BF)[:, None]
        nx = self.n_xf_ctx
        sts = self._unpack(cs, self.ctx[nx : nx + self.n_head_ctx])
        h2, new = self._dac(h, sts, part="head")
        cs_new = torch.cat([t.reshape(B, -1) for t in new if t is not None], 1)
        cs_new = torch.where((active > 0)[:, None], cs_new, cs)
        if Bh < B:
            h2 = h2[:Bh]
        if self.tail_cf(h.device):
            h2 = h2.transpose(1, 2).contiguous()
        return (h2, *[cs_new[g * GROUP : (g + 1) * GROUP] for g in range(Bh // GROUP)])

    def forward_dac_tail(self, h2, active, fresh, *gs):
        """DAC tail graph: decoder.3 .. conv_out of G whole slot groups: h2 as the head returns
        it, gs G x [GROUP, tail_state] BF16 -> (pcm [B, 1920] FP32, *G new group states). On a
        NeuronCore with NKI on, one fused kernel with the contexts channels-first; else the XLA
        code channels-last."""
        G = len(gs)
        cs = torch.cat(gs, 0) if G > 1 else gs[0]
        B = cs.shape[0]
        cs = cs * (1.0 - fresh).to(BF)[:, None]
        ctx = self.ctx[self.n_xf_ctx + self.n_head_ctx :]
        if self.tail_cf(h2.device):
            from .nki_dac import tail_nki

            pcm, new = tail_nki(h2, [t for t in self._unpack(cs, ctx, cf=True) if t is not None], self.tp)
        else:
            pcm, new = self._dac(h2, self._unpack(cs, ctx), nki=True, part="tail")
        cs_new = torch.cat([t.reshape(B, -1) for t in new if t is not None], 1)
        cs_new = torch.where((active > 0)[:, None], cs_new, cs)
        return (pcm, *[cs_new[g * GROUP : (g + 1) * GROUP] for g in range(G)])

    def forward_dac_tail_cat(self, *args, parts: int = 2):
        """Head split (experimental): the tail graph on h2 given as `parts` row blocks (one per head
        launch), concatenated inside the graph."""
        h2 = torch.cat(args[:parts], 0)
        return self.forward_dac_tail(h2, *args[parts:])

    def forward(self, x, cos, sin, kbias, active, fresh, *state):
        G = len(state) // 2
        kv = torch.cat(state[0::2], 0) if G > 1 else state[0]
        cs = torch.cat(state[1::2], 0) if G > 1 else state[1]
        B = kv.shape[0]
        keep = (1.0 - fresh).to(BF)
        kv = kv * keep[:, None, None, None, None, None]
        cs = cs * keep[:, None]
        # Unpack the conv contexts.
        sts, off = [], 0
        for c, k in self.ctx:
            sts.append(cs[:, off : off + c * k].view(B, k, c) if k else None)
            off += c * k
        si = iter(range(len(sts)))
        new_st = [None] * len(sts)

        def take():
            i = next(si)
            return i, sts[i]

        nh, hd, L = self.nh, self.hd, self.L
        h = x[:, None, :]  # [B, 1, 512]
        i, st = take()
        h, new_st[i] = self._conv(h, st, self.pre_conv)  # [B, 1, 1024]
        h = F.linear(h[:, 0].to(BF), self.in_w).float() + self.in_b  # [B, 512]
        c, s = cos[:, None, :], sin[:, None, :]
        scale = hd**-0.5
        kvs = []
        for li in range(L):
            r = h
            a = self._rms(h, self.ln1[li]).to(BF)
            q = F.linear(a, self.wq[li]).float().view(B, nh, hd)
            k = F.linear(a, self.wk[li]).float().view(B, nh, hd)
            v = F.linear(a, self.wv[li]).float().view(B, nh, hd)
            q = q * c + torch.cat((-q[..., hd // 2 :], q[..., : hd // 2]), -1) * s
            k = k * c + torch.cat((-k[..., hd // 2 :], k[..., : hd // 2]), -1) * s
            Ko, Vo = kv[:, li, 0], kv[:, li, 1]  # [B, H, 71, hd]
            kb, vb = k.to(BF), v.to(BF)
            s_old = torch.matmul(q[:, :, None, :], Ko.float().transpose(2, 3))[:, :, 0] * scale + kbias[:, None, :]  # [B, H, 71]
            s_new = (q * kb.float()).sum(-1, keepdim=True) * scale
            p = torch.softmax(torch.cat([s_old, s_new], -1), -1)
            o = torch.matmul(p[:, :, None, :-1], Vo.float())[:, :, 0] + p[:, :, -1:] * vb.float()
            o = F.linear(o.reshape(B, nh * hd).to(BF), self.wo[li]).float()
            h = r + self.s_attn[li] * o
            r = h
            a = self._rms(h, self.ln2[li]).to(BF)
            m = F.linear((F.silu(F.linear(a, self.wg[li]).float()) * F.linear(a, self.wu[li]).float()).to(BF), self.wd[li]).float()
            h = r + self.s_mlp[li] * m
            kvs.append(torch.stack([torch.cat([Ko[:, :, 1:], kb[:, :, None]], 2), torch.cat([Vo[:, :, 1:], vb[:, :, None]], 2)], 1))
        h = self._rms(h, self.norm).to(BF)
        h = (F.linear(h, self.out_w).float() + self.out_b)[:, None, :]  # [B, 1, 1024]
        nx = self.n_xf_ctx
        h, new_st[1:nx] = self._ups(h, sts[1:nx])
        pcm, new_st[nx:] = self._dac(h, sts[nx:])
        kv_new = torch.stack(kvs, 1)  # [B, L, 2, H, 71, hd]
        cs_new = torch.cat([t.reshape(B, -1) for t in new_st if t is not None], 1)
        on = active > 0
        kv_new = torch.where(on[:, None, None, None, None, None], kv_new, kv)
        cs_new = torch.where(on[:, None], cs_new, cs)
        out = [pcm]
        for g in range(G):
            out += [kv_new[g * GROUP : (g + 1) * GROUP], cs_new[g * GROUP : (g + 1) * GROUP]]
        return tuple(out)


def _raise_dynamo_limits() -> None:
    """Raise dynamo's per-code recompile limit (default 8). Each _Part has its own code object,
    but the limit was also hit with a shared one (in the startup self-test, and on the codec
    thread, where the main thread's setting did not apply: the log reported the limit as 8 after
    the main thread set 64). Called by DeviceCodec and again on the codec thread."""
    import torch._dynamo as _dynamo

    for k in ("recompile_limit", "cache_size_limit"):
        if hasattr(_dynamo.config, k):
            setattr(_dynamo.config, k, max(getattr(_dynamo.config, k), 64))


class _Part(nn.Module):
    """One of CodecStep's serving graphs (`forward_xf` or `forward_dac`) as a module to compile."""

    def __init__(self, step: CodecStep, fn: str, **kw):
        super().__init__()
        self.step, self.fn, self.kw = step, fn, kw  # kw: constants of the graph (head split)
        # A code object of its own per graph, so dynamo's per-code cache and recompile limit count
        # this graph's variants only (xf, head and tail at three buckets in one shared code object
        # hit the limit of 8 on the codec thread).
        f = _Part._forward
        tag = fn + "".join(f"_{k}{v}" for k, v in sorted(kw.items()))
        names = {"co_name": f"forward_{tag}"}
        if hasattr(f.__code__, "co_qualname"):  # Python 3.11+
            names["co_qualname"] = f"_Part.forward_{tag}"
        code = f.__code__.replace(**names)
        self.forward = types.MethodType(types.FunctionType(code, f.__globals__, code.co_name, f.__defaults__, f.__closure__), self)

    def _forward(self, *args):
        return getattr(self.step, self.fn)(*args, **self.kw)


class DeviceCodec:
    """Slot table and compiled per-bucket graphs of the batched streaming decoder on one device.

    `open()` returns a slot, `decode({slot: codes [16]})` decodes one frame for each given slot in
    as few calls as the buckets allow and returns {slot: 1920 FP32 samples}, `close(slot)` frees it.
    Not thread-safe: one thread owns it.
    """

    def __init__(self, cw: CodecWeights, device: str = "cpu", groups=(2, 4), n_groups: int | None = None, compile_: bool | None = None, log=print):
        self.tab = CodecTables(cw)
        self.device = torch.device(device)
        self.step_mod = s = CodecStep(cw).to(self.device).eval()
        if compile_ is None:
            compile_ = self.device.type == "neuron"
        cc = (lambda m: torch.compile(m, backend="neuron", fullgraph=True, dynamic=False)) if compile_ else (lambda m: m)
        if compile_:
            _raise_dynamo_limits()
        self.xf, self.dac = cc(_Part(s, "forward_xf")), cc(_Part(s, "forward_dac"))  # dac: the one-graph DAC (bench A/B)
        # The serving DAC is two graphs, head (decoder.0 .. 2) and tail (decoder.3 .. conv_out).
        self.dac_head, self.dac_tail = cc(_Part(s, "forward_dac_head")), cc(_Part(s, "forward_dac_tail"))
        self._cc = cc
        # Buckets in slot groups. No 8-stream bucket by default: its transformer graph compiles to
        # slow code (measured on trn2: 18.5 ms at B=8 vs 2.1 ms at B=16), so a call of one group
        # pads to two.
        self.buckets = sorted(groups)
        # More slot groups than the largest bucket: a tick then takes several calls.
        self.n_groups = max(n_groups or self.buckets[-1], self.buckets[-1])
        self.slots = self.n_groups * GROUP
        self.scratch = self.slots  # the K/V ring and xf-context row of rows with no frame
        dev, ns = self.device, self.slots + 1
        self.K = torch.zeros((s.L, ns * s.nh * s.ring, s.hd), dtype=BF, device=dev)  # [L, NS+1, H, RING, hd] flat
        self.V = torch.zeros_like(self.K)
        self.xctx = [torch.zeros((ns, k, c), dtype=BF, device=dev) for c, k in s.ctx[: s.n_xf_ctx]]
        # Serving DAC: head + tail graphs (QWEN3TTS_CODEC_DAC_SPLIT=0: the one-graph DAC, a
        # benchmark control) and head calls padded to the next bucket of >= HEAD_MIN groups
        # (QWEN3TTS_CODEC_HEAD_PAD=0 off): at B=16 the head graph spills to HBM (6.38 ms vs
        # 3.82 ms at B=32), so a smaller call runs the B=32 head graph with scratch pad groups.
        # On a NeuronCore the split needs the NKI tail: the XLA tail as its own graph fails to
        # compile at B=32 (neuronx-cc NCC_IBIR228, state buffer overflow), so
        # QWEN3TTS_CODEC_NKI=0 falls back to the one-graph DAC there.
        self.split = os.environ.get("QWEN3TTS_CODEC_DAC_SPLIT", "1") != "0" and (self.device.type != "neuron" or s.tail_cf(self.device))
        self.head_pad = self.split and os.environ.get("QWEN3TTS_CODEC_HEAD_PAD", "1") != "0"
        if self.split:
            self.hstate = [torch.zeros((GROUP, s.head_state), dtype=BF, device=dev) for _ in range(self.n_groups)]
            self.tstate = [torch.zeros((GROUP, s.tail_state), dtype=BF, device=dev) for _ in range(self.n_groups)]
            npad = max(self.head_groups(b) - b for b in self.buckets)
            self.hpad = [torch.zeros((GROUP, s.head_state), dtype=BF, device=dev) for _ in range(npad)]  # never active
        else:
            self.gstate = [torch.zeros((GROUP, s.dac_state), dtype=BF, device=dev) for _ in range(self.n_groups)]
        # Head split (experimental): groups per head launch; the launches' graphs by first row.
        self.head_max = HEAD_MAX if self.split and HEAD_MAX > 0 else 0
        self.dac_head_at, self.dac_tail_cat = {}, {}
        self.free = list(range(self.slots))  # lowest first, so streams pack into few groups
        self.pos = {}  # slot -> frames decoded so far
        self.log = log
        # Per-call sub-phase totals (s) since creation, wall then codec-thread CPU, of
        # SPLIT_PH (prep incl. H2D, xf launch, DAC head, DAC tail, .cpu() D2H), and the call count.
        self.ph = np.zeros(2 * len(SPLIT_PH))
        self.ph_calls = 0
        self._mk = (0.0, 0.0)
        if s.nki_blocks and self.device.type == "neuron":
            from .nki_dac import wrapped

            wrapped(2)  # register the fused DAC kernels before the DAC graphs are traced
            wrapped(2, "tail")
        if s.xf_attn == "nki" and self.device.type == "neuron":
            from .nki_xf import wrapped as xf_wrapped

            xf_wrapped(2)  # the ring-attention kernel, before the xf graphs are traced

    def xf_attn_selftest(self, scales=(1.0, 4.0), min_db: float = XF_NKI_SELFTEST_DB) -> float | None:
        """Safeguard: run the NKI ring-attention kernel (engine.nki_xf) alone on the device
        at the largest bucket (B rows x H heads), against ring tables of the serving shape filled
        with seeded random BF16 values, slots in shuffled order (the scratch slot included), the
        first and last layer, key biases masking a random number of ring columns per row (as at
        the start of a stream), at two query scales (4.0 makes the softmax peaked). Compares with
        the FP32 CPU form of the `mm` path (nki_xf.ring_attn_ref). Exits the process non-zero on
        an error, a NaN or an SNR below `min_db` (a wrong row, layer or layout gives < 10 dB).
        Returns the worst SNR (dB); None when the kernel is not used."""
        s = self.step_mod
        if not (s.xf_attn == "nki" and self.device.type == "neuron"):
            return None
        from .nki_xf import ring_attn_nki, ring_attn_ref

        B, nh, hd, W, L = self.buckets[-1] * GROUP, s.nh, s.hd, s.ring, s.L
        nsh = self.K.shape[1] // W
        g = torch.Generator().manual_seed(29)
        K2 = torch.randn(L * nsh, W * hd, generator=g).to(BF)
        V2 = torch.randn(L * nsh, W * hd, generator=g).to(BF)
        K2d, V2d = K2.to(self.device), V2.to(self.device)
        slots = torch.randperm(self.slots + 1, generator=g)[:B]
        nmask = torch.randint(0, W, (B,), generator=g)
        kbias = torch.where(torch.arange(W)[None, :] < nmask[:, None], torch.finfo(torch.float32).min, 0.0)
        kbN = kbias[:, None, :].expand(B, nh, W).reshape(B * nh, W).contiguous()
        krow = (slots[:, None] * nh + torch.arange(nh)[None, :]).reshape(B * nh, 1).to(torch.int32)
        worst = math.inf
        with torch.inference_mode():
            for scale in scales:
                for li in (0, L - 1):
                    q = torch.randn(B * nh, hd, generator=g) * scale
                    snew = torch.randn(B * nh, 1, generator=g) * scale
                    vnew = torch.randn(B * nh, hd, generator=g).to(BF).float()
                    idx = krow + li * nsh
                    ref = ring_attn_ref(q, snew, kbN, vnew, K2, V2, idx)
                    try:
                        got = ring_attn_nki(q.to(self.device), snew.to(self.device), kbN.to(self.device), vnew.to(self.device),
                                            K2d, V2d, idx.to(self.device)).float().cpu()
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        self.log(f"[selftest] NKI xf ring-attention kernel FAILED B={B}: {type(e).__name__}: {e}")
                        os._exit(2)
                    v = snr_db(ref.numpy(), got.numpy()) if bool(torch.isfinite(got).all()) else -math.inf
                    worst = min(worst, v)
                    if not v >= min_db:
                        j = int((got - ref).abs().nan_to_num(math.inf).argmax())
                        self.log(f"[selftest] NKI xf ring-attention kernel FAILED B={B} layer {li} scale {scale}: SNR {v:.1f} dB (bar {min_db})\n"
                                 f" kernel: {got.flatten()[max(j - 3, 0) : j + 3].tolist()}\n FP32 CPU: {ref.flatten()[max(j - 3, 0) : j + 3].tolist()}")
                        os._exit(2)
                    self.log(f"[selftest] NKI xf ring-attention kernel B={B} layer {li} scale {scale}: SNR vs FP32 CPU {v:.1f} dB")
        del K2d, V2d
        return worst

    def nki_selftest(self, cw: CodecWeights, scales=(0.3, 3.0), min_db: float = NKI_SELFTEST_DB, ctx_db_large: float = 25.0) -> float:
        """Safeguard: run the fused DAC tail kernel (decoder.3 .. conv_out, which holds the fused
        residual stacks) alone on the device at the largest bucket (B rows), on fixed
        seeded inputs at two scales with non-zero contexts, and compare its pcm and its nine
        returned contexts with the FP32 CPU tail (nki_dac.tail_ref). At 3.0 decoder.4's output
        reaches |x| ~ 600 and the snake arguments |a x| the hundreds, so the sin range reduction
        is exercised. Bars: pcm >= `min_db` at both scales; contexts >= `min_db` at the small
        scale and >= `ctx_db_large` at the large one (the BF16 XLA tail itself gives 29-33 dB on
        the last contexts there; a wrong tap, offset or layout gives < 10 dB). Exits the process
        non-zero below a bar or on an error. Returns the worst pcm SNR (dB); None when unused.
        First runs xf_attn_selftest (the xf ring-attention kernel), result in xf_selftest_db."""
        s = self.step_mod
        self.xf_selftest_db = self.xf_attn_selftest()
        if not s.tail_cf(self.device):
            return None
        from .nki_dac import tail_ctx_shapes, tail_nki, tail_ref

        B = self.buckets[-1] * GROUP
        g = torch.Generator().manual_seed(20)
        shapes = tail_ctx_shapes(s)
        worst = math.inf
        with torch.inference_mode():
            for si, scale in enumerate(scales):
                h2 = torch.randn(B, 384, 160, generator=g) * scale
                ctx = [(torch.randn(B, c, k, generator=g) * scale).to(BF) for c, k in shapes]
                pcm_ref, n_ref, ymax, h4max = tail_ref(cw, h2, ctx)
                try:
                    pcm, new = tail_nki(h2.to(self.device), [c.to(self.device) for c in ctx], s.tp)
                    got = [pcm.float().cpu()] + [t.float().cpu() for t in new]
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    self.log(f"[selftest] NKI DAC tail kernel FAILED B={B}: {type(e).__name__}: {e}")
                    os._exit(2)
                v = snr_db(pcm_ref.numpy(), got[0].numpy())
                vc = [snr_db(a.numpy(), b.numpy()) for a, b in zip(n_ref, got[1:])]
                worst = min(worst, v)
                cbar = min_db if si == 0 else ctx_db_large
                if not (v >= min_db and min(vc) >= cbar):
                    j = int((got[0] - pcm_ref).abs().argmax())
                    self.log(f"[selftest] NKI DAC tail kernel FAILED B={B} scale {scale}: pcm SNR {v:.1f} dB (bar {min_db}), contexts {[round(t, 1) for t in vc]} dB (bar {cbar})\n"
                             f" kernel: {got[0].flatten()[max(j - 3, 0) : j + 3].tolist()}\n FP32 CPU: {pcm_ref.flatten()[max(j - 3, 0) : j + 3].tolist()}")
                    os._exit(2)
                self.log(f"[selftest] NKI DAC tail kernel B={B} scale {scale} (max |decoder.4 out| {h4max:.0f}, |pre-clamp| {ymax:.2f}): "
                         f"pcm SNR vs FP32 CPU {v:.1f} dB, contexts {[round(t, 1) for t in vc]} dB")
        return worst

    @property
    def max_batch(self) -> int:
        return self.slots

    def open(self) -> int:
        if not self.free:
            raise RuntimeError("no free codec slot")
        self.free.sort()
        s = self.free.pop(0)
        self.pos[s] = 0
        return s

    def close(self, slot: int) -> None:
        if slot in self.pos:
            del self.pos[slot]
            self.free.append(slot)

    @torch.inference_mode()
    def decode(self, frames: dict) -> dict:
        """{slot: codes [16]} -> {slot: np.float32 [1920]}: one frame per slot."""
        out = {}
        by_group: dict[int, list[int]] = {}
        for s in frames:
            by_group.setdefault(s // GROUP, []).append(s)
        todo = sorted(by_group)
        while todo:
            n = next((b for b in self.buckets if b >= len(todo)), self.buckets[-1])
            take, todo = todo[:n], todo[n:]
            # Pad with groups that have nothing to decode (their state passes through unchanged).
            pad = [g for g in range(self.n_groups) if g not in take and g not in todo][: n - len(take)]
            groups = take + pad
            out.update(self._call(groups, {s: frames[s] for g in take for s in by_group[g]}))
        return out

    def _mark(self, i: int) -> None:
        """Profiling: add the time since the previous mark to sub-phase i."""
        w, c = time.perf_counter(), time.thread_time()
        self.ph[i] += w - self._mk[0]
        self.ph[len(SPLIT_PH) + i] += c - self._mk[1]
        self._mk = (w, c)

    def _call(self, groups: list[int], frames: dict) -> dict:
        self._mk = (time.perf_counter(), time.thread_time())
        self.ph_calls += 1
        rows, args = self._prep(groups, frames)
        self._mark(0)
        pcm = self._run(groups, args)
        out = {}
        for sl, r in rows.items():
            out[sl] = pcm[r]
            self.pos[sl] += 1
        return out

    def _prep(self, groups: list[int], frames: dict):
        """Host inputs of one call, on the device: ({slot: row}, [x, cos, sin, kbias, kidx, slots, fresh, active])."""
        s = self.step_mod
        B, H, W = len(groups) * GROUP, s.nh, s.ring
        codes = np.zeros((B, 16), dtype=np.int64)
        pos = np.zeros(B, dtype=np.int64)
        slots = np.full(B, self.scratch, dtype=np.int64)
        active = torch.zeros(B)
        rows = {}
        for gi, g in enumerate(groups):
            for j in range(GROUP):
                sl = g * GROUP + j
                if sl in frames:
                    r = gi * GROUP + j
                    rows[sl] = r
                    codes[r] = frames[sl]
                    pos[r] = self.pos[sl]
                    slots[r] = sl
                    active[r] = 1.0
        fresh = active * torch.as_tensor(pos == 0, dtype=torch.float32)
        x = self.tab.quantize(codes)
        cos, sin, _ = self.tab.rope(pos)
        p = torch.as_tensor(pos)[:, None]
        j = torch.arange(W)[None, :]
        f = p - 1 - torch.remainder(p - 1 - j, W)  # the frame held by ring column j before this one
        kbias = torch.where((f >= 0) & (j != p % W), 0.0, torch.finfo(torch.float32).min)
        sl_t = torch.as_tensor(slots)
        kidx = (sl_t[:, None] * (H * W) + torch.arange(H)[None, :] * W + torch.remainder(p, W)).reshape(-1)
        dev = self.device
        return rows, [t.to(dev) for t in (x, cos, sin, kbias, kidx, sl_t, fresh, active)]

    def head_groups(self, n: int) -> int:
        """Groups the head graph of an n-group call runs at: n, or with head padding the smallest
        bucket >= max(n, HEAD_MIN) (the next bucket, not the largest; the largest when no
        bucket reaches HEAD_MIN)."""
        if not (self.split and self.head_pad):
            return n
        return next((b for b in self.buckets if b >= max(n, HEAD_MIN)), max(n, self.buckets[-1]))

    def head_states(self, groups: list[int]) -> list:
        """The head graph's state inputs of a call on `groups`: theirs, then scratch pad groups up
        to `head_groups` when head padding is on."""
        hs = [self.hstate[g] for g in groups]
        return hs + self.hpad[: self.head_groups(len(groups)) - len(groups)]

    def _head(self, groups: list[int], h, active, fresh) -> list:
        """The DAC head of one call; updates the groups' head state; returns h2 as row blocks (one
        per head launch)."""
        hm = self.head_max
        if not hm or len(groups) <= hm:
            res = self.dac_head(h, active, fresh, *self.head_states(groups))
            for gi, g in enumerate(groups):
                self.hstate[g] = res[1 + gi]
            return [res[0]]
        assert len(groups) % hm == 0, (len(groups), hm)
        out = []
        for c in range(0, len(groups), hm):
            r0 = c * GROUP
            if r0 not in self.dac_head_at:
                self.dac_head_at[r0] = self._cc(_Part(self.step_mod, "forward_dac_head", row0=r0))
            res = self.dac_head_at[r0](h, active, fresh, *[self.hstate[g] for g in groups[c : c + hm]])
            for gi, g in enumerate(groups[c : c + hm]):
                self.hstate[g] = res[1 + gi]
            out.append(res[0])
        return out

    def _tail(self, groups: list[int], h2s: list, active, fresh):
        """The DAC tail of one call on h2 row blocks; updates the groups' tail state; returns pcm."""
        ts = [self.tstate[g] for g in groups]
        if len(h2s) == 1:
            res = self.dac_tail(h2s[0], active, fresh, *ts)
        else:
            k = len(h2s)
            if k not in self.dac_tail_cat:
                self.dac_tail_cat[k] = self._cc(_Part(self.step_mod, "forward_dac_tail_cat", parts=k))
            res = self.dac_tail_cat[k](*h2s, active, fresh, *ts)
        for gi, g in enumerate(groups):
            self.tstate[g] = res[1 + gi]
        return res[0]

    def _run(self, groups: list[int], args) -> np.ndarray:
        """The two graphs of one call; updates the groups' state; returns pcm [B, 1920] FP32."""
        x, cos, sin, kbias, kidx, sl_t, fresh, active = args
        h = self.xf(x, cos, sin, kbias, kidx, sl_t, fresh, self.K, self.V, *self.xctx)
        self._mark(1)
        if not self.split:
            res = self.dac(h, active, fresh, *[self.gstate[g] for g in groups])
            for gi, g in enumerate(groups):
                self.gstate[g] = res[1 + gi]
            self._mark(2)
            out = res[0].float().cpu().numpy()
            self._mark(4)
            return out
        h2s = self._head(groups, h, active, fresh)
        self._mark(2)
        pcm = self._tail(groups, h2s, active, fresh)
        self._mark(3)
        out = pcm.float().cpu().numpy()
        self._mark(4)
        return out

    def warm(self) -> None:
        """Compile (or load) every bucket's graph by one call with no active slot."""
        for n in self.buckets:
            self._call(list(range(n)), {})


# Scheduler knobs (env overrides for tuning; the defaults are the measured configuration).
TTFA_S = float(os.environ.get("QWEN3TTS_CODEC_TTFA_S", "0.2"))  # first-frame deadline after arrival
FRAME_S = SAMPLES_PER_FRAME / 24000  # 80 ms of audio a frame
WAIT_S = float(os.environ.get("QWEN3TTS_CODEC_WAIT_S", "0.015"))  # longest wait for a call to fill
MARGIN_S = float(os.environ.get("QWEN3TTS_CODEC_MARGIN_S", "0.04"))  # wait only if nothing is due within one call + this
DENSE = int(os.environ.get("QWEN3TTS_CODEC_DENSE", "24"))  # active slots that make a call worth taking now
QUANT_S = 0.02  # deadlines this close count as a tie (the denser group goes first)
STATS_S = float(os.environ.get("QWEN3TTS_CODEC_STATS_S", "10"))
LATE_K = ((1, 1), (2, 2), (3, 5), (6, 1 << 30))  # stats: frame-index bins of the late histogram
LATE_MS = (5.0, 10.0, 20.0, math.inf)  # and lateness bins (upper edges, ms)
SPLIT_PH = ("prep", "xf", "head", "tail", "d2h")  # profiling: DeviceCodec call sub-phases
FPS_ALPHA, FPS_FILL, FPS_FLOOR, FPS_GAP_S = 0.02, 0.75, 0.5, 0.003  # CodecService.fps_cap estimator


_NO_END = object()


def _fanout(items: list) -> None:
    """On the event loop: one codec call's items onto their async streams' queues."""
    for st, item in items:
        st.aq.put_nowait(item)


def late_hist_add(s: dict, st) -> None:
    """Count an ended stream in the stats window's late histogram."""
    if st.max_late <= 0:
        s["ok"] += 1
        return
    ki = next(i for i, (lo, hi) in enumerate(LATE_K) if lo <= st.max_late_k <= hi)
    li = next(i for i, e in enumerate(LATE_MS) if 1000 * st.max_late < e)
    s["hist"][ki, li] += 1


def late_hist_str(h, ok: int) -> str:
    ks = ["k1", "k2", "k3-5", "k6+"]
    ls = ["<5", "5-10", "10-20", "20+"]
    rows = "; ".join(f"{k} " + "/".join(str(int(v)) for v in h[i]) for i, k in enumerate(ks))
    return f"streams ended {ok + int(h.sum())}, never late {ok}; worst late frame k x ms {'/'.join(ls)}: {rows}"


class CodecStream:
    """One utterance on the codec thread: `push(codes)` frames in, PCM16 byte chunks out of `out`."""

    def __init__(self, svc: "CodecService"):
        self.svc = svc
        self.frames: list = []  # pending frames, guarded by svc.cv
        self.ended = False  # no more frames will come
        self.aborted = False
        self.slot: int | None = None
        self.out: queue.Queue = queue.Queue()  # bytes, then None (end) or an Exception
        self.t_arrival = time.time()  # when the stream was opened (the request's submit)
        self.t_push: float | None = None  # when the first frame was pushed
        self.t_first_out: float | None = None  # when the first PCM went on `out`
        self.delivered = 0  # frames whose PCM is on `out`
        self.codec_ms: float | None = None  # the device call of the first chunk, incl. lock wait
        self.first_frames = 0  # frames in the first chunk
        self.late = False  # some frame went out after its deadline
        # First-audio hold: the server releases the first PCM no earlier than `hold_until` (None = no
        # hold), so the player's clock starts at max(t_first_out, hold_until) and frame k is due
        # 80 ms x k after that release.
        self.hold_until: float | None = None
        self.held_ms = 0.0  # how long the first chunk waited for the release (set by chunks())
        self.max_late = 0.0  # worst lateness (s) of a frame against its deadline, and its index
        self.max_late_k = 0
        # An async stream (bind_async) gets its items on `aq`, an asyncio.Queue of the
        # event loop `loop`, one loop callback per codec call (CodecService._put); `out` is unused.
        self.aq: asyncio.Queue | None = None
        self.loop = None

    def anchor(self) -> float | None:
        """When the player starts (the release of the first PCM), None before the first PCM."""
        if self.t_first_out is None:
            return None
        return self.t_first_out if self.hold_until is None else max(self.t_first_out, self.hold_until)

    def deadline(self) -> float:
        """When this stream's next frame is due: the TTFA budget before the first frame, then the
        moment a player that started at the first byte (the release) runs dry."""
        if self.t_first_out is None:
            return self.t_arrival + TTFA_S
        return self.anchor() + FRAME_S * self.delivered

    def push(self, codes) -> None:
        with self.svc.cv:
            if self.t_push is None:
                self.t_push = time.time()
            self.frames.append(codes)
            self.svc.backlog += 1
            self.svc.cv.notify()

    def finish(self) -> None:
        with self.svc.cv:
            self.ended = True
            self.svc.cv.notify()

    def abort(self) -> None:
        with self.svc.cv:
            self.aborted = True
            self.svc.cv.notify()

    def bind_async(self, loop) -> None:
        """Deliver this stream's items to an asyncio.Queue on `loop` (consumed by
        `achunks` on that loop). Call on the loop, before the stream's first frame is pushed."""
        self.aq, self.loop = asyncio.Queue(), loop

    def fail(self, exc: BaseException) -> None:
        """Put an error on the stream from any thread (the consumer raises it)."""
        if self.aq is None:
            self.out.put(exc)
        else:
            try:
                self.loop.call_soon_threadsafe(self.aq.put_nowait, exc)
            except RuntimeError:  # the loop is closed: nobody consumes
                pass

    async def achunks(self):
        """`chunks` for an async stream (bind_async), on its event loop: the same chunks, the
        same first-chunk hold (an asyncio sleep) and merge, the same end and errors."""
        aq = self.aq
        try:
            item = await aq.get()
            if self.hold_until is not None and isinstance(item, bytes):
                wait = self.hold_until - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                    self.held_ms = 1000 * wait
                parts, end = [item], _NO_END
                while True:
                    try:
                        nxt = aq.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not isinstance(nxt, bytes):
                        end = nxt
                        break
                    parts.append(nxt)
                self.first_frames = len(parts)
                yield b"".join(parts) if len(parts) > 1 else item
                if end is not _NO_END:
                    if isinstance(end, Exception):
                        raise end
                    return
                item = await aq.get()
            while True:
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
                item = await aq.get()
        finally:
            self.abort()

    def chunks(self):
        """Yield PCM16 byte chunks until the utterance ends (raises the codec thread's error).
        First-audio hold: the first chunk waits until `hold_until` (when set and later than its PCM), then
        carries every PCM already on `out` with it, so frames decoded during the hold go out as
        player buffer at once."""
        try:
            item = self.out.get()
            if self.hold_until is not None and isinstance(item, bytes):
                wait = self.hold_until - time.time()
                if wait > 0:
                    time.sleep(wait)
                    self.held_ms = 1000 * wait
                parts, end = [item], _NO_END
                while True:
                    try:
                        nxt = self.out.get_nowait()
                    except queue.Empty:
                        break
                    if not isinstance(nxt, bytes):
                        end = nxt
                        break
                    parts.append(nxt)
                self.first_frames = len(parts)
                yield b"".join(parts) if len(parts) > 1 else item
                if end is not _NO_END:  # the end marker (None) or an error was right behind
                    if isinstance(end, Exception):
                        raise end
                    return
                item = self.out.get()
            while True:
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
                item = self.out.get()
        finally:
            self.abort()


class CodecService:
    """The codec thread: one device call at a time, each decoding one frame (the oldest pending)
    of every pending stream in up to `max_groups` slot groups, chosen earliest deadline first
    (`CodecStream.deadline`); each stream gets its PCM as soon as the call returns. When nothing is
    due soon, no stream is fresh and the call would be sparse, it waits up to WAIT_S for more
    frames so the call fills. `lock` (shared with the language model when both use one NeuronCore)
    is held around each device call."""

    def __init__(self, dc: DeviceCodec, lock=None, log=print, start: bool = True):
        self.dc, self.lock, self.log = dc, lock or threading.Lock(), log
        self.cv = threading.Condition()
        self.streams: list[CodecStream] = []
        self.max_groups = dc.buckets[-1]
        self.ticks = self.frames = 0  # device calls, frames decoded
        self.last_call_s = 0.03
        self._st = self._zero_stats()
        # For the admission gate (engine.admit): frames pushed but not yet decoded (exact after
        # each _reap), and the codec's capacity in frames per busy second: an EWMA (FPS_ALPHA per
        # call) over calls at >= FPS_FILL of the largest bucket, of frames / (call + host pick and
        # post time, excluding fill and idle waits). Seeded at startup from timed full-bucket calls
        # plus the host gap (_warm_thread); never below FPS_FLOOR x that seed.
        self.backlog = 0
        self.fps_cap = self.fps_seed = float(os.environ.get("QWEN3TTS_CODEC_FPS0", "1500"))
        # For engine.admit.DeadlineGate: the last 64 calls as (active slots, slot groups,
        # cycle s = call + host pick and post, excluding fill and idle waits), appended by _run.
        self.calls: collections.deque = collections.deque(maxlen=64)
        # QWEN3TTS_CODEC_PIPE=1: the codec thread converts a call's PCM in one array op and hands
        # every write to a stream's `out` (PCM, the end marker, errors) to one FIFO delivery thread,
        # in order, with one put per call; =0: the inline per-stream path (comparison control).
        self.pipe = os.environ.get("QWEN3TTS_CODEC_PIPE", "1") == "1"
        self.dq: queue.SimpleQueue = queue.SimpleQueue()
        # Experimental: QWEN3TTS_ASYNC_DIRECT=1 hands a call's items to the streams (async: one
        # loop callback per call) from the codec thread itself, without the delivery thread.
        self.direct = os.environ.get("QWEN3TTS_ASYNC_DIRECT", "0") == "1"
        if start:
            self.start()

    def start(self) -> None:
        # Process-wide GIL switch interval (Python's default is 5 ms). The codec thread re-takes
        # the GIL after each device call from ~100 HTTP handler threads; at 5 ms that wait added
        # ~5 ms to the call and ~3 ms to the PCM hand-off at saturation (measured on trn2).
        sw = float(os.environ.get("QWEN3TTS_CODEC_SWITCH_S", "0.0005"))
        if sw != sys.getswitchinterval():
            sys.setswitchinterval(sw)
        self.log(f"[codec] GIL switch interval {1000 * sys.getswitchinterval():.2f} ms, delivery {'thread' if self.pipe else 'inline'}")
        if self.pipe:
            threading.Thread(target=self._deliver, daemon=True, name="codec-out").start()
        threading.Thread(target=self._run, daemon=True, name="codec").start()

    def _send(self, items: list) -> None:
        """Codec thread: [(stream, bytes | None | Exception)] to the streams' `out`, in order."""
        if self.pipe and not self.direct:
            self.dq.put(items)
            d = self.dq.qsize()
            if d > self._st["dq"]:
                self._st["dq"] = d
        else:
            self._put(items)

    def _put(self, items: list) -> None:
        """Delivery thread: one call's [(stream, item)] onto the streams' `out`, in order. The
        items of async streams go to their event loop as ONE callback per loop
        (`_fanout`), which puts them on the streams' asyncio queues in the same order."""
        by_loop = None
        for st, item in items:
            if st.aq is None:
                st.out.put(item)
            else:
                if by_loop is None:
                    by_loop = {}
                by_loop.setdefault(st.loop, []).append((st, item))
        if by_loop:
            for loop, its in by_loop.items():
                try:
                    loop.call_soon_threadsafe(_fanout, its)
                except RuntimeError:  # the loop is closed (shutdown): nobody consumes
                    pass

    def _deliver(self) -> None:
        """The delivery thread: FIFO, so a stream's end marker never overtakes its last PCM."""
        try:
            while True:
                self._put(self.dq.get())
        except BaseException:  # noqa: BLE001 - streams would block forever without it
            traceback.print_exc()
            self.log("[codec] delivery thread died; exiting")
            os._exit(3)

    def open(self) -> CodecStream:
        st = CodecStream(self)
        with self.cv:
            self.streams.append(st)
            self.cv.notify()
        return st

    @staticmethod
    def _zero_stats():
        # Per-call phases (wall and codec-thread CPU, s): pick = reap/pick work under the cv
        # (excluding cv waits), call = dc.decode incl. the lock, post = bookkeeping + PCM16 + puts;
        # gap = end of post to the next call's start (pick + fill waits + idle waits).
        # hist: streams ended in the window by the index k of their latest frame (vs
        # the release-anchored deadline) x that worst lateness; `ok` = never late.
        return dict(t0=time.time(), calls=0, active=0, busy=0.0, waited=0.0, idle=0.0, late=0, first=[],
                    hist=np.zeros((len(LATE_K), len(LATE_MS)), dtype=np.int64), ok=0, pick=0.0, pick_c=0.0, call_c=0.0, post=0.0, post_c=0.0, gap=0.0, waiters=0, dq=0)

    def _stats(self, now: float) -> None:
        s = self._st
        dt = now - s["t0"]
        if dt < STATS_S:
            return
        if s["calls"]:
            f = np.array(s["first"]) if s["first"] else np.zeros(1)
            self.log(
                f"[codec] {s['calls'] / dt:.1f} calls/s, {s['active'] / s['calls']:.1f} active slots/call, {s['active'] / dt:.0f} frames/s, "
                f"device busy {100 * s['busy'] / dt:.0f}%, fill waits {1000 * s['waited'] / dt:.0f} ms/s, late frames {s['late']}, "
                f"first frame push->pcm ms p50 {np.median(f):.0f} p99 {np.quantile(f, 0.99):.0f} (n {len(s['first'])}), "
                f"streams {len(self.streams)}"
            )
            n = s["calls"]
            self.log(
                f"[codec] phases ms/call (wall/cpu): pick {1000 * s['pick'] / n:.2f}/{1000 * s['pick_c'] / n:.2f}, "
                f"call {1000 * s['busy'] / n:.2f}/{1000 * s['call_c'] / n:.2f}, post {1000 * s['post'] / n:.2f}/{1000 * s['post_c'] / n:.2f}, "
                f"gap {1000 * s['gap'] / n:.2f} (fill wait {1000 * s['waited'] / n:.2f}, idle {1000 * s['idle'] / n:.2f}); "
                f"cycle {1000 * dt / n:.2f} ms; slot waiters/call {s['waiters'] / n:.1f}; delivery queue max {s['dq']}"
            )
        if s["ok"] or s["hist"].any():
            self.log(f"[codec] {late_hist_str(s['hist'], s['ok'])}")
        self._st = self._zero_stats()

    def _split_log(self, now: float) -> None:
        """Profiling: every STATS_S, the device call's sub-phases (DeviceCodec.ph), ms per call."""
        dc = self.dc
        last = getattr(self, "_split_last", None)
        if last is None or now - last[0] < STATS_S:
            if last is None:
                self._split_last = (now, dc.ph.copy(), dc.ph_calls)
            return
        n = dc.ph_calls - last[2]
        if n:
            d = 1000 * (dc.ph - last[1]) / n
            k = len(SPLIT_PH)
            self.log("[codec] call split ms/call (wall/cpu): " + ", ".join(f"{p} {d[i]:.2f}/{d[k + i]:.2f}" for i, p in enumerate(SPLIT_PH)))
        self._split_last = (now, dc.ph.copy(), dc.ph_calls)

    def _reap(self) -> None:
        """Under cv: end finished and aborted streams, give slots to streams that have frames."""
        for st in list(self.streams):
            if st.aborted or (st.ended and not st.frames):
                self.streams.remove(st)
                if st.t_first_out is not None:
                    late_hist_add(self._st, st)
                if st.slot is not None:
                    self.dc.close(st.slot)
                    st.slot = None
                self._send([(st, None)])
                continue
            if st.frames and st.slot is None and self.dc.free:
                st.slot = self.dc.open()  # lowest free slot: live streams stay in few groups
        self.backlog = sum(len(st.frames) for st in self.streams)

    def urgent(self, h: float, now: float | None = None) -> float:
        """For the deadline admission gate: the codec frames every open stream needs within the next h seconds: the frames
        due by now + h (the next at `deadline()`, then one every FRAME_S; frames already late all
        count), and at least its steady playback rate (h / FRAME_S frames) for a stream ahead of
        its player, which will need them once its buffer runs down. A stream whose LM has ended
        counts at most its pending frames. Server state only: no request's text."""
        with self.cv:
            now = time.time() if now is None else now
            rate = h / FRAME_S
            n = 0.0
            for st in self.streams:
                if st.aborted:
                    continue
                k = max(rate, math.floor((now + h - st.deadline()) / FRAME_S) + 1)
                n += min(k, len(st.frames)) if st.ended else k
            return n

    def call_full_s(self) -> tuple[float, float, float, int]:
        """For the deadline admission gate: (per-call cycle s of a full 64-slot call, a, b, calls used), from an online
        least-squares fit cycle = a + b x active over the ring's calls at the largest bucket. With
        too little spread in active slots, b falls back to a prior of 0.6% of the mean cycle per
        slot; with no full-bucket call yet, the startup seed (fps_seed)."""
        bs = self.dc.buckets
        lo = bs[-2] if len(bs) > 1 else 0
        full = self.max_groups * GROUP
        pts = [(n, c) for n, g, c in list(self.calls) if g > lo]
        if len(pts) < 8:
            return full / self.fps_seed, float("nan"), float("nan"), len(pts)
        x = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        mx, my = x.mean(), y.mean()
        vx = ((x - mx) ** 2).mean()
        b = float(((x - mx) * (y - my)).mean() / vx) if vx >= 16 else 0.006 * my
        b = min(max(b, 0.0), my / max(mx, 1.0))  # never negative, never above proportional
        a = my - b * mx
        return max(a + b * full, 1e-4), a, b, len(pts)

    def _pick(self):
        """Under cv: the groups of the next call, earliest deadline first, and their pending streams."""
        groups: dict[int, list] = {}  # group -> [earliest deadline, streams]
        for st in self.streams:
            if st.slot is None or not st.frames:
                continue
            e = groups.setdefault(st.slot // GROUP, [math.inf, []])
            e[0] = min(e[0], st.deadline())
            e[1].append(st)
        order = sorted(groups, key=lambda g: (math.floor(groups[g][0] / QUANT_S), -len(groups[g][1]), g))
        take = order[: self.max_groups]
        return [st for g in take for st in groups[g][1]], (min(groups[g][0] for g in take) if take else math.inf), len(order)

    def _warm_thread(self) -> None:
        """Dynamo's config and guards can differ on this thread, so raise the
        recompile limit here too and run every bucket once (no active slot: no state changes)
        before serving, so any recompile happens at startup, not on a live request."""
        dc = self.dc
        if dc.device.type == "cpu":
            return
        _raise_dynamo_limits()
        t = time.time()
        with self.lock, torch.inference_mode():
            for n in dc.buckets:
                dc._call(list(range(n)), {})
            ts = []
            for _ in range(5):  # seed fps_cap from the largest bucket's call time
                t1 = time.perf_counter()
                dc._call(list(range(dc.buckets[-1])), {})
                ts.append(time.perf_counter() - t1)
        call = sorted(ts)[len(ts) // 2]
        self.fps_cap = self.fps_seed = dc.buckets[-1] * GROUP / (call + FPS_GAP_S)
        self.log(f"[codec] codec thread warm-up of buckets {[n * GROUP for n in dc.buckets]}: {time.time() - t:.1f}s; "
                 f"full call {1000 * call:.1f} ms -> fps_cap seed {self.fps_cap:.0f}")

    def _run(self) -> None:
        self._warm_thread()
        tp, cp = time.time(), time.thread_time()  # end of the previous call's post phase
        while True:
            with self.cv:
                waited = idle = 0.0
                while True:
                    self._reap()
                    work, due, ngroups = self._pick()
                    now = time.time()
                    self._stats(now)
                    self._split_log(now)
                    if not work:
                        # Nothing pending, or streams wait for a free slot (re-check soon).
                        starved = any(st.frames and st.slot is None for st in self.streams)
                        t = time.time()
                        self.cv.wait(0.005 if starved else 1.0)
                        idle += time.time() - t
                        continue
                    fresh = any(st.t_first_out is None for st in work)
                    if (not fresh and len(work) < DENSE and ngroups <= self.max_groups and waited < WAIT_S
                            and due - now > self.last_call_s + MARGIN_S):
                        t = time.time()
                        self.cv.wait(WAIT_S - waited)
                        waited += time.time() - t
                        continue
                    break
                batch = {st.slot: st.frames.pop(0) for st in work}
                self.backlog -= len(batch)
                waiters = sum(1 for st in self.streams if st.frames and st.slot is None)
                self._st["waited"] += waited
            try:
                t0, c0 = time.time(), time.thread_time()
                with self.lock:
                    res = self.dc.decode(batch)
                now, c1 = time.time(), time.thread_time()
                self.last_call_s = now - t0
                s = self._st
                if t0 - tp < 1.0:  # not after a stats reset or a long idle stretch
                    s["gap"] += t0 - tp
                    s["pick"] += t0 - tp - waited - idle
                    s["pick_c"] += c0 - cp
                s["idle"] += idle
                s["call_c"] += c1 - c0
                s["waiters"] += waiters
                s["calls"] += 1
                s["active"] += len(batch)
                s["busy"] += self.last_call_s
                self.ticks += 1
                self.frames += len(batch)
                for st in work:
                    if st.t_first_out is None:
                        st.codec_ms, st.first_frames = 1000 * self.last_call_s, 1
                        st.t_first_out = now
                        s["first"].append(1000 * (now - st.t_push))
                    else:
                        lt = now - st.deadline()
                        if lt > 0:
                            s["late"] += 1
                            st.late = True
                            if lt > st.max_late:
                                st.max_late, st.max_late_k = lt, st.delivered
                    st.delivered += 1
                    if not self.pipe:
                        self._put([(st, to_pcm16(res[st.slot]))])
                if self.pipe:
                    # engine.codec.to_pcm16 row-wise, as one array op: bit-identical bytes.
                    w = np.stack([np.asarray(res[st.slot], dtype=np.float32) for st in work])
                    pcm = np.clip(np.round(w * 32767), -32768, 32767).astype("<i2")
                    self._send([(st, pcm[i].tobytes()) for i, st in enumerate(work)])
                tq = tp
                tp, cp = time.time(), time.thread_time()
                if t0 - tq < 1.0 and len(batch) >= FPS_FILL * self.max_groups * GROUP:
                    cyc = (tp - tq) - waited - idle
                    if cyc > 0:
                        x = min(2 * self.fps_cap, max(0.5 * self.fps_cap, len(batch) / cyc))
                        self.fps_cap = max(FPS_FLOOR * self.fps_seed, self.fps_cap + FPS_ALPHA * (x - self.fps_cap))
                if t0 - tq < 1.0 and (tp - tq) - waited - idle > 0:  # the call-timing ring (DeadlineGate)
                    self.calls.append((len(batch), len({sl // GROUP for sl in batch}), (tp - tq) - waited - idle))
                s["post"] += tp - now
                s["post_c"] += cp - c1
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                with self.cv:
                    self._send([(st, e) for st in work])
                    for st in work:
                        st.aborted = True


def scheduler_selftest(svc: CodecService, n: int = 40, frames: int = 6, seed: int = 1, min_db: float = 30.0, log=print) -> float:
    """Push `n` streams of random codes through the service out of order (staggered starts,
    uneven bursts) and compare each stream's PCM16 with `DeviceCodec.decode` of the same frames in
    plain order (all streams each frame). Returns the worst SNR (inf when bit-exact); exits the
    process non-zero below `min_db`. Run before serving, with the service thread started.

    The bar is not bit-exactness: on the device the 8-stream and 32-stream graphs round BF16
    differently, so a stream served by the other bucket than in the plain-order run differs at
    ~38 dB (measured on trn2). A frame out of order, skipped or
    decoded twice gives ~0 dB on random codes."""
    import os as _os

    dc = svc.dc
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(0, 2048, size=(frames, 16)) for _ in range(n)]
    with svc.cv:  # hold the codec thread off the slot table while the reference runs
        slots = [dc.open() for _ in range(n)]
        ref = {s: [] for s in slots}
        with svc.lock:
            for i in range(frames):
                r = dc.decode({s: seqs[k][i] for k, s in enumerate(slots)})
                for s in slots:
                    ref[s].append(r[s])
        for s in slots:
            dc.close(s)
    ref = [to_pcm16(np.concatenate(ref[s])) for s in slots]
    sts = [svc.open() for _ in range(n)]
    left = [0] * n
    order = list(rng.permutation(n))
    for rnd in range(3 * frames):
        for k in order[: n * (rnd + 1) // 3]:
            m = int(rng.integers(1, 3))
            for _ in range(m):
                if left[k] < frames:
                    sts[k].push(seqs[k][left[k]])
                    left[k] += 1
            if left[k] == frames:
                sts[k].finish()
        time.sleep(0.002)
    worst, exact = math.inf, 0
    for k, st in enumerate(sts):
        got = b"".join(st.chunks())
        a, b = np.frombuffer(ref[k], "<i2"), np.frombuffer(got, "<i2")
        if a.shape != b.shape:
            log(f"[selftest] codec scheduler FAILED: stream {k} gave {b.size} samples, want {a.size}")
            _os._exit(2)
        v = snr_db(a, b)
        exact += got == ref[k]
        worst = min(worst, v)
        if not v >= min_db:
            log(f"[selftest] codec scheduler FAILED on stream {k}: SNR {v:.1f} dB < {min_db} vs plain-order decode\n scheduler: {b[:16]} ...\n plain: {a[:16]} ...")
            _os._exit(2)
    log(f"[selftest] codec scheduler: {n} streams out of order vs plain-order decode: {exact}/{n} bit-exact, worst SNR {worst:.1f} dB")
    return worst


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = np.asarray(ref, np.float64), np.asarray(test, np.float64)
    n = np.sum((ref - test) ** 2)
    return math.inf if n == 0 else float(10 * np.log10(np.sum(ref**2) / max(n, 1e-20)))


__all__ = ["scheduler_selftest", "CodecService", "CodecStream", "CodecStep", "CodecTables", "DeviceCodec", "GROUP", "SAMPLES_PER_FRAME", "snr_db"]
