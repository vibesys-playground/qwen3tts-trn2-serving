"""The Qwen3-TTS talker (codebook 0) and code predictor (codebooks 1-15) as fixed-shape decode steps.

Each step module processes one position for each of B sequences against a static key/value cache
[layers, B, kv_heads, S, head_dim] and returns the updated cache, so a compiled graph has one shape
for its whole life (torch.compile(backend="neuron"), or plain eager on CPU for tests). The prompt
goes through `PrefillStep`, one call for all its positions on the same Parameters, and both
/generate and /score start with it.

Numerics follow the reference (qwen-tts 0.1.1) in BF16: residual stream and projections in BF16,
RMSNorm in FP32, softmax in FP32, logits returned in FP32. The talker's interleaved M-RoPE reduces
to plain RoPE for text-only positions (all three position ids are equal).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

BF = torch.bfloat16


def load_checkpoint(root: str | Path) -> tuple[dict, dict]:
    """(config, {name: tensor}) for the talker part of the checkpoint (BF16 on the host)."""
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    w = {}
    with safe_open(str(root / "model.safetensors"), "pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k)
    return cfg, w


def rope_table(theta: float, head_dim: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [n, head_dim] in FP32 (the reference computes them in FP32, then casts)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    f = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((f, f), -1)
    return emb.cos(), emb.sin()


def _rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class DecodeStep(nn.Module):
    """One position of a Qwen3 decoder stack for B sequences with a static KV cache.

    forward(x [B, in_dim] BF16, pos [B] int64, K, V [L, B, kvh, S, hd] BF16)
        -> (hidden after the final norm [B, H] BF16, logits [B, vocab] FP32, K', V')
    Cache slots >= pos are ignored (masked), the slot at pos is written with this position's key
    and value. `heads` is [G, vocab, H]: the head used is heads[head_index(pos)].
    """

    def __init__(self, w: dict, prefix: str, cfg: dict, heads: torch.Tensor, max_pos: int, in_proj=None, head_offset: int = 0):
        super().__init__()
        L = cfg["num_hidden_layers"]
        self.L, self.H = L, cfg["hidden_size"]
        self.nh, self.kvh, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.head_offset = head_offset

        def stack(name):
            return nn.Parameter(torch.stack([w[f"{prefix}layers.{i}.{name}"] for i in range(L)]).to(BF).contiguous(), requires_grad=False)

        self.wq, self.wk, self.wv, self.wo = (stack(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = stack("self_attn.q_norm.weight"), stack("self_attn.k_norm.weight")
        self.ln1, self.ln2 = stack("input_layernorm.weight"), stack("post_attention_layernorm.weight")
        self.wg, self.wu, self.wd = (stack(f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.norm = nn.Parameter(w[f"{prefix}norm.weight"].to(BF), requires_grad=False)
        self.heads = nn.Parameter(heads.to(BF).contiguous(), requires_grad=False)
        if in_proj is not None:
            self.pw = nn.Parameter(in_proj[0].to(BF), requires_grad=False)
            self.pb = nn.Parameter(in_proj[1].to(BF), requires_grad=False)
        else:
            self.pw = self.pb = None
        cos, sin = rope_table(cfg["rope_theta"], self.hd, max_pos)
        self.register_buffer("cos", cos.to(BF), persistent=False)
        self.register_buffer("sin", sin.to(BF), persistent=False)

    def forward(self, x, pos, K, V, inplace: bool = False, head_index=None, pages=None):
        """inplace=True writes the slot at pos[b] of each row into K, V themselves (scatter_; the
        Neuron backend writes the input buffer in place) and returns them; else a
        new cache is returned (torch.where over the whole cache).

        pages (inplace only; see `page_inputs`): a long stream keeps its batch row (the
        leader, positions 0..page_at-1) and borrows other rows of the batch as pages of S - 1
        slots each (slot S - 1 of every row of a group is a masked junk slot). A page row attends
        with its leader's query (qsrc), writes the leader's key/value (wsrc) at wslot, and masks
        its own new key (snb); the leader's output is the exact FP32 log-sum-exp merge of its own
        and its pages' unnormalized attention (part/pw). With no group every row is its own
        partner with weight 0 and the merge is the plain softmax. Tuple: ("gather", nval, wslot,
        snb, wsrc [B], qsrc [B], part [B, P], pw [B, P] FP32) or ("mm", nval, wslot, snb,
        Wm, Qm, Cg [B, B] FP32 0/1, unused): row gathers as index_select or as 0/1 matmuls."""
        B = x.shape[0]
        S = K.shape[3]
        nh, kvh, hd = self.nh, self.kvh, self.hd
        rep = nh // kvh
        if self.pw is not None:
            x = F.linear(x, self.pw, self.pb)
        cos = self.cos.index_select(0, pos)[:, None, :]  # [B, 1, hd]
        sin = self.sin.index_select(0, pos)[:, None, :]
        slots = torch.arange(S, device=x.device)[None, :]
        mode = None if pages is None else pages[0]
        if mode is not None:
            assert inplace
            nval, wslot, snb = pages[1:4]
            snb4 = snb.view(B, 1, 1, 1)
            if mode == "gather":
                wsrc, qsrc, part, pw = pages[4:8]
                P = part.shape[1]
                partf = part.reshape(-1)
                pw5 = pw.view(B, P, 1, 1, 1)
            else:
                Wm, Qm, Cg = pages[4:7]
                cgm = (Cg > 0.5)[:, :, None, None]  # [B, B, 1, 1]: row b merges row c
        else:
            nval = pos
        neg = torch.finfo(torch.float32).min
        bias = torch.where(slots < nval[:, None], 0.0, neg)[:, None, None, :]  # [B,1,1,S]
        scale = hd**-0.5
        h = x
        ks, vs = [], []
        for i in range(self.L):
            r = h
            a = _rms(h, self.ln1[i], self.eps)
            q = F.linear(a, self.wq[i]).view(B, nh, hd)
            k = F.linear(a, self.wk[i]).view(B, kvh, hd)
            v = F.linear(a, self.wv[i]).view(B, kvh, hd)
            q = _rms(q, self.qn[i], self.eps)
            k = _rms(k, self.kn[i], self.eps)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            qg = q.view(B, kvh, rep, hd)
            # Attention in FP32 (as the reference's SDPA path), over the cache plus this position.
            qf = qg.float()
            if mode is None:
                s_old = torch.matmul(qf, K[i].float().transpose(2, 3)) * scale + bias  # [B, kvh, rep, S]
                s_new = (qf * k.float()[:, :, None, :]).sum(-1, keepdim=True) * scale  # [B, kvh, rep, 1]
                p = torch.softmax(torch.cat([s_old, s_new], -1), -1)
                o = torch.matmul(p[..., :S], V[i].float()) + p[..., S:] * v.float()[:, :, None, :]
            else:
                o = self._paged_attn(mode, pages, qf, k, v, K[i], V[i], bias, snb4, scale,
                                     (partf, pw5, P) if mode == "gather" else cgm)
            o = F.linear(o.reshape(B, nh * hd).to(h.dtype), self.wo[i])
            h = r + o
            r = h
            a = _rms(h, self.ln2[i], self.eps)
            h = r + F.linear(F.silu(F.linear(a, self.wg[i])) * F.linear(a, self.wu[i]), self.wd[i])
            ks.append(k)
            vs.append(v)
        hn = _rms(h, self.norm, self.eps)
        if head_index is None:
            hi = (pos[:1] - self.head_offset).clamp(min=0, max=self.heads.shape[0] - 1)
        else:  # [1] int64 from the host: no clamp on an input-derived index in the batched graphs
            hi = head_index
        head = self.heads.index_select(0, hi)[0]
        logits = F.linear(hn, head).float()
        if inplace:
            kn, vn = torch.stack(ks), torch.stack(vs)  # [L, B, kvh, hd]
            wp = pos
            if mode == "gather":
                kn, vn, wp = kn.index_select(1, wsrc), vn.index_select(1, wsrc), wslot
            elif mode == "mm":
                kn = torch.matmul(Wm.to(kn.dtype), kn.transpose(0, 1).reshape(B, -1)).view(B, self.L, kvh, hd).transpose(0, 1)
                vn = torch.matmul(Wm.to(vn.dtype), vn.transpose(0, 1).reshape(B, -1)).view(B, self.L, kvh, hd).transpose(0, 1)
                wp = wslot
            idx = wp.view(1, B, 1, 1, 1).expand(self.L, B, kvh, 1, hd)
            K.scatter_(3, idx, kn.unsqueeze(3))
            V.scatter_(3, idx, vn.unsqueeze(3))
            return hn, logits, K, V
        write = (slots == pos[:, None])[None, :, None, :, None]  # [1, B, 1, S, 1]
        Kn = torch.where(write, torch.stack(ks)[:, :, :, None, :], K)
        Vn = torch.where(write, torch.stack(vs)[:, :, :, None, :], V)
        return hn, logits, Kn, Vn

    @staticmethod
    def _paged_attn(mode, pages, qf, k, v, Ki, Vi, bias, snb4, scale, mx):
        """One layer's attention with page rows (DecodeStep's `pages`): FP32 scores over the
        row's cache plus its new key, the softmax as its explicit pieces (max m, unnormalized
        output ou, sum l), then each leader's merge with its pages at the joint max."""
        B, kvh, rep, hd = qf.shape
        if mode == "gather":
            qf = qf.index_select(0, pages[5])  # a page row attends with its leader's query
        else:
            qf = torch.matmul(pages[5], qf.reshape(B, -1)).view(B, kvh, rep, hd)
        s_old = torch.matmul(qf, Ki.float().transpose(2, 3)) * scale + bias  # [B, kvh, rep, S]
        s_new = (qf * k.float()[:, :, None, :]).sum(-1, keepdim=True) * scale + snb4  # [B, kvh, rep, 1]
        m = torch.maximum(s_old.amax(-1, keepdim=True), s_new)
        e = torch.exp(s_old - m)
        en = torch.exp(s_new - m)
        lsum = e.sum(-1, keepdim=True) + en  # [B, kvh, rep, 1]
        ou = torch.matmul(e, Vi.float()) + en * v.float()[:, :, None, :]  # [B, kvh, rep, hd]
        if mode == "gather":
            partf, pw5, P = mx
            mp = m.index_select(0, partf).view(B, P, kvh, rep, 1)
            M = torch.maximum(m, mp.amax(1))
            a = torch.exp(m - M)
            b = torch.exp(mp - M[:, None]) * pw5  # [B, P, kvh, rep, 1]
            num = a * ou + (b * ou.index_select(0, partf).view(B, P, kvh, rep, hd)).sum(1)
            den = a * lsum + (b * lsum.index_select(0, partf).view(B, P, kvh, rep, 1)).sum(1)
            return num / den
        cgm = mx
        neg = torch.finfo(torch.float32).min
        mm = m[..., 0]  # [B, kvh, rep]
        M = torch.where(cgm, mm[None], neg).amax(1)  # [B, kvh, rep]: max over row b's group
        w = torch.exp(torch.where(cgm, mm[None] - M[:, None], neg)).permute(2, 3, 0, 1)  # [kvh, rep, B, B]
        num = torch.matmul(w, ou.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        den = torch.matmul(w, lsum.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        return num / den


class PrefillStep(nn.Module):
    """The first T positions (0..T-1) of a `DecodeStep` stack in one call, on an empty cache.

    forward(x [B, T, in_dim] BF16, K0, V0 [L, B, kvh, S, hd] BF16 zero caches)
        -> (hidden after the final norm of the last position [B, H] BF16,
            logits of the last position [B, vocab] FP32, K, V [L, B, kvh, S, hd])
    with slots 0..T-1 of K, V written and the rest copied from the empty input (zero). It holds
    the decode step itself as a submodule, so both graphs read the same Parameters (one copy of
    the weights on the device). Same math as T decode steps: BF16 projections, FP32 RMSNorm, FP32
    causal softmax over the BF16 keys/values (as the decode step reads them from its cache).
    """

    def __init__(self, step: DecodeStep, T: int):
        super().__init__()
        self.step = step
        self.T = T
        assert step.head_offset == 0 and step.heads.shape[0] == 1, "prefill is for the talker (one head)"
        bias = torch.full((T, T), torch.finfo(torch.float32).min).triu(1)
        self.register_buffer("bias", bias, persistent=False)  # [T, T] causal, FP32

    def forward(self, x, K0, V0):
        hn, logits, ks, vs = self.run(x)
        T = self.T
        K = torch.cat([ks, K0[:, :, :, T:]], 3)
        V = torch.cat([vs, V0[:, :, :, T:]], 3)
        return hn, logits, K, V

    def run(self, x):
        """(hidden [B, H], logits [B, vocab] of the last position, K, V [L, B, kvh, T, hd])."""
        st = self.step
        B, T = x.shape[0], x.shape[1]
        assert T == self.T
        nh, kvh, hd = st.nh, st.kvh, st.hd
        rep = nh // kvh
        if st.pw is not None:
            x = F.linear(x, st.pw, st.pb)
        cos = st.cos[:T][None, :, None, :]  # [1, T, 1, hd]
        sin = st.sin[:T][None, :, None, :]
        scale = hd**-0.5
        h = x
        ks, vs = [], []
        for i in range(st.L):
            r = h
            a = _rms(h, st.ln1[i], st.eps)
            q = F.linear(a, st.wq[i]).view(B, T, nh, hd)
            k = F.linear(a, st.wk[i]).view(B, T, kvh, hd)
            v = F.linear(a, st.wv[i]).view(B, T, kvh, hd)
            q = _rms(q, st.qn[i], st.eps)
            k = _rms(k, st.kn[i], st.eps)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            k = k.transpose(1, 2)  # [B, kvh, T, hd] (cache layout)
            v = v.transpose(1, 2)
            qf = q.view(B, T, kvh, rep, hd).permute(0, 2, 3, 1, 4).float()  # [B, kvh, rep, T, hd]
            s = torch.matmul(qf, k.float()[:, :, None].transpose(3, 4)) * scale + self.bias  # [B, kvh, rep, T, T]
            p = torch.softmax(s, -1)
            o = torch.matmul(p, v.float()[:, :, None])  # [B, kvh, rep, T, hd]
            o = o.permute(0, 3, 1, 2, 4).reshape(B, T, nh * hd).to(h.dtype)
            h = r + F.linear(o, st.wo[i])
            r = h
            a = _rms(h, st.ln2[i], st.eps)
            h = r + F.linear(F.silu(F.linear(a, st.wg[i])) * F.linear(a, st.wu[i]), st.wd[i])
            ks.append(k)
            vs.append(v)
        hn = _rms(h[:, -1], st.norm, st.eps)
        logits = F.linear(hn, st.heads[0]).float()
        return hn, logits, torch.stack(ks), torch.stack(vs)


class FrameStep(nn.Module):
    """Everything one frame needs, as one graph: sample code 0 from the talker's logits, run the
    16 code-predictor positions sampling codes 1-15, embed the frame, run the next talker step.

    forward(h [B, H] BF16, logits0 [B, 3072] FP32, pos [B] int64, K, V (talker cache),
            presence [B, 3072] FP32 0/1, nframes [B] int64, hostin [B, width] FP32)
        -> (codes [B, 16] int64, h', logits0', pos + 1, K', V', presence', nframes + 1,
            sub_logits [B, 15, 2048] FP32)

    `hostin` is the one host->device copy per frame: Gumbel noise for codebook 0 [3072] and for
    codebooks 1-15 [15 x 2048], the trailing-text slot of this frame [H] (FP32), the forced codes
    [16] and a teacher-forcing flag [1]. With the flag set (/score) every code is the forced one
    and the noise is ignored, so /score runs this same graph. The input `logits0` and the output
    `sub_logits` are the raw logits of the frame's 16 codes, for /score's log-softmax.
    """

    def __init__(self, talker: DecodeStep, pred: DecodeStep, codec_emb: torch.Tensor, cp_emb: torch.Tensor, eos_id: int):
        super().__init__()
        self.talker, self.pred = talker, pred
        self.V0 = codec_emb.shape[0]
        self.n_sub, self.V1 = cp_emb.shape[0], cp_emb.shape[1]
        self.H = codec_emb.shape[1]
        self.eos_id = eos_id
        # Embedding tables in their checkpoint dtype (BF16); rows are upcast to FP32 before the
        # sum (exact), summed in FP32 in code order and rounded to BF16 once, as the reference.
        self.codec_emb = nn.Parameter(codec_emb.to(BF).contiguous(), requires_grad=False)
        self.cp_emb = nn.Parameter(cp_emb.to(BF).reshape(-1, self.H).contiguous(), requires_grad=False)
        sup = torch.zeros(self.V0, dtype=torch.bool)
        sup[self.V0 - 1024 :] = True
        sup[eos_id] = False
        self.register_buffer("suppress", sup, persistent=False)
        self.o_g1 = self.V0
        self.o_trail = self.V0 + self.n_sub * self.V1
        self.o_forced = self.o_trail + self.H
        self.width = self.o_forced + self.n_sub + 2  # forced codes 0..15, then the flag

    def forward(self, h, logits0, pos, K, V, presence, nframes, hostin):
        from .sampling import code0_scores, gumbel_pick

        B = h.shape[0]
        dev = h.device
        n_groups = self.n_sub + 1
        forced = hostin[:, self.o_forced : self.o_forced + n_groups].to(torch.int64)
        use = hostin[:, self.o_forced + n_groups] > 0.5  # [B]
        trail = hostin[:, self.o_trail : self.o_trail + self.H]

        c0 = gumbel_pick(code0_scores(logits0, presence, nframes, self.suppress, self.eos_id), hostin[:, : self.V0])
        c0 = torch.where(use, forced[:, 0], c0)
        p = self.pred
        Kp = torch.zeros((p.L, B, p.kvh, n_groups, p.hd), dtype=BF, device=dev)
        Vp = torch.zeros((p.L, B, p.kvh, n_groups, p.hd), dtype=BF, device=dev)
        _, _, Kp, Vp = p(h, torch.zeros(B, dtype=torch.long, device=dev), Kp, Vp)
        x = self.codec_emb.index_select(0, c0)
        emb = x.float()
        codes, subs = [c0], []
        for g in range(1, n_groups):
            _, lg, Kp, Vp = p(x, torch.full((B,), g, dtype=torch.long, device=dev), Kp, Vp)
            subs.append(lg)
            o = self.o_g1 + (g - 1) * self.V1
            c = gumbel_pick(lg, hostin[:, o : o + self.V1])
            c = torch.where(use, forced[:, g], c)
            codes.append(c)
            x = self.cp_emb.index_select(0, c + (g - 1) * self.V1)
            emb = emb + x.float()
        xt = (emb + trail).to(BF)
        # A position past the cache (only after the last allowed frame) is clamped; its outputs
        # are never used.
        S = K.shape[3]
        hn, l2, K2, V2 = self.talker(xt, pos.clamp(max=S - 1), K, V)
        # (This B = 1 graph rewrites the whole KV cache with torch.where each step; the batched
        # graphs below write only the slot at pos, in place.)
        idx = torch.arange(self.V0, device=dev)[None, :]
        presence2 = torch.where(idx == c0[:, None], 1.0, presence)
        return torch.stack(codes, 1), hn, l2, pos + 1, K2, V2, presence2, nframes + 1, torch.stack(subs, 1)


def build_frame(cfg: dict, w: dict, talker: DecodeStep, pred: DecodeStep) -> FrameStep:
    """The per-frame graph module over the given step modules (their parameters are shared)."""
    tc = cfg["talker_config"]
    n = tc["code_predictor_config"]["num_code_groups"] - 1
    cp = torch.stack([w[f"talker.code_predictor.model.codec_embedding.{i}.weight"] for i in range(n)])
    return FrameStep(talker, pred, w["talker.model.codec_embedding.weight"], cp, tc["codec_eos_token_id"])


def build_steps(cfg: dict, w: dict, talker_max_pos: int) -> tuple[DecodeStep, DecodeStep]:
    """(talker step, code predictor step) on the host; move them with .to(device)."""
    tc = dict(cfg["talker_config"])
    cc = dict(tc["code_predictor_config"])
    talker = DecodeStep(w, "talker.model.", tc, w["talker.codec_head.weight"][None], talker_max_pos)
    n = cc["num_code_groups"] - 1
    heads = torch.stack([w[f"talker.code_predictor.lm_head.{i}.weight"] for i in range(n)])
    proj = (w["talker.code_predictor.small_to_mtp_projection.weight"], w["talker.code_predictor.small_to_mtp_projection.bias"])
    # Code predictor positions: 0 = talker hidden, 1 = code 0, g = code g-1; the head at position
    # p predicts code p, i.e. lm_head[p - 1] (position 0's logits are unused).
    pred = DecodeStep(w, "talker.code_predictor.model.", cc, heads, cc["num_code_groups"], in_proj=proj, head_offset=1)
    return talker, pred


# ------------------------------------------------------------------------- batched serving graphs
class PredFirst(nn.Module):
    """Code-predictor positions 0 (talker hidden) and 1 (code 0) for B streams.

    forward(h [B, H] BF16, c0 [B] int64) -> (logits of code 1 [B, 2048] FP32, Kp, Vp)

    The batched step samples every code on the host (exact Gumbel-max, engine.sampling): the
    in-graph samplers miscompiled in a fused batched frame graph on the device (the
    full-vocabulary form; the top-k candidate form near ties at the k-th value; a counting form),
    so on the non-speculative path (QWEN3TTS_SPEC=0) one frame is these small graphs with a host
    draw between them. They share every Parameter with the B = 1 FrameStep `fs`.
    """

    def __init__(self, fs: FrameStep):
        super().__init__()
        self.fs = fs

    def forward(self, h, c0):
        fs = self.fs
        p = fs.pred
        B = h.shape[0]
        dev = h.device
        G = fs.n_sub + 1
        Kp = torch.zeros((p.L, B, p.kvh, G, p.hd), dtype=BF, device=dev)
        Vp = torch.zeros((p.L, B, p.kvh, G, p.hd), dtype=BF, device=dev)
        _, _, Kp, Vp = p(h, torch.zeros(B, dtype=torch.long, device=dev), Kp, Vp)
        _, lg, Kp, Vp = p(fs.codec_emb.index_select(0, c0), torch.ones(B, dtype=torch.long, device=dev), Kp, Vp)
        return lg, Kp, Vp


class PredNext(nn.Module):
    """Code-predictor position g (2..15) for B streams, given code g-1 (codebook g-1 >= 1).

    forward(c [B] int64, g [B] int64 (all equal, from the host), head [1] int64 (= g - 1), Kp, Vp)
        -> (logits of code g [B, 2048] FP32, Kp', Vp')
    """

    def __init__(self, fs: FrameStep):
        super().__init__()
        self.fs = fs

    def forward(self, c, g, head, Kp, Vp):
        fs = self.fs
        x = fs.cp_emb.index_select(0, c + (g - 2) * fs.V1)
        _, lg, Kp, Vp = fs.pred(x, g, Kp, Vp, head_index=head)
        return lg, Kp, Vp


class TalkerIn(nn.Module):
    """The frame's embedding (16 codes, summed in FP32 in code order, plus the text slot, rounded
    to BF16 once, as FrameStep) and the talker step at each row's position, written in place into
    a 512-slot cache (one scatter_ per row).

    forward(codes [B, 16] int64, trail [B, H] FP32, pos [B] int64 (from the host, < S), K, V)
        -> (h' [B, H] BF16, logits0' [B, 3072] FP32)
    """

    def __init__(self, fs: FrameStep):
        super().__init__()
        self.fs = fs

    def forward(self, codes, trail, pos, K, V):
        fs = self.fs
        emb = fs.codec_emb.index_select(0, codes[:, 0]).float()
        for g in range(1, fs.n_sub + 1):
            emb = emb + fs.cp_emb.index_select(0, codes[:, g] + (g - 1) * fs.V1).float()
        hn, l2, _, _ = fs.talker((emb + trail).to(BF), pos, K, V, inplace=True)
        return hn, l2


class TalkerInP(nn.Module):
    """TalkerIn with paged rows: long streams keep their batch row and borrow free rows
    as KV pages (DecodeStep's `pages`; host inputs from `page_inputs`). Same embedding and talker
    math; with no group it computes TalkerIn's function.

    forward(codes [B, 16], trail [B, H] FP32, pos [B] (true positions, < 2048), nval, wslot,
    snb, x1, x2, x3, x4 (mode "gather": wsrc, qsrc, part, pw; "mm": Wm, Qm, Cg, unused), K, V)
        -> (h' [B, H] BF16, logits0' [B, 3072] FP32)
    """

    def __init__(self, fs: FrameStep, mode: str = "gather"):
        super().__init__()
        self.fs = fs
        self.mode = mode

    def forward(self, codes, trail, pos, nval, wslot, snb, x1, x2, x3, x4, K, V):
        fs = self.fs
        emb = fs.codec_emb.index_select(0, codes[:, 0]).float()
        for g in range(1, fs.n_sub + 1):
            emb = emb + fs.cp_emb.index_select(0, codes[:, g] + (g - 1) * fs.V1).float()
        hn, l2, _, _ = fs.talker((emb + trail).to(BF), pos, K, V, inplace=True, pages=(self.mode, nval, wslot, snb, x1, x2, x3, x4))
        return hn, l2


def page_inputs(pos, groups, P: int = 1, mode: str = "gather", page_at: int = 511, S: int = 512, ps: int | None = None):
    """Host inputs of TalkerInP (after codes, trail): (pos, nval, wslot, snb, x1, x2, x3, x4).

    pos: true position of every row (padding rows: any value < S). groups: {leader row: [page
    rows]} (at most P pages each). The leader holds positions 0..page_at-1 in slots 0..page_at-1;
    page j (0-based) holds positions page_at + j ps + t in its slot t < ps (ps = S - 1 when
    serving; the self-test uses small page_at and ps). Every row of a group whose range does not
    hold this position writes into its junk slot S - 1 (never valid: page_at, ps <= S - 1)."""
    ps = S - 1 if ps is None else ps
    assert 0 < page_at <= S - 1 and 0 < ps <= S - 1
    B = len(pos)
    pos = torch.as_tensor(pos, dtype=torch.long).clone()
    nval, wslot = pos.clone(), pos.clone()
    ar = torch.arange(B)
    wsrc, qsrc = ar.clone(), ar.clone()
    part = ar[:, None].repeat(1, P).contiguous()
    pw = torch.zeros((B, P), dtype=torch.float32)
    snb = torch.zeros(B, dtype=torch.float32)
    for ld, pages in groups.items():
        assert len(pages) <= P, (ld, pages, P)
        p = int(pos[ld])
        if p >= page_at:
            nval[ld], wslot[ld] = page_at, S - 1
        for j, r in enumerate(pages):
            off = p - page_at - j * ps
            nval[r] = min(max(off, 0), ps)
            wslot[r] = off if 0 <= off < ps else S - 1
            wsrc[r] = qsrc[r] = ld
            snb[r] = torch.finfo(torch.float32).min
            pos[r] = p
            part[ld, j] = r
            pw[ld, j] = 1.0
    if mode == "gather":
        return pos, nval, wslot, snb, wsrc, qsrc, part, pw
    Wm = torch.zeros((B, B), dtype=torch.float32)
    Wm[ar, wsrc] = 1.0
    Qm = torch.zeros((B, B), dtype=torch.float32)
    Qm[ar, qsrc] = 1.0
    Cg = torch.eye(B, dtype=torch.float32)
    for ld, pages in groups.items():
        for r in pages:
            Cg[ld, r] = 1.0
    return pos, nval, wslot, snb, Wm, Qm, Cg, torch.zeros(1, dtype=torch.float32)


class PrefillInto(nn.Module):
    """The prompt prefill (PrefillStep.run) of k new streams, written into rows `rows` of a batch:
    slots 0..T-1 of K, V in place, and the rows' h, logits0, presence (= 0) and nframes (= 0) in
    the returned small state (positions are the host's). Rows may repeat (padding duplicates a real entry: same
    input, same values written).

    forward(x [k, T, H] BF16, rows [k] int64, K, V, h, logits0, presence, nframes)
        -> (h', logits0', presence', nframes')
    """

    def __init__(self, pre: PrefillStep):
        super().__init__()
        self.pre = pre

    def forward(self, x, rows, K, V, h, logits0, presence, nframes):
        hn, lg, ks, vs = self.pre.run(x)  # ks [L, k, kvh, T, hd]
        L, k, kvh, T, _ = ks.shape
        dev = x.device
        li = torch.arange(L, device=dev).view(L, 1, 1, 1).expand(L, k, kvh, T)
        bi = rows.view(1, k, 1, 1).expand(L, k, kvh, T)
        hi = torch.arange(kvh, device=dev).view(1, 1, kvh, 1).expand(L, k, kvh, T)
        ti = torch.arange(T, device=dev).view(1, 1, 1, T).expand(L, k, kvh, T)
        K.index_put_((li, bi, hi, ti), ks)
        V.index_put_((li, bi, hi, ti), vs)
        h2 = h.index_put((rows,), hn)
        l2 = logits0.index_put((rows,), lg)
        pr2 = presence.index_put((rows,), torch.zeros((k, presence.shape[1]), dtype=presence.dtype, device=dev))
        nf2 = nframes.index_put((rows,), torch.zeros((k,), dtype=nframes.dtype, device=dev))
        return h2, l2, pr2, nf2


class MoveRows(nn.Module):
    """Copy rows idx of a source talker cache into rows 0..m-1 of a destination cache of another
    batch size (m = len(idx)), in place: how the batch changes bucket.

    forward(dK, dV, sK, sV, idx [m] int64) -> idx + 0 (a small output to sync on)
    """

    def forward(self, dK, dV, sK, sV, idx):
        m = idx.shape[0]
        ar = torch.arange(m, device=idx.device)
        dK.index_copy_(1, ar, sK.index_select(1, idx))
        dV.index_copy_(1, ar, sV.index_select(1, idx))
        return idx + 0


# ------------------------------------------------------------------ speculative chain
def pred_position(p: DecodeStep, x, g: int, ks: list, vs: list):
    """Code-predictor position g (a Python int) for B rows: x [B, in] BF16 -> hidden after the
    final norm [B, H]. Appends this position's key/value [B, kvh, hd] to ks[i], vs[i] per layer and
    attends over exactly those g + 1 entries (FP32 scores, softmax and weighted sum): the same
    math as DecodeStep over a 16-slot cache whose slots > g are masked (they contribute 0), with
    the RoPE row and (in PredChain) the head as static slices (one specialized code path per
    position, so the compiled graph has no dynamic indexing)."""
    B = x.shape[0]
    nh, kvh, hd = p.nh, p.kvh, p.hd
    rep = nh // kvh
    if p.pw is not None:
        x = F.linear(x, p.pw, p.pb)
    cos, sin = p.cos[g], p.sin[g]  # [hd], static rows
    scale = hd**-0.5
    h = x
    for i in range(p.L):
        r = h
        a = _rms(h, p.ln1[i], p.eps)
        q = F.linear(a, p.wq[i]).view(B, nh, hd)
        k = F.linear(a, p.wk[i]).view(B, kvh, hd)
        v = F.linear(a, p.wv[i]).view(B, kvh, hd)
        q = _rms(q, p.qn[i], p.eps)
        k = _rms(k, p.kn[i], p.eps)
        q = q * cos + _rot(q) * sin
        k = k * cos + _rot(k) * sin
        ks[i].append(k)
        vs[i].append(v)
        qf = q.view(B, kvh, rep, hd).float()
        if g == 0:  # one key: softmax is 1, the output is v
            o = v.float()[:, :, None, :].expand(B, kvh, rep, hd)
        else:
            Kc = torch.stack(ks[i], 2).float()  # [B, kvh, g+1, hd]
            Vc = torch.stack(vs[i], 2).float()
            s = torch.matmul(qf, Kc.transpose(2, 3)) * scale  # [B, kvh, rep, g+1]
            o = torch.matmul(torch.softmax(s, -1), Vc)  # [B, kvh, rep, hd]
        o = F.linear(o.reshape(B, nh * hd).to(h.dtype), p.wo[i])
        h = r + o
        r = h
        a = _rms(h, p.ln2[i], p.eps)
        h = r + F.linear(F.silu(F.linear(a, p.wg[i])) * F.linear(a, p.wu[i]), p.wd[i])
    return _rms(h, p.norm, p.eps)


class PredChain(nn.Module):
    """All 16 code-predictor positions of a frame for B rows in one graph, each position
    specialized to its index (static RoPE rows and heads), proposing codes 1-15 with the topk-free sampler
    (engine.sampling.gumbel_pick_free) on host Gumbel noise. The proposals are NOT the served
    codes: engine.batch verifies every one on the host (exact Gumbel-max on these logits and the
    same noise) and reruns this graph with the verified prefix forced from the first wrong one.

    forward(h [B, H] BF16, c0 [B] int64, gum [B, 15 * V1] FP32, forced [B, 15] int64)
        -> (codes [B, 16] int64 (c0, then the codes used at positions 1-15),
            sub_logits [B, 15, V1] FP32: logits of code g given the codes before it)
    forced[b, g-1] >= 0 makes row b take that code at codebook g (teacher forcing for /score, and
    the verified prefix of a fix-up pass); -1 lets the sampler propose.
    """

    def __init__(self, fs: FrameStep):
        super().__init__()
        self.fs = fs

    def forward(self, h, c0, gum, forced):
        from .sampling import gumbel_pick_free

        fs = self.fs
        p = fs.pred
        V1 = fs.V1
        ks = [[] for _ in range(p.L)]
        vs = [[] for _ in range(p.L)]
        pred_position(p, h, 0, ks, vs)  # position 0: keys/values only (its head output is unused)
        x = fs.codec_emb.index_select(0, c0)
        codes, subs = [c0], []
        for g in range(1, fs.n_sub + 1):
            hn = pred_position(p, x, g, ks, vs)
            lg = F.linear(hn, p.heads[g - 1]).float()
            subs.append(lg)
            o = (g - 1) * V1
            c = gumbel_pick_free(lg, gum[:, o : o + V1])
            f = forced[:, g - 1]
            c = torch.where(f >= 0, f, c)
            codes.append(c)
            if g < fs.n_sub:
                x = fs.cp_emb.index_select(0, c + (g - 1) * V1)
        return torch.stack(codes, 1), torch.stack(subs, 1)
