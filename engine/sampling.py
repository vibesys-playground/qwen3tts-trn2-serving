"""Sampling with the checkpoint's defaults, in the order of transformers' generate().

Codebook 0 (talker): repetition penalty 1.05 over the codes generated so far, the 1024 control
tokens suppressed except EOS, EOS suppressed before 2 frames, then temperature 0.9, top-k 50
(top-p 1.0 is the identity), softmax and a categorical draw. Codebooks 1-15 (code predictor):
temperature 0.9 and top-k 50 only. All in FP32.

Every served code is an exact draw made by the pure tensor functions below: `code0_scores`
(penalty, suppression, min-frames EOS mask) and `gumbel_pick` (temperature, top-k threshold over
the full vocabulary, Gumbel-max). Gumbel-max, argmax(s + G) with G = -log(-log(U)) i.i.d., is an
exact draw from softmax(s); the top-k mask keeps every entry >= the k-th largest value, ties
included, exactly as the reference's masked_fill. The noise is drawn on the host with the
server's generator (`gumbel_noise`).

Where the draw happens: the batched serving path (engine.batch) draws code 0 on the host with
`gumbel_pick`; codes 1-15 are proposed on the device by the speculative chain with
`gumbel_pick_free` and each proposal is accepted only if `gumbel_check` confirms it equals
`gumbel_pick` on the same logits and noise (otherwise the host's draw replaces it). The B = 1
path (engine.model.FrameStep, used for long streams) runs `gumbel_pick` inside its graph.

`Code0Sampler` and `_draw` are the host reference (softmax + multinomial) the tests compare with.
"""

from __future__ import annotations

import os

import torch

TEMPERATURE = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05
MIN_NEW_TOKENS = 2
NEG = -1.0e30  # finite "minus infinity" for masked scores (no inf/NaN arithmetic on the device)


# ---------------------------------------------------------------------------- graph sampler
def code0_scores(logits: torch.Tensor, presence: torch.Tensor, nframes: torch.Tensor, suppress: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Codebook-0 scores before temperature: logits [B, V] FP32, presence [B, V] 0/1 or bool (codes
    generated so far), nframes [B] (frames generated so far), suppress [V] bool."""
    s = logits.float()
    s = torch.where(presence > 0, torch.where(s < 0, s * REPETITION_PENALTY, s / REPETITION_PENALTY), s)
    s = torch.where(suppress[None, :], NEG, s)
    idx = torch.arange(s.shape[-1], device=s.device)[None, :]
    early = (nframes[:, None] < MIN_NEW_TOKENS) & (idx == eos_id)
    return torch.where(early, NEG, s)


def gumbel_pick(scores: torch.Tensor, gumbel: torch.Tensor) -> torch.Tensor:
    """One categorical draw per row from softmax(top-k-masked scores / T): [B, V] -> [B] int64."""
    s = scores.float() / TEMPERATURE
    kth = torch.topk(s, TOP_K, dim=-1).values[..., -1:]
    s = torch.where(s >= kth, s, NEG)
    return torch.argmax(s + gumbel, dim=-1)


TOPK_CAND = 64  # candidates the batched sampler takes: the top-k set plus room for ties at the k-th value


def gumbel_pick_topk(scores: torch.Tensor, gumbel: torch.Tensor, mode: str = "count") -> torch.Tensor:
    """`gumbel_pick` restricted to the candidates topk returns (the batched graphs' sampler).

    The full-vocabulary form was observed to miscompile at B > 1 inside large fused frame graphs
    on the device (near-threshold picks); this one takes the noise at the TOPK_CAND top indices and the argmax
    among those that are >= the k-th value. Identical draws to `gumbel_pick` whenever the entries
    tied with the k-th value fit in the candidates; a row where they might not (its last candidate
    still equals the k-th value) takes the full-vocabulary result instead, so the draw stays exact
    in every case. [B, V] -> [B] int64.

    mode "count" (default) finds the k-th value by counting (see below) and takes the full-vocabulary
    Gumbel-max under that threshold. mode "onehot" selects with one-hot masks and sums (exact: one nonzero term) instead
    of torch.gather, whose lowering made the B = 16 frame graph ~9x slower on the device
    (measured on trn2: 243 ms vs 27.6 ms); "gather" is that form, "onehot_noovf" drops the overflow fallback (benches).
    """
    s = scores.float() / TEMPERATURE
    if mode == "count":
        # The k-th value found by counting, from the exact scores at the topk indices: the value u
        # with count(s > u) < k <= count(s >= u). On the device topk's indices can be paired with
        # the wrong values around ties at the k-th value (every wrong pick observed was a tied
        # threshold element), so neither its values nor its indices are trusted beyond naming candidates.
        _, idx = torch.topk(s, TOPK_CAND, dim=-1)
        oh = idx[:, :, None] == torch.arange(s.shape[-1], device=s.device)[None, None, :]  # [B, C, V]
        u = torch.where(oh, s[:, None, :], 0.0).sum(-1)  # [B, C] scores of the candidates (one term)
        cge = (s[:, None, :] >= u[:, :, None]).sum(-1)
        cgt = (s[:, None, :] > u[:, :, None]).sum(-1)
        hit = (cgt < TOP_K) & (cge >= TOP_K)
        kth = torch.where(hit, u, NEG).amax(-1, keepdim=True)  # every hit has the same value
        # (no candidate at the k-th value cannot happen when topk names the top elements; the
        # startup self-test checks every code against the host Gumbel-max)
        v = torch.where(s >= kth, s + gumbel, NEG)
        return torch.argmax(v, dim=-1)
    if mode.startswith("full"):  # full-vocabulary forms (no topk indices), for benches
        kth = torch.topk(s, TOP_K, dim=-1).values[..., -1:]
        if mode == "full":
            v = torch.where(s >= kth, s + gumbel, NEG)
        elif mode == "fullx":  # the threshold materialized per element (no row broadcast in the compare)
            v = torch.where(s >= kth * torch.ones_like(s), s + gumbel, NEG)
        else:  # "fullarith": an additive mask instead of a select
            v = s + gumbel + (s < kth).float() * NEG
        return torch.argmax(v, dim=-1)
    vals, idx = torch.topk(s, TOPK_CAND, dim=-1)  # sorted, descending
    kth = vals[:, TOP_K - 1 : TOP_K]
    if mode == "gather":
        g = torch.gather(gumbel, 1, idx)
    else:
        oh = idx[:, :, None] == torch.arange(s.shape[-1], device=s.device)[None, None, :]  # [B, C, V]
        g = torch.where(oh, gumbel[:, None, :], 0.0).sum(-1)
    v = torch.where(vals >= kth, vals + g, NEG)
    if mode == "gather":
        c = torch.gather(idx, 1, torch.argmax(v, dim=-1, keepdim=True))[:, 0]
    else:
        j = torch.argmax(v, dim=-1)
        c = torch.where(torch.arange(TOPK_CAND, device=s.device)[None, :] == j[:, None], idx, 0).sum(-1)
    if mode == "onehot_noovf":
        return c
    overflow = vals[:, TOPK_CAND - 1] >= kth[:, 0]
    full = torch.argmax(torch.where(s >= kth, s + gumbel, NEG), dim=-1)
    return torch.where(overflow, full, c)


FREE_BISECT = int(os.environ.get("QWEN3TTS_FREE_BISECT", "12"))  # threshold bisection steps of the proposal sampler
FREE_ROUNDS = int(os.environ.get("QWEN3TTS_FREE_ROUNDS", "2"))  # candidate rounds after it


def gumbel_pick_free(scores: torch.Tensor, gumbel: torch.Tensor, rounds: int | None = None, bisect: int | None = None) -> torch.Tensor:
    """A topk-free *proposal* of `gumbel_pick` for the device chain: [B, V] -> [B] int64.

    Only elementwise ops, max, argmax and sums (no topk or gather: the device's topk was observed
    to pair indices with wrong values at ties). First `bisect` halvings of [min, max] find a threshold lo <= the k-th
    largest score (count(s >= lo) >= TOP_K), so {s >= lo} is a superset of the top-k set, tight
    after ~10 halvings. Then the winner c of argmax(s + G) over the allowed set is the exact draw
    when it lies in the top-k set, i.e. when fewer than TOP_K entries score strictly above it
    (exact with ties at the k-th value: the set keeps every entry >= the k-th largest). Otherwise
    every top-k entry scores above s_c, so the next round restricts to s > s_c (still a superset)
    and tries again; after `rounds` rounds the last candidate is returned as is. It is only a
    proposal: the host verifies every code with `gumbel_pick` and recomputes from the first wrong
    one (engine.batch), so the served codes never depend on this function being right.
    (On real predictor logits, CPU: 3 rounds without bisection miss 1.3% of codes; 12 halvings
    and 2 rounds miss 0 of 3000.)"""
    rounds = FREE_ROUNDS if rounds is None else rounds
    bisect = FREE_BISECT if bisect is None else bisect
    s = scores.float() / TEMPERATURE
    z = s + gumbel
    allowed = None
    if bisect:
        lo = s.amin(-1, keepdim=True)
        hi = s.amax(-1, keepdim=True)
        for _ in range(bisect):
            mid = (lo + hi) * 0.5
            ok = (s >= mid).float().sum(-1, keepdim=True) >= TOP_K
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        allowed = s >= lo
    res = done = None
    for _ in range(max(1, rounds)):
        zm = z if allowed is None else torch.where(allowed, z, NEG)
        mx = zm.amax(-1, keepdim=True)
        c = torch.argmax(zm, dim=-1)
        sc = torch.where(zm == mx, s, NEG).amax(-1, keepdim=True)  # the winner's score (max over z ties)
        above = s > sc
        valid = above.float().sum(-1) < TOP_K
        if res is None:
            res, done = c, valid
        else:
            res = torch.where(done, res, c)
            done = done | valid
        allowed = above
    return res


def gumbel_check(scores: torch.Tensor, gumbel: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Host verification of proposed draws: [N, V], [N, V], [N] -> [N] bool, True exactly where
    codes == gumbel_pick(scores, gumbel) (ties included), without the top-k sort.

    With s = scores / T and z = s + G, gumbel_pick returns the first index of max z over the
    top-k set {s >= k-th largest s}. c is that draw iff (1) c is in the set: fewer than TOP_K
    entries have s > s_c; and (2) no entry of the set beats it: among U = {z > z_c, or z == z_c
    at a lower index}, the entry with the largest s is outside the set (TOP_K or more entries
    score above it), or U is empty."""
    s = scores.float() / TEMPERATURE
    z = s + gumbel
    c = codes[:, None]
    sc = s.gather(1, c)
    zc = z.gather(1, c)
    in_set = (s > sc).sum(-1) < TOP_K
    idx = torch.arange(s.shape[-1])[None, :]
    U = (z > zc) | ((z == zc) & (idx < c))
    sU = torch.where(U, s, NEG).amax(-1, keepdim=True)
    beaten = U.any(-1) & ((s > sU).sum(-1) < TOP_K)
    return in_set & ~beaten


def gumbel_from_uniform(u: torch.Tensor) -> torch.Tensor:
    """Standard Gumbel noise from U ~ [0, 1) (clamped into (0, 1) so it stays finite)."""
    u = u.clamp(min=torch.finfo(torch.float32).tiny, max=1.0 - 2.0**-24)
    return -torch.log(-torch.log(u))


def gumbel_noise(shape, gen: torch.Generator | None = None) -> torch.Tensor:
    """Standard Gumbel noise in FP32 on the host, from U ~ (0, 1) of `gen`."""
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    u.clamp_(min=torch.finfo(torch.float32).tiny)  # torch.rand is [0, 1): exclude 0
    return -torch.log(-torch.log(u))


# ---------------------------------------------------------------------------- host reference
def _draw(scores: torch.Tensor, gen: torch.Generator | None) -> int:
    scores = scores / TEMPERATURE
    kth = torch.topk(scores, TOP_K).values[-1]
    scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = torch.softmax(scores, -1)
    return int(torch.multinomial(probs, 1, generator=gen)[0])


class Code0Sampler:
    """Host reference sampler of codebook 0 (penalty, suppression, min frames, softmax +
    multinomial), used by the tests."""

    def __init__(self, vocab: int, eos_id: int):
        self.eos_id = eos_id
        sup = torch.zeros(vocab, dtype=torch.bool)
        sup[vocab - 1024 :] = True
        sup[eos_id] = False
        self.suppress = sup

    def scores(self, logits: torch.Tensor, history: list[int]) -> torch.Tensor:
        s = logits.float().clone()
        if history:
            idx = torch.tensor(sorted(set(history)), dtype=torch.long)
            v = s[idx]
            s[idx] = torch.where(v < 0, v * REPETITION_PENALTY, v / REPETITION_PENALTY)
        s = s.masked_fill(self.suppress, float("-inf"))
        if len(history) < MIN_NEW_TOKENS:
            s[self.eos_id] = float("-inf")
        return s

    def __call__(self, logits: torch.Tensor, history: list[int], gen: torch.Generator | None = None) -> int:
        return _draw(self.scores(logits, history), gen)


def sample_sub(logits: torch.Tensor, gen: torch.Generator | None = None) -> int:
    return _draw(logits.float(), gen)
