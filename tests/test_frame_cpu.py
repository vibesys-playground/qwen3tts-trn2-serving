"""CPU tests of the fused frame graph (engine.model.FrameStep), run eagerly.

    python tests/test_frame_cpu.py [checkpoint]

1. generate: under the same noise, the frame graph's codes equal the previous host path (talker
   step, 16 code-predictor steps one launch each, host sampling, host embeddings) with a host
   Gumbel-max over -inf-masked scores.
2. sampler: on fixed logits with many ties at the top-k threshold, 20k in-graph draws
   (gumbel_pick) and 20k reference draws (_draw: softmax + multinomial) both fit the exact
   distribution (chi-square), and code0_scores equals the host Code0Sampler's scores.
3. score: /score through the frame graph equals the previous host path's scoring, and stays
   within the check's gate on two golden utterances.
"""
import os
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import chisquare
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.runtime import LM, MAX_POSITIONS
from engine.sampling import NEG, TEMPERATURE, TOP_K, Code0Sampler, _draw, code0_scores, gumbel_noise, gumbel_pick

W = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
G = json.loads((ROOT / "eval/golden/golden.json").read_text())
torch.set_num_threads(4)
fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


# ------------------------------------------------------------------ 2. sampler (no model)
def sampler_tests():
    g = torch.Generator().manual_seed(1)
    V = 2048
    # BF16-rounded logits on a coarse grid: many exact ties, including at the 50th value.
    lg = (torch.randn(V, generator=g) * 2).to(torch.bfloat16).float()
    lg = torch.round(lg * 4) / 4
    s = lg / TEMPERATURE
    kth = torch.topk(s, TOP_K).values[-1]
    keep = s >= kth
    p = torch.softmax(s.masked_fill(~keep, float("-inf")), -1).double()
    n_keep = int(keep.sum())
    check(n_keep > TOP_K, f"tie fixture keeps {n_keep} > {TOP_K} entries (ties at the k-th value)")
    N = 20000
    noise = gumbel_noise((N, V), g)
    picks = gumbel_pick(lg[None].expand(N, V), noise)
    ref = torch.tensor([_draw(lg, g) for _ in range(N)])
    idx = torch.nonzero(keep).ravel()
    check(bool(keep[picks].all()), "graph sampler never picks outside the top-k set")
    for name, d in (("graph (gumbel_pick)", picks), ("reference (_draw)", ref)):
        cnt = torch.bincount(d, minlength=V)[idx].double().numpy()
        exp = (p[idx] * N).numpy()
        big = exp >= 5  # pool small cells
        obs_c = np.append(cnt[big], cnt[~big].sum())
        exp_c = np.append(exp[big], exp[~big].sum())
        if exp_c[-1] == 0:
            obs_c, exp_c = obs_c[:-1], exp_c[:-1]
        pv = chisquare(obs_c, exp_c * obs_c.sum() / exp_c.sum()).pvalue
        check(pv > 1e-3, f"{name}: chi-square p = {pv:.3f} over {len(obs_c)} cells")

    # code0_scores vs Code0Sampler.scores (penalty, suppression, EOS mask)
    V0, eos = 3072, 2150
    smp = Code0Sampler(V0, eos)
    for hist in ([], [5], [5, 7, 5, 2000, 100]):
        l0 = torch.randn(V0, generator=g) * 3
        pres = torch.zeros(1, V0, dtype=torch.bool)
        if hist:
            pres[0, torch.tensor(hist)] = True
        a = code0_scores(l0[None], pres, torch.tensor([len(hist)]), smp.suppress, eos)[0]
        b = smp.scores(l0, hist)
        b = torch.where(torch.isinf(b), torch.full_like(b, NEG), b)
        check(torch.equal(a, b), f"code0_scores == Code0Sampler.scores for history {hist}")


# ------------------------------------------------------------------ previous host path (reference)
def ref_pick(scores, gum):
    s = scores.float() / TEMPERATURE
    kth = torch.topk(s, TOP_K).values[-1]
    s = s.masked_fill(s < kth, float("-inf"))
    return int(torch.argmax(s + gum))


def ref_frame_codes(lm, h, code0, gum=None, given=None):
    p_shape = (lm.pred_mod.L, 1, lm.pred_mod.kvh, lm.n_groups, lm.pred_mod.hd)
    K = torch.zeros(p_shape, dtype=torch.bfloat16)
    V = torch.zeros(p_shape, dtype=torch.bfloat16)
    _, _, K, V = lm.pred_mod(h, lm.pos(0), K, V)
    codes, subs = [code0], []
    x = lm.emb.code0_embed(code0)
    f = lm.frame_mod
    for g in range(1, lm.n_groups):
        _, logits, K, V = lm.pred_mod(x.reshape(1, -1), lm.pos(g), K, V)
        lg = logits[0].float()
        subs.append(lg)
        o = f.o_g1 + (g - 1) * f.V1
        c = int(given[g]) if given is not None else ref_pick(lg, gum[o : o + f.V1])
        codes.append(c)
        if g < lm.n_groups - 1:
            x = lm.emb.cp_embed(g, c)
    return codes, torch.stack(subs)


@torch.inference_mode()
def ref_generate(lm, text, gen, n_max):
    cache, h, logits, trailing, n_ids, pos = lm.prefill(text, "Ryan", "English")
    smp = Code0Sampler(lm.frame_mod.V0, lm.eos_id)
    hist, out = [], []
    for t in range(min(n_max, MAX_POSITIONS - n_ids)):
        gum = gumbel_noise((1, lm.frame_mod.o_trail), gen)[0]
        tok = ref_pick(smp.scores(logits[0], hist), gum[: lm.frame_mod.V0])
        if tok == lm.eos_id:
            break
        hist.append(tok)
        codes, _ = ref_frame_codes(lm, h, tok, gum=gum)
        out.append(codes)
        x = lm.emb.frame_input(codes, trailing, t)
        h, logits = lm.talker_step(x.reshape(1, -1), pos, cache)
        pos += 1
    return out


@torch.inference_mode()
def ref_score(lm, text, codes):
    cache, h, logits, trailing, _, pos = lm.prefill(text, "Ryan", "English")
    out = []
    for t, fr in enumerate(codes):
        lp0 = torch.log_softmax(logits[0].float(), -1)
        _, sub = ref_frame_codes(lm, h, int(fr[0]), given=fr)
        lps = torch.log_softmax(sub, -1)
        out.append([float(lp0[int(fr[0])])] + [float(lps[g - 1, int(fr[g])]) for g in range(1, lm.n_groups)])
        x = lm.emb.frame_input([int(c) for c in fr], trailing, t)
        h, logits = lm.talker_step(x.reshape(1, -1), pos, cache)
        pos += 1
    return out, float(torch.log_softmax(logits[0].float(), -1)[lm.eos_id])


def model_tests():
    lm = LM(W)
    f = lm.frame_mod
    check(f.codec_emb.dtype == torch.bfloat16 and torch.equal(f.codec_emb.float(), lm.emb.codec_emb), "codec_emb table equals the host FP32 table")
    check(torch.equal(f.cp_emb.float().view(lm.emb.cp_emb.shape), lm.emb.cp_emb), "cp_emb table equals the host FP32 table")

    # 1. generate parity under identical noise
    for text, seed in (("Hello there, this is a test.", 0), ("Real dishonour consists in sin.", 7)):
        n = 12
        t0 = time.time()
        a = []
        for codes in lm.generate(text, "Ryan", "English", gen=torch.Generator().manual_seed(seed)):
            a.append(codes)
            if len(a) >= n:
                break
        t1 = time.time()
        b = ref_generate(lm, text, torch.Generator().manual_seed(seed), n)
        check(a == b, f"generate '{text}' seed {seed}: {len(a)} frames, frame graph == host path ({t1 - t0:.1f}s vs {time.time() - t1:.1f}s)")
        if a != b:
            print(" graph:", a[:3], "\n host: ", b[:3])

    # 3. score parity and gate
    for c in (G["cases"][8], G["cases"][0]):
        codes = c["codes"]
        lp, e = lm.score(c["text"], "Ryan", "English", codes)
        lr, er = ref_score(lm, c["text"], codes)
        d = np.abs(np.array(lp) - np.array(lr))
        check(d.max() < 1e-4 and abs(e - er) < 1e-4, f"score {c['name']} T={len(codes)}: frame graph vs host path max |d| {d.max():.2e}, eos {abs(e - er):.2e}")
        g = np.abs(np.array(lp) - np.array(c["logprobs"]))
        check(g.mean() < 0.08 and g.max() < 2.0 and abs(e - c["eos_logprob"]) < 0.5, f"score {c['name']} vs golden FP32: mean {g.mean():.4f} max {g.max():.3f} eos {abs(e - c['eos_logprob']):.3f}")


sampler_tests()
if "--sampler-only" not in sys.argv:
    model_tests()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
