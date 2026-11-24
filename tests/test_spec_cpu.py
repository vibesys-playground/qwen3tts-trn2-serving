"""CPU tests of the speculative predictor chain, eager, small buckets.

    QWEN3TTS_BUCKETS=2,4 python tests/test_spec_cpu.py [checkpoint]

1. proposal sampler: gumbel_pick_free == gumbel_pick on random scores, BF16-like scores with ties
   at the k-th value, and tie overflow; and its miss rate when the noise favours a low-ranked entry.
2. PredChain fully forced == today's PredFirst/PredNext logits (same math, other op order).
3. exactness: for real hidden states, the spec codes (verify + fix-up) equal the host's sequential
   codes (PredFirst/PredNext + host Gumbel-max per code, the non-speculative path) on the same noise,
   clean and with corrupted proposals at random codebooks (rounding ties excused and counted).
4. the replica self-test in spec mode (recheck every step, injected corruption every 3rd step).
5. a broken verify (accept every proposal) fails the self-test.
"""
import os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QWEN3TTS_BUCKETS", "2,4")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "3")
from engine.sampling import TOP_K, TEMPERATURE, NEG, gumbel_noise, gumbel_pick, gumbel_pick_free

W = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
torch.set_num_threads(8)
fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


g = torch.Generator().manual_seed(0)
V = 2048
s = torch.randn((2048, V), generator=g) * 3
n = gumbel_noise((2048, V), g)
check(bool((gumbel_pick_free(s, n) == gumbel_pick(s, n)).all()), "free sampler == exact, random scores")
s = torch.round(torch.randn((2048, V), generator=g) * 8) / 4
n = gumbel_noise((2048, V), g)
kth = torch.topk(s / TEMPERATURE, TOP_K, -1).values[:, -1:]
check(int(((s / TEMPERATURE) >= kth).sum(-1).max()) > TOP_K, "tie fixture has ties at the k-th value")
check(bool((gumbel_pick_free(s, n) == gumbel_pick(s, n)).all()), "free sampler == exact, tied scores")
s = torch.zeros((8, V))
s[:, :40] = 5.0
n = gumbel_noise((8, V), g)
check(bool((gumbel_pick_free(s, n) == gumbel_pick(s, n)).all()), "free sampler == exact, tie overflow")
# the host verification (no sort) == comparing with gumbel_pick, on random, tied and overflow
# scores, with proposals that are right, random, or a near-miss (the runner-up)
from engine.sampling import gumbel_check
for name, sc in (("random", torch.randn((2048, V), generator=g) * 3), ("tied", torch.round(torch.randn((2048, V), generator=g) * 8) / 4),
                 ("overflow", torch.cat([torch.full((64, 40), 5.0), torch.zeros((64, V - 40))], 1))):
    nz = gumbel_noise(sc.shape, g)
    w = gumbel_pick(sc, nz)
    sv = sc / TEMPERATURE
    kth = torch.topk(sv, TOP_K, -1).values[:, -1:]
    second = torch.topk(torch.where(sv >= kth, sv, NEG) + nz, 2, -1).indices[:, 1]
    for pname, prop in (("right", w), ("random", torch.randint(0, V, w.shape, generator=g)), ("runner-up", second),
                        ("mixed", torch.where(torch.rand(w.shape, generator=g) < 0.5, w, second))):
        check(bool((gumbel_check(sc, nz, prop) == (prop == w)).all()), f"gumbel_check == (code == gumbel_pick), {name} scores, {pname} proposals")
# exact z ties: two entries with the same score and noise; the lower index wins
sc = torch.zeros((4, V)); sc[:, 10] = 3.0; sc[:, 5] = 3.0
nz = torch.zeros((4, V))
w = gumbel_pick(sc, nz)
check(bool((w == 5).all()) and bool(gumbel_check(sc, nz, w).all()) and not bool(gumbel_check(sc, nz, torch.full((4,), 10)).any()),
      "gumbel_check keeps argmax's first-index rule on exact ties")
# adversarial: flat-ish scores with one low-ranked entry given huge noise -> needs later rounds
s = torch.randn((512, V), generator=g) * 0.5
n = gumbel_noise((512, V), g)
low = torch.argsort(s, -1)[:, 100]
n[torch.arange(512), low] = 50.0
d = int((gumbel_pick_free(s, n) != gumbel_pick(s, n)).sum())
d1 = int((gumbel_pick_free(s, n, rounds=1, bisect=0) != gumbel_pick(s, n)).sum())
print(f"     adversarial: misses {d}/512 with the default sampler, {d1}/512 with 1 round and no bisection (the host verify fixes these)")
check(d1 > 0, "adversarial fixture defeats one round (the verify path is needed)")

from engine import replica
from engine.batch import BatchLM, Job
from engine.runtime import LM

lm = LM(W)
t0 = time.time()
refs = replica.host_refs(lm)
print(f"host refs {time.time() - t0:.1f}s", flush=True)
events = []
blm = BatchLM(lm, events.extend, compile_=False, log=print, spec=True)


def seq_codes(blm, h, c0, n1):
    """The non-speculative path: PredFirst, then PredNext per code, each code the host's draw."""
    B = h.shape[0]
    lg, Kp, Vp = blm.predfirst(h, c0)
    codes, subs = [], []
    for gg in range(1, blm.G):
        subs.append(lg)
        c = gumbel_pick(lg, n1[:, gg - 1])
        codes.append(c)
        if gg + 1 < blm.G:
            lg, Kp, Vp = blm.prednext(c, blm.gpos[(B, gg + 1)], blm.ghead[gg + 1], Kp, Vp)
    return torch.stack(codes, 1), torch.stack(subs, 1)


def margins(sc, nz):
    s = sc.float() / TEMPERATURE
    kth = torch.topk(s, TOP_K, -1).values[..., -1:]
    v = torch.where(s >= kth, s, NEG) + nz
    t2 = torch.topk(v, 2, -1).values
    return t2[..., 0] - t2[..., 1]


# real hidden states: 4 streams a few frames in
blm.recheck = False
js = [Job(("x", i), "gen", replica.SELFTEST_TEXTS[i % 4], max_frames=40) for i in range(4)]
for j in js:
    blm.add(j)
for _ in range(6):
    blm.step()
B = blm.B
h = blm.st[0]
gen = torch.Generator().manual_seed(5)
with torch.inference_mode():
    tot = diff = seqdiff = 0
    maxd = meand = 0.0
    for trial in range(12):
        c0 = torch.randint(0, 2048, (B,), generator=gen)
        n1 = gumbel_noise((B, blm.G - 1, blm.V1), gen)
        ref, ref_sub = seq_codes(blm, h, c0, n1)
        _, sd = blm.chain(h, c0, n1.reshape(B, -1), ref)  # forced chain vs sequential logits on the same codes
        maxd = max(maxd, float((sd - ref_sub).abs().max()))
        meand = max(meand, float((sd - ref_sub).abs().mean()))
        got = {}
        for corrupt in (False, True):
            blm.inject = None
            if corrupt:
                cg = torch.Generator().manual_seed(trial)
                blm.inject = lambda B_, ver: [(r, int(torch.randint(1, 16, (1,), generator=cg)), int(torch.randint(0, 2048, (1,), generator=cg)))
                                              for r in ver.nonzero()[:, 0].tolist()]
            blm.phase.clear()
            got[corrupt] = blm._spec_codes(B, h, c0, n1, torch.full((B, 15), -1, dtype=torch.long), torch.ones(B, dtype=torch.bool))[0]
            if corrupt:
                check(blm.phase["passes"] > 0 or blm.phase["mismatches"] > 0, f"trial {trial}: corruption triggered fix-up ({dict(blm.phase)})")
        tot += got[True].numel()
        diff += int((got[True] != got[False]).sum())
        seqdiff += int(((got[False] != ref).cumsum(1) > 0).any(1).sum())
    blm.inject = None
# a bucket run on a larger chain graph, padded (QWEN3TTS_CHAIN_PAD): same codes and logits per row
with torch.inference_mode():
    c0 = torch.randint(0, 2048, (2,), generator=gen)
    n1 = gumbel_noise((2, blm.G - 1, blm.V1), gen)
    fo = torch.full((2, 15), -1, dtype=torch.long)
    saved_pad = blm.chain_pad
    blm.chain_pad = {}
    ref_c, ref_s = blm._chain_host(2, h[:2], c0, n1, fo)
    blm.chain_pad = {2: 4}
    pc, ps = blm._chain_host(2, h[:2], c0, n1, fo)
    blm.phase.clear()
    blm.inject = lambda B_, ver: [(0, 3, 7)]
    pcodes, _ = blm._spec_codes(2, h[:2], c0, n1, fo, torch.ones(2, dtype=torch.bool))
    blm.inject = None
    blm.chain_pad = {}
    ucodes, _ = blm._spec_codes(2, h[:2], c0, n1, fo, torch.ones(2, dtype=torch.bool))
blm.chain_pad = saved_pad
check(bool((pc == ref_c).all()) and float((ps - ref_s).abs().max()) < 0.05 and bool((pcodes == ucodes).all()),
      f"padded chain (2 rows on the 4-row graph): same proposals, logits max |diff| {float((ps - ref_s).abs().max()):.2e}, corrupted+fix-up codes == unpadded")
check(diff == 0, f"corrupted proposals + fix-up == clean speculative codes, bit for bit ({tot} codes, {diff} differ)")
print(f"     vs the sequential PredFirst/PredNext path (other op order): logits max |diff| {maxd:.3f}, worst mean {meand:.2e}; rows whose codes differ somewhere {seqdiff}/{12 * B}")
# the BF16 difference is rounding order only: in FP32 the chain and the sequential graphs agree
import copy
from engine.model import PredChain, PredFirst, PredNext
f32 = copy.deepcopy(lm.frame_mod).float()
with torch.inference_mode():
    gg_ = torch.Generator().manual_seed(1)
    hh = torch.randn(4, 2048, generator=gg_)
    cc0 = torch.randint(0, 2048, (4,), generator=gg_)
    nn1 = gumbel_noise((4, 15, 2048), gg_)
    lg, Kp, Vp = PredFirst(f32)(hh, cc0)
    cs, ss = [], []
    for k in range(1, 16):
        ss.append(lg)
        cs.append(gumbel_pick(lg, nn1[:, k - 1]))
        if k < 15:
            lg, Kp, Vp = PredNext(f32)(cs[-1], torch.full((4,), k + 1), torch.tensor([k]), Kp, Vp)
    cd32, sd32 = PredChain(f32)(hh, cc0, nn1.reshape(4, -1), torch.full((4, 15), -1))
    m32 = float((sd32 - torch.stack(ss, 1)).abs().max())
check(m32 < 1e-3 and bool((cd32[:, 1:] == torch.stack(cs, 1)).all()), f"FP32: chain logits == sequential logits (max |diff| {m32:.1e}) and its proposals == the sequential host codes")
del f32
for j in js:
    blm.cancel(j.jid)
blm.run_until_idle(100)

t0 = time.time()
f = replica.selftest(blm, refs, print)
check(not f, f"replica self-test, spec mode, buckets {blm.buckets} ({time.time() - t0:.1f}s): {f}")

# a broken verify (proposals accepted as they are) must fail the self-test when proposals are corrupted
import engine.batch as eb
saved = eb.gumbel_check
eb.gumbel_check = lambda scores, gumbel, codes: torch.ones(codes.shape[0], dtype=torch.bool)  # accept every proposal
f = replica.selftest(blm, refs, lambda m: None)
eb.gumbel_check = saved
check(any("mismatch" in x for x in f), f"a verify that accepts every proposal fails the self-test on sampler mismatches ({len(f)} failures: {[x[:100] for x in f]})")
print("FAILED:" if fails else "ALL OK", fails, flush=True)
sys.exit(1 if fails else 0)
