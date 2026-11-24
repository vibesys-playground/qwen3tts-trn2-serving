"""CPU tests of the continuous batch (engine.batch.BatchLM), run eagerly with small buckets.

    QWEN3TTS_BUCKETS=2,4 python tests/test_batch_cpu.py [checkpoint]

1. sampler: gumbel_pick_topk == gumbel_pick (full vocabulary) on random scores, on BF16-rounded
   scores with ties at the k-th value, and on rows whose ties overflow the candidates.
2. the replica's startup self-test (engine.replica.selftest): score rows vs the host serial path,
   sampled rows vs a host Gumbel-max, bucket growth and shrink with live rows, the B = 1 path.
3. cancel and position cap bookkeeping on a short run.
"""
import os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QWEN3TTS_BUCKETS", "2,4")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "3")
from engine.sampling import TOP_K, TOPK_CAND, gumbel_noise, gumbel_pick, gumbel_pick_topk

W = next((a for a in sys.argv[1:] if not a.startswith("--")), None) or os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
torch.set_num_threads(8)
fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


g = torch.Generator().manual_seed(0)
for V in (3072, 2048):
    s = torch.randn((256, V), generator=g) * 3
    n = gumbel_noise((256, V), g)
    check(bool((gumbel_pick_topk(s, n) == gumbel_pick(s, n)).all()), f"topk sampler == full sampler, random scores V={V}")
    s = torch.round(torch.randn((256, V), generator=g) * 8) / 4  # coarse grid: ties at the k-th value
    n = gumbel_noise((256, V), g)
    kth = torch.topk(s / 0.9, TOP_K, -1).values[:, -1:]
    check(int(((s / 0.9) >= kth).sum(-1).max()) > TOP_K, "tie fixture has ties at the k-th value")
    check(bool((gumbel_pick_topk(s, n) == gumbel_pick(s, n)).all()), f"topk sampler == full sampler, tied scores V={V}")
    s = torch.zeros((8, V))
    s[:, :40] = 5.0  # 40 above, then V - 40 entries tied at the k-th value: overflow
    n = gumbel_noise((8, V), g)
    check(bool((gumbel_pick_topk(s, n) == gumbel_pick(s, n)).all()), f"topk sampler == full sampler, tie overflow V={V} (> {TOPK_CAND} candidates)")

from engine import replica
from engine.batch import BatchLM, Job
from engine.runtime import LM

lm = LM(W)
t0 = time.time()
refs = replica.host_refs(lm)
print(f"host refs {time.time() - t0:.1f}s", flush=True)
events = []
blm = BatchLM(lm, events.extend, compile_=False, log=print)
t0 = time.time()
f = replica.selftest(blm, refs, print)
check(not f, f"replica self-test on buckets {blm.buckets} ({time.time() - t0:.1f}s): {f}")

# A broken batch must fail the self-test: swap two rows' small state after every prefill.
orig = blm.prefill
def bad_prefill(x, rows, K, V, *st):
    out = list(orig(x, rows, K, V, *st))
    out[0] = out[0].flip(0)
    return tuple(out)
blm.prefill = bad_prefill
f = replica.selftest(blm, refs, lambda m: None)
check(bool(f), f"a row-mixing bug fails the self-test ({len(f)} failures)")
blm.prefill = orig

# cancel: a cancelled stream leaves at the next step with a done event
events.clear()
j1 = Job("a", "gen", "Hello there, this is a cancel test.")
j2 = Job("b", "gen", "Second stream.")
blm.add(j1); blm.add(j2)
for _ in range(3):
    blm.step()
blm.cancel("a")
blm.run_until_idle(400)
kinds = [(e[0], e[1]) for e in events if e[1] != "frame"]
check(("a", "done") in kinds and ("b", "done") in kinds, f"cancel and EOS end both streams: {kinds}")
fr_a = sum(1 for e in events if e[0] == "a" and e[1] == "frame")
check(fr_a <= 4, f"cancelled stream stopped after {fr_a} frames")
print("FAILED:" if fails else "ALL OK", fails, flush=True)
sys.exit(1 if fails else 0)
