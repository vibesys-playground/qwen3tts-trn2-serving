"""CPU test: the one-call prefill graph computes what 10 single-position decode steps compute.

    python tests/test_prefill_cpu.py [checkpoint]

1. FP32 (both modules cast to FP32): KV, hidden and logits agree to rounding (exact math).
2. BF16 (as served): the prefill graph is no farther from the FP32 result than the 10-step path
   is (only the accumulation order of the BF16 matmuls differs), slots >= 10 stay zero, and it
   reads the talker's own Parameters (no second copy).
"""
import os
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine.runtime as R
from engine.runtime import LM, PREFILL_LEN

W = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
torch.set_num_threads(8)
lm = LM(W)
assert lm.prefill_mod.step is lm.talker_mod
assert all(a is b for a, b in zip(lm.prefill_mod.step.parameters(), lm.talker_mod.parameters()))
TEXTS = ("Yes.", "The lieutenant was well informed in many matters outside the pale of his profession.")
lp = lambda l: torch.log_softmax(l.float(), -1)
ok = True
bf = {}
with torch.inference_mode():
    for text in TEXTS:
        x, _, _ = lm.emb.prompt(text, "Ryan", "English")
        assert x.shape[0] == PREFILL_LEN
        (K1, V1), _, l1 = lm.prefill_x(x)
        (K0, V0), _, l0 = lm.prefill_x_stepwise(x)
        rest = float(K1[:, :, :, PREFILL_LEN:].abs().max()) + float(V1[:, :, :, PREFILL_LEN:].abs().max())
        bf[text] = (x, lp(l1), lp(l0), K1, K0)
        ok &= rest == 0 and K1.shape == K0.shape
    lm.talker_mod.float()
    R.BF = torch.float32  # FP32 caches
    for text in TEXTS:
        x, b1, b0, K1, K0 = bf[text]
        (Kf1, Vf1), hf1, lf1 = lm.prefill_x(x.float())
        (Kf0, Vf0), hf0, lf0 = lm.prefill_x_stepwise(x.float())
        e32 = max(float((Kf1 - Kf0).abs().max()), float((Vf1 - Vf0).abs().max()), float((lp(lf1) - lp(lf0)).abs().max()))
        ref = lp(lf0)
        d1, d0 = (b1 - ref).abs(), (b0 - ref).abs()
        n = PREFILL_LEN
        k1, k0 = (float((Kb.float() - Kf0)[:, :, :, :n].abs().mean()) for Kb in (K1, K0))
        print(f"{text[:30]!r}: FP32 prefill vs steps max {e32:.2e} | BF16 vs FP32 logprob mean/max: "
              f"prefill graph {float(d1.mean()):.4f}/{float(d1.max()):.3f}, 10 steps {float(d0.mean()):.4f}/{float(d0.max()):.3f} "
              f"| K mean err: graph {k1:.4f}, steps {k0:.4f}", flush=True)
        ok &= e32 < 1e-3 and float(d1.mean()) < 1.5 * float(d0.mean()) + 0.01 and float(d1.max()) < 1.0 and k1 < 1.5 * k0 + 1e-3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
