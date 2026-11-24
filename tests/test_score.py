"""CPU test: teacher-forced /score numerics against the golden FP32 reference (the check's gate)."""
import os
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.runtime import LM
G = ROOT / "eval/golden/golden.json"
W = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
torch.set_num_threads(8)
lm = LM(W)
cases = json.loads(G.read_text())["cases"][:n]
cases.sort(key=lambda c: c["frames"])
deltas, eos = [], []
for c in cases:
    t0 = time.time()
    lp, e = lm.score(c["text"], "Ryan", "English", c["codes"])
    d = np.abs(np.array(lp) - np.array(c["logprobs"]))
    deltas.append(d.ravel()); eos.append(abs(e - c["eos_logprob"]))
    print(f"{c['name']} T={c['frames']}: mean {d.mean():.4f} max {d.max():.3f} cb0 mean {d[:,0].mean():.4f} eos |d| {eos[-1]:.3f} ({time.time()-t0:.1f}s)", flush=True)
d = np.concatenate(deltas)
print(f"ALL: mean {d.mean():.4f} p99 {np.quantile(d, .99):.4f} max {d.max():.4f} eos max {max(eos):.3f}")
