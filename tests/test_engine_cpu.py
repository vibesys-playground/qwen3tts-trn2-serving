"""In-process CPU smoke test of the serving engine (no device, no HTTP server process).

    QWEN3TTS_WEIGHTS=<checkpoint> python tests/test_engine_cpu.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("QWEN3TTS_BUCKETS", "2,4")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "3")
os.environ.setdefault("QWEN3TTS_TIMING", "0")
os.environ.setdefault("QWEN3TTS_CODEC_GROUPS", "1")  # one 8-slot bucket: the CPU runs the codec graph eagerly
from engine import server

G = json.loads((ROOT / "eval/golden/golden.json").read_text())
eng = server.Engine("cpu", server.Watchdog(1e9))
app = server.make_app(eng)
for r in app.routes:
    if getattr(r, "path", "") in ("/generate", "/score"):
        assert not r.dependant.query_params, (r.path, r.dependant.query_params)
print("routes", sorted(r.path for r in app.routes if hasattr(r, "path")))
c = G["cases"][8]  # "Yes."
lp, eos = eng.score(c["text"], "Ryan", "English", c["codes"])
print("score mean |d|", float(np.abs(np.array(lp) - np.array(c["logprobs"])).mean()), "eos", eos, c["eos_logprob"])
pcm = b"".join(eng.codes_stream(c["codes"]))
print("codes mode samples", len(pcm) // 2, "expected", 1920 * c["frames"])
for text in ("Hello there, this is a test.", "Hello there, this is a test."):
    t0 = time.time(); chunks = list(eng.generate_stream(text, "Ryan", "English"))
    n = sum(len(x) for x in chunks) // 2
    print(f"generate: {len(chunks)} chunks, {n / 24000:.2f}s audio in {time.time() - t0:.1f}s, md5", hash(b"".join(chunks)))
