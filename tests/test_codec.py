"""CPU test: the streaming codec decode against the reference decode of the golden codes."""
import os
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.codec import CodecWeights, StreamDecoder, to_pcm16
G = ROOT / "eval/golden"
W = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")

def snr(ref, test):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    n = np.sum((ref - test) ** 2)
    return float("inf") if n == 0 else 10 * np.log10(np.sum(ref**2) / max(n, 1e-12))

torch.set_num_threads(8)
cw = CodecWeights(W)
cases = json.loads((G / "golden.json").read_text())["cases"]
pcm = np.load(G / "golden_pcm.npz")
for chunk in (1, 4, 10):
    worst = 1e9; t0 = time.time(); frames = 0
    for c in cases:
        d = StreamDecoder(cw); codes = np.array(c["codes"]); out = []
        for i in range(0, len(codes), chunk):
            out.append(d.push(codes[i:i + chunk]))
        got = np.frombuffer(to_pcm16(np.concatenate(out)), dtype="<i2")
        ref = pcm[c["name"]]
        assert len(got) == len(ref)
        worst = min(worst, snr(ref, got)); frames += len(codes)
    dt = time.time() - t0
    print(f"chunk {chunk}: worst SNR {worst:.1f} dB, {dt:.1f}s for {frames} frames ({frames*0.08/dt:.1f}x realtime)")
