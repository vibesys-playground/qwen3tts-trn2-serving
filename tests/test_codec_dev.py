"""CPU test: the batched slot decoder (engine.codec_dev, eager BF16 on the CPU) against the host
FP32 streaming decoder and the reference decode of the golden codes, all ten streams decoded
concurrently with staggered starts (slots reused), as the server's codec thread drives it."""
import os
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.codec import CodecWeights, to_pcm16
from engine.codec_dev import DeviceCodec, snr_db
G = ROOT / "eval/golden"
W = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")

torch.set_num_threads(8)
cw = CodecWeights(W)
cases = json.loads((G / "golden.json").read_text())["cases"]
pcm = np.load(G / "golden_pcm.npz")
dc = DeviceCodec(cw, "cpu", groups=(1, 2), n_groups=3)  # 24 slots, ticks of up to 2 calls
# Pre-occupy a few slots, then free them, so streams land in reused slots of two groups.
junk = [dc.open() for _ in range(14)]
dc.decode({s: cases[0]["codes"][0] for s in junk})
for s in junk[:3]:
    dc.close(s)
live_junk = junk[3:]  # 11 streams that keep decoding: the golden ones span three groups (two calls a tick)
streams = []  # [case, slot, next frame, chunks]
pending = list(cases)
t0 = time.time(); calls = 0
while pending or streams:
    if pending and (calls % 3 == 0 or not streams):
        c = pending.pop(0)
        streams.append([c, dc.open(), 0, []])
    batch = {st[1]: st[0]["codes"][st[2]] for st in streams}
    batch.update({s: cases[1]["codes"][calls % 29] for s in live_junk})
    res = dc.decode(batch)
    calls += 1
    for st in list(streams):
        st[3].append(res[st[1]]); st[2] += 1
        if st[2] == len(st[0]["codes"]):
            dc.close(st[1]); streams.remove(st)
            got = np.frombuffer(to_pcm16(np.concatenate(st[3])), dtype="<i2")
            ref = pcm[st[0]["name"]]
            assert len(got) == len(ref), (len(got), len(ref))
            s = snr_db(ref, got)
            win = [snr_db(ref[i:i + 19200], got[i:i + 19200]) for i in range(0, len(ref) - 19199, 19200) if np.abs(ref[i:i + 19200]).max() > 300]
            print(f"{st[0]['name']}: {len(st[0]['codes'])} frames, SNR {s:.1f} dB, worst window {min(win) if win else float('nan'):.1f} dB")
            assert s >= 25, s
print(f"OK {calls} calls in {time.time() - t0:.1f}s")

# The startup self-test past the 72-frame window (every slot's KV ring wraps), with a stream that
# skips calls and a fresh stream in a reused slot, on a fresh 24-slot codec.
from engine.server import codec_selftest
t0 = time.time()
dc2 = DeviceCodec(cw, "cpu", groups=(1, 2), n_groups=3)
w = codec_selftest(dc2, cw, frames=80)
print(f"OK self-test, ring wrapped: worst SNR {w:.1f} dB over {dc2.max_batch} + 8 streams in {time.time() - t0:.1f}s")
assert w >= 25, w
