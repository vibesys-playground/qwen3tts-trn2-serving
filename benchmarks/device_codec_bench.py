"""Device codec benchmark (needs one NeuronCore; run inside the server image).

Compares the host and device streaming codec decoders and checks the device decode's accuracy:

(a) the host FP32 streaming decoder (engine.codec), frames/s with 1 and 4 concurrent streams,
    pushing one frame at a time, QWEN3TTS_CPU_THREADS (default 4) torch threads;
(b) the batched device decoder (engine.codec_dev) on the first visible NeuronCore: compile time
    of each bucket, ms per call and frames/s with every slot active, and a lone stream;
(c) SNR of the device decode against the host FP32 decode and the reference decode
    (eval/golden/golden_pcm.npz) on the ten golden code sequences, decoded concurrently, per
    utterance and the worst non-silent 10-frame window.

    NEURON_RT_VISIBLE_CORES=0 QWEN3TTS_WEIGHTS=<checkpoint> python benchmarks/device_codec_bench.py

It was written to decide whether the codec belongs on a NeuronCore at all (it does: the host FP32
decoder is far below the codec throughput the server needs).
"""
import json, os, sys, threading, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
T0 = time.time()
from engine import server  # noqa: E402

server.setup_env()  # compile caches under $VIBESYS_CACHE_DIR, cwd in a fresh temp dir
import torch  # noqa: E402
from engine.codec import CodecWeights, StreamDecoder, to_pcm16  # noqa: E402
from engine.codec_dev import GROUP, DeviceCodec, snr_db  # noqa: E402

COMPILE_LIMIT = float(os.environ.get("CODEC_COMPILE_LIMIT_S", "2400"))
G = ROOT / "eval/golden"
cases = json.loads((G / "golden.json").read_text())["cases"]
ref_pcm = np.load(G / "golden_pcm.npz")
torch.set_num_threads(int(os.environ.get("QWEN3TTS_CPU_THREADS", "4")))
print({k: v for k, v in os.environ.items() if "NEURON" in k or "VIBESYS" in k}, f"torch threads {torch.get_num_threads()}, cpus {os.cpu_count()}", flush=True)
cw = CodecWeights(server.WEIGHTS)


def windows(ref, got):
    w = 10 * 1920
    out = [snr_db(ref[i:i + w], got[i:i + w]) for i in range(0, len(ref) - w + 1, w) if np.abs(ref[i:i + w]).max() > 300]
    return min(out) if out else float("nan")


# ----------------------------------------------------------------- (a) host FP32 codec
host_pcm = {}
for c in cases:
    d = StreamDecoder(cw)
    host_pcm[c["name"]] = np.frombuffer(to_pcm16(np.concatenate([d.push([f]) for f in c["codes"]])), dtype="<i2")
codes = [f for c in cases for f in c["codes"]]


def host_run(nstreams, nframes):
    def one():
        d = StreamDecoder(cw)
        for i in range(nframes):
            d.push([codes[i % len(codes)]])
    th = [threading.Thread(target=one) for _ in range(nstreams)]
    t = time.time()
    for x in th:
        x.start()
    for x in th:
        x.join()
    return nstreams * nframes / (time.time() - t)


for ns in (1, 4):
    fps = host_run(ns, 40)
    print(f"HOST fp32 {ns} stream(s), 1 frame/push, {torch.get_num_threads()} threads: {fps:.1f} frames/s", flush=True)
t = time.time(); d = StreamDecoder(cw); d.push([codes[0]])
print(f"HOST first push of a new stream: {1000 * (time.time() - t):.1f} ms", flush=True)

# ----------------------------------------------------------------- (b) device codec
dev = server.pick_device()
stage = {"name": "init", "t": time.time()}


def watchdog():
    while True:
        time.sleep(10)
        if time.time() - stage["t"] > COMPILE_LIMIT:
            print(f"STALLED > {COMPILE_LIMIT:.0f}s in {stage['name']}; exiting", flush=True)
            os._exit(3)


threading.Thread(target=watchdog, daemon=True).start()
t = time.time()
dc = DeviceCodec(cw, dev, groups=tuple(int(g) for g in os.environ.get("QWEN3TTS_CODEC_GROUPS", "2,4").split(",")))
print(f"DEV {dev}: weights on device in {time.time() - t:.1f}s; {dc.slots} slots, buckets {[b * GROUP for b in dc.buckets]}", flush=True)
for nb in dc.buckets:
    stage.update(name=f"compile B={nb * GROUP}", t=time.time())
    t = time.time()
    dc._call(list(range(nb)), {})
    print(f"DEV compile+first call B={nb * GROUP}: {time.time() - t:.1f}s", flush=True)
stage.update(name="timing", t=time.time())


def timed(nb, nactive, iters=20):
    slots = [dc.open() for _ in range(nactive)]
    fr = {s: codes[s % len(codes)] for s in slots}
    groups = sorted({s // GROUP for s in slots}) or [0]
    groups += [g for g in range(dc.n_groups) if g not in groups][: nb - len(groups)]
    dc._call(groups, fr)
    t = time.time()
    for _ in range(iters):
        dc._call(groups, fr)
    ms = 1000 * (time.time() - t) / iters
    for s in slots:
        dc.close(s)
    return ms


for nb, na in [(dc.buckets[0], 1)] + [(b, b * GROUP) for b in dc.buckets]:
    ms = timed(nb, na)
    print(f"DEV B={nb * GROUP} bucket, {na} active: {ms:.2f} ms/call (incl. host quantize, H2D, PCM D2H) -> {na * 1000 / ms:.0f} frames/s", flush=True)
# Graph-only: the two graphs of a call chained, state updated, PCM D2H only at the end of each call.
for nb in dc.buckets:
    groups = list(range(nb)); B = nb * GROUP
    rows, args = dc._prep(groups, {})
    dc._run(groups, args)
    t = time.time()
    for _ in range(20):
        dc._run(groups, args)
    ms = 1000 * (time.time() - t) / 20
    print(f"DEV graphs B={B}: {ms:.2f} ms/call -> {B * 1000 / ms:.0f} frames/s", flush=True)

# ----------------------------------------------------------------- (c) SNR, concurrent streams
stage.update(name="snr", t=time.time())
live = [[c, dc.open(), 0, []] for c in cases]
while live:
    res = dc.decode({s: c["codes"][i] for c, s, i, _ in live})
    for st in list(live):
        st[3].append(res[st[1]]); st[2] += 1
        if st[2] == len(st[0]["codes"]):
            dc.close(st[1]); live.remove(st)
            got = np.frombuffer(to_pcm16(np.concatenate(st[3])), dtype="<i2")
            n = st[0]["name"]
            print(f"SNR {n} ({len(st[0]['codes'])} frames): vs host fp32 {snr_db(host_pcm[n], got):.1f} dB, vs reference {snr_db(ref_pcm[n], got):.1f} dB, "
                  f"worst 10-frame window {windows(ref_pcm[n], got):.1f} dB", flush=True)
print(f"DONE {time.time() - T0:.1f}s", flush=True)
