"""CPU test: the deadline-ordered codec scheduler (engine.codec_dev.CodecService) gives the same
PCM per stream as the older tick scheduler (a copy below), on the eager BF16 CPU
DeviceCodec. A mix of fresh, continuing and backlogged streams is pushed in a random interleaving
(bursts, staggered starts, slot reuse) into both, from the same driver, and every stream's bytes
are compared. Also runs the startup scheduler self-test on the CPU.

    python tests/test_codec_sched.py [weights dir]
"""
import os
import sys, threading, time, traceback
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.codec import CodecWeights, to_pcm16
from engine.codec_dev import CodecService, CodecStream, DeviceCodec, scheduler_selftest, snr_db

W = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


class OldStream(CodecStream):
    pass


class OldTickService:
    """The older tick-based CodecService._run: per tick up to per_tick frames a stream, sequential rounds."""

    def __init__(self, dc, per_tick=4):
        self.dc, self.lock, self.per_tick = dc, threading.Lock(), per_tick
        self.cv = threading.Condition()
        self.streams = []
        threading.Thread(target=self._run, daemon=True).start()

    def open(self):
        st = OldStream(self)
        with self.cv:
            self.streams.append(st)
            self.cv.notify()
        return st

    def _ready(self):
        return any(st.frames or st.aborted or (st.ended and not st.frames) for st in self.streams)

    def _run(self):
        while True:
            with self.cv:
                while not self._ready():
                    self.cv.wait()
                work = []
                for st in list(self.streams):
                    if st.aborted or (st.ended and not st.frames):
                        self.streams.remove(st)
                        if st.slot is not None:
                            self.dc.close(st.slot)
                            st.slot = None
                        st.out.put(None)
                        continue
                    if not st.frames:
                        continue
                    if st.slot is None:
                        if not self.dc.free:
                            continue
                        st.slot = self.dc.open()
                    take, st.frames = st.frames[: self.per_tick], st.frames[self.per_tick:]
                    work.append((st, take))
            if not work:
                time.sleep(0.005)
                continue
            try:
                pcm = {id(st): [] for st, _ in work}
                for k in range(max(len(t) for _, t in work)):
                    batch = {st.slot: t[k] for st, t in work if k < len(t)}
                    with self.lock:
                        res = self.dc.decode(batch)
                    for st, t in work:
                        if k < len(t):
                            pcm[id(st)].append(res[st.slot])
                for st, t in work:
                    st.out.put(to_pcm16(np.concatenate(pcm[id(st)])))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                for st, _ in work:
                    st.out.put(e)


def drive(svc, plan):
    """plan: list of (time offset s, stream id, action, codes) applied in order; returns bytes per id."""
    sts, got, ths = {}, {}, []

    def reader(i, st):
        got[i] = b"".join(st.chunks())

    t0 = time.time()
    for t, i, act, codes in plan:
        d = t0 + t - time.time()
        if d > 0:
            time.sleep(d)
        if act == "open":
            sts[i] = svc.open()
            th = threading.Thread(target=reader, args=(i, sts[i]))
            th.start()
            ths.append(th)
        elif act == "push":
            sts[i].push(codes)
        else:
            sts[i].finish()
    for th in ths:
        th.join(600)
    return got


def make_plan(rng, n=36, max_frames=14):
    """Streams arrive over ~1.2 s; each pushes its frames in uneven bursts (1-5 frames, as LM
    replicas do), so some streams back up while others start fresh."""
    ev = []
    lens = {}
    for i in range(n):
        t = float(rng.uniform(0, 1.2))
        nf = int(rng.integers(2, max_frames + 1))
        lens[i] = nf
        ev.append((t, i, "open", None))
        codes = rng.integers(0, 2048, size=(nf, 16))
        k = 0
        while k < nf:
            t += float(rng.uniform(0.0, 0.12))
            m = int(rng.integers(1, 6))
            for c in codes[k:k + m]:
                ev.append((t, i, "push", c))
            k += m
        ev.append((t, i, "finish", None))
    ev.sort(key=lambda e: (e[0], e[1], {"open": 0, "push": 1, "finish": 2}[e[2]]))
    return ev, lens


torch.set_num_threads(6)
cw = CodecWeights(W)
rng = np.random.default_rng(7)
plan, lens = make_plan(rng)
# 4 groups of 8 slots, one bucket of 2 groups: more pending groups than one call holds, so the
# new scheduler picks groups by deadline and the old one makes two calls a round. One bucket
# shape: a slot's numerics then cannot depend on which bucket served it, so the PCM must match
# bit for bit. (With buckets (1, 2) a few streams differ at ~50 dB: the CPU BF16 matmul of the
# 8-row and 16-row calls round differently; the old scheduler has the same property.)
GROUPS = tuple(int(g) for g in (sys.argv[2] if len(sys.argv) > 2 else "2").split(","))
dc_old = DeviceCodec(cw, "cpu", groups=GROUPS, n_groups=4)
dc_new = DeviceCodec(cw, "cpu", groups=GROUPS, n_groups=4)
t = time.time()
old = drive(OldTickService(dc_old), plan)
print(f"old tick scheduler: {len(old)} streams in {time.time() - t:.1f}s")
new_svc = CodecService(dc_new, log=print)
t = time.time()
new = drive(new_svc, plan)
print(f"deadline scheduler: {len(new)} streams in {time.time() - t:.1f}s, {new_svc.ticks} calls, {new_svc.frames} frames, "
      f"{new_svc.frames / max(new_svc.ticks, 1):.1f} active/call")
assert sorted(old) == sorted(new) == list(range(len(lens)))
exact = 0
worst = float("inf")
for i in sorted(lens):
    a, b = np.frombuffer(old[i], "<i2"), np.frombuffer(new[i], "<i2")
    assert a.size == b.size == 1920 * lens[i], (i, a.size, b.size, lens[i])
    exact += old[i] == new[i]
    if old[i] != new[i]:
        worst = min(worst, snr_db(a, b))
print(f"per-stream PCM: {exact}/{len(lens)} bit-identical to the old scheduler" + (f", worst SNR of the rest {worst:.1f} dB" if exact < len(lens) else ""))
if len(GROUPS) == 1:
    assert exact == len(lens), "PCM differs from the old tick scheduler"
else:
    assert worst >= 40, worst
assert not dc_new.pos, f"slots leaked: {dc_new.pos}"
v = scheduler_selftest(new_svc, n=20, frames=4)
print(f"OK (self-test worst SNR {v})")
