"""Codec host-contention benchmark (needs one NeuronCore; run inside the server image).

Question: is the codec call's per-slot wall-time slope under load (measured on trn2: 0.167 ms per
active slot, of which only 0.040 ms is codec-thread CPU) caused by per-chunk PCM delivery to the
HTTP streams (GIL contention), rather than by the device?

It builds the device codec and its CodecService exactly as engine.server does (NKI self-tests,
every bucket compiled, the codec self-test against the host FP32 decoder, the scheduler
self-test), then drives it in several modes, one after the other, in the same process:

  http  the real HTTP path: engine.server.make_app on an Engine whose LM replica is a feeder
        thread (it pushes random valid codes through Engine._on_msg, as the replicas' events
        arrive), served by uvicorn on 127.0.0.1; N client connections (http.client threads in a
        child process, so client CPU is off this process's GIL) each POST /generate, read the
        stream to its end and start the next. Engine.generate_stream, the first-audio hold and
        StreamingResponse are the server's own code.
  drain no HTTP: the feeder submits the jobs itself and the codec's delivery thread drops every
        item (counts bytes) instead of putting it on the stream's queue.

Mode spec (MODES, comma separated): http:<switch ms>[:sync|async|direct] or drain:<switch ms>. The
GIL switch interval is set per mode; sync/async sets engine.server.ASYNC_OUT; direct is async
with items handed to the event loop by the codec thread itself, without the delivery thread
(the path QWEN3TTS_ASYNC_DIRECT=1 selects in the server).

Load per mode: the feeder ticks every TICK_S (as frame events from 3 LM replicas arrive) and
pushes each live stream's frames at a per-stream rate of RATE/N frames/s, but never more than
PENDING frames waiting in a stream (the LM runs ahead of the codec by a bounded amount, as with
the admission gate); a stream ends after a random 40-120 frames. The offered rate steps through
PHASES (fraction of RATE, seconds each), so the call's fill varies and the wall-vs-slots slope
can be regressed; the last phase saturates the codec.

Per call (a wrapper around DeviceCodec.decode, as CodecService calls it): active slots, wall,
codec-thread CPU and the sub-phase split (DeviceCodec.ph). Per mode it prints one RESULT line
(JSON): frames/s in the saturated phase, mean slots/call, call wall and CPU there, the split, the
fit call_wall = a + b * active over every call after WARM_S, the fitted wall at 48 slots and the
mean wall of calls with 44-52 active slots.

Measured on trn2: with the sync delivery path (RATE 2400), http wall at 48 slots 23.4 ms, slope
0.188 ms/slot, the excess in `prep` (H2D, 7.3 ms wall vs 2.1 in drain); drain 19.0 ms, slope
0.072; the GIL switch interval (5 / 0.1 ms) changes little. At RATE 3200, A/B/A/B: async
2737/2764 frames/s, slope 0.062/0.100, wall at 48 slots 19.2/19.1 ms; sync 2240/2460, slope
0.231/0.138; direct (no delivery thread) 2415, 0.225. This is why async delivery is the default.

    python benchmarks/codec_gil_bench.py                (MODES=http:0.5,drain:0.5,http:5,http:0.1)
    python benchmarks/codec_gil_bench.py --client PORT N SECONDS   (the client child process)
CPU smoke: QWEN3TTS_DEVICE=cpu QWEN3TTS_CODEC_GROUPS=1 SLOTS=24 N=6 RATE=40 PHASES=1:4 ...
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N = int(os.environ.get("N", "120"))
RATE = float(os.environ.get("RATE", "2400"))
PENDING = int(os.environ.get("PENDING", "6"))
TICK_S = float(os.environ.get("TICK_S", "0.02"))
PHASES = [tuple(float(x) for x in p.split(":")) for p in os.environ.get("PHASES", "0.35:8,0.6:8,1.0:20").split(",")]
WARM_S = float(os.environ.get("WARM_S", "3"))
SLOTS = int(os.environ.get("SLOTS", "192"))
PORT = int(os.environ.get("PORT", "18901"))
MODES = os.environ.get("MODES", "http:0.5,drain:0.5,http:5,http:0.1").split(",")
STALL_S = float(os.environ.get("BENCH_STALL_S", "2400"))


# ---------------------------------------------------------------------------------- client child
def client_main(port: int, n: int, seconds: float) -> None:
    import http.client

    stop = time.time() + seconds
    tot = {"bytes": 0, "reqs": 0, "errors": 0}
    lock = threading.Lock()
    body = json.dumps({"text": "bench", "speaker": "Ryan", "language": "English", "stream": True}).encode()

    def run():
        while time.time() < stop:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
                c.request("POST", "/generate", body=body, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                nb = 0
                while True:
                    b = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not b:
                        break
                    nb += len(b)
                c.close()
                with lock:
                    tot["bytes"] += nb
                    tot["reqs"] += 1
            except Exception:  # noqa: BLE001
                with lock:
                    tot["errors"] += 1
                time.sleep(0.05)

    ts = [threading.Thread(target=run, daemon=True) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=max(0.0, stop - time.time()) + 30)
    print("CLIENT " + json.dumps(tot), flush=True)


# ---------------------------------------------------------------------------------- server side
def main() -> None:
    import numpy as np

    from engine import server
    from engine.codec_dev import SPLIT_PH, CodecService, DeviceCodec, scheduler_selftest

    log = server.log
    server.setup_env()  # compile caches; cwd in a fresh temp dir (neuronx-cc writes logs there)
    wd = server.Watchdog(STALL_S)
    import torch

    from engine.codec import CodecWeights

    torch.set_num_threads(int(os.environ.get("QWEN3TTS_CPU_THREADS", "4")))
    dev = os.environ.get("QWEN3TTS_DEVICE") or server.pick_device()
    wd.beat("loading the codec decoder")
    cw = CodecWeights(server.WEIGHTS)
    groups = tuple(int(g) for g in os.environ.get("QWEN3TTS_CODEC_GROUPS", "2,4,8").split(","))
    t0 = time.time()
    dc = DeviceCodec(cw, dev, groups=groups, n_groups=-(-SLOTS // 8), log=log)
    if dc.device.type != "cpu":
        wd.beat("NKI self-tests")
        dc.nki_selftest(cw)
    for nb in dc.buckets:
        wd.beat(f"compiling codec bucket of {nb * 8} streams")
        with torch.inference_mode():
            dc._call(list(range(nb)), {})
    log(f"[bench] codec compile/load {time.time() - t0:.1f}s")
    wd.beat("codec self-test against the host FP32 decoder")
    worst = server.codec_selftest(dc, cw)
    log(f"[selftest] device codec vs host FP32: worst SNR {worst:.1f} dB")

    class BenchService(CodecService):
        """CodecService whose delivery thread can drop items (mode drain)."""

        drop = False
        dropped = 0

        def _put(self, items):
            if self.drop:
                self.dropped += sum(len(it) for _, it in items if isinstance(it, bytes))
                return
            super()._put(items)

    svc = BenchService(dc, lock=None, log=log)
    wd.beat("codec scheduler self-test")
    scheduler_selftest(svc, log=log)
    wd.active = False

    # Per-call records: (t, active, wall, cpu, split[2 * len(SPLIT_PH)]).
    recs: list = []
    rec_on = [False]
    inner = dc.decode

    def decode(frames):
        k = len(SPLIT_PH)
        p0 = dc.ph.copy()
        t, c = time.perf_counter(), time.thread_time()
        out = inner(frames)
        w, cc = time.perf_counter() - t, time.thread_time() - c
        if rec_on[0]:
            recs.append((time.time(), len(frames), w, cc, dc.ph - p0))
        return out

    dc.decode = decode

    # The Engine without replicas: its LM is the feeder below.
    class Inbox:
        def put(self, msg):
            if msg[0] == "gen":
                feeder.add(msg[1])
            elif msg[0] == "cancel":
                feeder.cancel(msg[1])

    class R:
        rid, core, outstanding, inbox, proc = 0, "bench", 0, Inbox(), None

    eng = server.Engine.__new__(server.Engine)
    eng.jobs, eng.lock, eng.next_id, eng.rr, eng.replicas = {}, threading.Lock(), 0, 0, [R()]
    eng.codecs, eng.gate, eng.fatal, eng.wd = svc, None, None, wd
    rng = np.random.default_rng(0)
    pool = [list(map(int, rng.integers(0, 2048, size=16))) for _ in range(4096)]

    class Feeder:
        def __init__(self):
            self.cv = threading.Lock()
            self.live: dict = {}  # jid -> [frames left, credit]
            self.rate = 0.0
            self.self_submit = False
            self.run = True
            self.pushed = 0
            self.i = 0
            threading.Thread(target=self._loop, daemon=True, name="feeder").start()

        def add(self, jid):
            with self.cv:
                self.live[jid] = [int(rng.integers(40, 121)), 0.0]

        def cancel(self, jid):
            with self.cv:
                self.live.pop(jid, None)

        def _loop(self):
            last = time.perf_counter()
            while self.run:
                time.sleep(TICK_S)
                now = time.perf_counter()
                dt, last = now - last, now
                ev, ended = [], 0
                with self.cv:
                    per = self.rate / N * dt
                    for jid, e in list(self.live.items()):
                        job = eng.jobs.get(jid)
                        if job is None:
                            self.live.pop(jid, None)
                            continue
                        e[1] = min(e[1] + per, PENDING)
                        q = len(job.sink.frames)
                        while e[1] >= 1 and e[0] > 0 and q < PENDING:
                            q += 1
                            ev.append((jid, "frame", pool[self.i % len(pool)]))
                            self.i += 1
                            e[0] -= 1
                            e[1] -= 1
                        if e[0] == 0:
                            ev.append((jid, "done", {}))
                            self.live.pop(jid)
                            ended += 1
                self.pushed += sum(1 for x in ev if x[1] == "frame")
                if ev:
                    eng._on_msg(("ev", 0, ev))
                if self.self_submit:
                    while len(self.live) < N:
                        eng.submit("gen", "bench", "Ryan", "English")

        def stop_all(self):
            with self.cv:
                jids = list(self.live)
                self.live.clear()
            if jids:
                eng._on_msg(("ev", 0, [(j, "done", {}) for j in jids]))

    feeder = Feeder()

    import subprocess

    import uvicorn

    srv = None

    def start_http():
        nonlocal srv
        if srv is not None:
            return
        cfg = uvicorn.Config(server.make_app(eng), host="127.0.0.1", port=PORT, log_level="warning")
        srv = uvicorn.Server(cfg)
        threading.Thread(target=srv.run, daemon=True, name="uvicorn").start()
        t = time.time()
        while not srv.started:
            if time.time() - t > 60:
                log("[bench] uvicorn did not start; exiting")
                os._exit(2)
            time.sleep(0.05)

    def wait_idle(limit=120):
        t = time.time()
        while (svc.backlog or len(svc.streams)) and time.time() - t < limit:
            time.sleep(0.2)
        return svc.backlog == 0 and not svc.streams

    total_s = sum(d for _, d in PHASES)
    k = len(SPLIT_PH)
    results = []
    for spec in MODES:
        parts = spec.split(":")
        kind, sw = parts[0], float(parts[1]) / 1000
        flav = parts[2] if len(parts) > 2 else None
        if flav is not None and hasattr(server, "ASYNC_OUT"):
            server.ASYNC_OUT = flav in ("async", "direct")
            svc.direct = flav == "direct"  # async items handed over by the codec thread itself
        sys.setswitchinterval(sw)
        recs.clear()
        svc.drop = kind == "drain"
        child = None
        if kind == "http":
            start_http()
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--client", str(PORT), str(N), str(total_s + 2)],
                                     stdout=subprocess.PIPE, text=True)
        else:
            feeder.self_submit = True
        log(f"[bench] mode {spec}: switch interval {1000 * sys.getswitchinterval():.2f} ms, async {getattr(server, 'ASYNC_OUT', None)}")
        t_start = time.time()
        rec_on[0] = False
        sat = None
        f0 = svc.frames
        for frac, dur in PHASES:
            feeder.rate = frac * RATE
            tp = time.time()
            if tp - t_start >= WARM_S:
                rec_on[0] = True
            while time.time() - tp < dur:
                time.sleep(0.1)
                if not rec_on[0] and time.time() - t_start >= WARM_S:
                    rec_on[0] = True
            sat = (tp, time.time())
            log(f"[bench]   phase {frac:.2f} x {RATE:.0f} fps for {dur:.0f}s: codec {svc.frames - f0} frames total, streams {len(svc.streams)}, backlog {svc.backlog}")
        rec_on[0] = False
        feeder.rate = 0.0
        feeder.self_submit = False
        feeder.stop_all()
        cl = {}
        if child is not None:
            while child.poll() is None:  # a request posted after the stop gets no frames: end it at once
                time.sleep(0.2)
                feeder.stop_all()
            out, _ = child.communicate(timeout=300)
            for line in out.splitlines():
                if line.startswith("CLIENT "):
                    cl = json.loads(line[7:])
        idle = wait_idle()
        svc.drop = False
        # Results.
        R_ = [r for r in recs]
        a = np.array([r[1] for r in R_], dtype=float)
        w = np.array([r[2] for r in R_]) * 1000
        c = np.array([r[3] for r in R_]) * 1000
        sp = np.array([r[4] for r in R_]) * 1000 if R_ else np.zeros((0, 2 * k))
        ts = np.array([r[0] for r in R_])
        m = (ts >= sat[0] + 1.0) & (ts <= sat[1]) if R_ else np.zeros(0, bool)
        res = {"mode": spec, "switch_ms": 1000 * sw, "async": getattr(server, "ASYNC_OUT", None), "direct": svc.direct, "calls": int(len(R_)), "idle_after": bool(idle), "client": cl}
        if m.sum() > 5:
            dur = ts[m][-1] - ts[m][0]
            res.update(sat_fps=float(a[m][1:].sum() / dur) if dur > 0 else None, sat_calls_s=float((m.sum() - 1) / dur) if dur > 0 else None,
                       sat_active=float(a[m].mean()), sat_wall=float(w[m].mean()), sat_cpu=float(c[m].mean()),
                       sat_split={p: [round(float(sp[m][:, i].mean()), 3), round(float(sp[m][:, k + i].mean()), 3)] for i, p in enumerate(SPLIT_PH)})
        if len(R_) > 10 and np.ptp(a) > 0:
            b, a0 = np.polyfit(a, w, 1)
            bc, ac0 = np.polyfit(a, c, 1)
            near = (a >= 44) & (a <= 52)
            res.update(fit_wall=[round(float(a0), 3), round(float(b), 4)], fit_cpu=[round(float(ac0), 3), round(float(bc), 4)],
                       wall_at48=round(float(a0 + 48 * b), 2), wall_44_52=round(float(w[near].mean()), 2) if near.any() else None,
                       n_44_52=int(near.sum()), active_range=[int(a.min()), int(a.max())])
            # per-phase means (for reading the fit)
            res["by_active"] = {f"{lo}-{lo + 7}": [int(((a >= lo) & (a < lo + 8)).sum()), round(float(w[(a >= lo) & (a < lo + 8)].mean()), 2)]
                                for lo in range(0, 72, 8) if ((a >= lo) & (a < lo + 8)).any()}
        if kind == "drain":
            res["dropped_bytes"] = svc.dropped
        log("RESULT " + json.dumps(res))
        results.append(res)
        art = os.environ.get("VIBESYS_ARTIFACTS_DIR")
        if art:
            Path(art).mkdir(parents=True, exist_ok=True)
            np.save(Path(art) / f"calls_{len(results)}_{spec.replace(':', '_')}.npy", np.column_stack([ts, a, w, c, sp]) if R_ else np.zeros((0, 4)))
    log("SUMMARY")
    for r in results:
        log(f"  {r['mode']:>16}: sat {r.get('sat_fps', 0) or 0:.0f} fps, {r.get('sat_active', 0):.1f} slots/call, wall {r.get('sat_wall', 0):.2f} cpu {r.get('sat_cpu', 0):.2f} ms; "
            f"fit wall {r.get('fit_wall')} cpu {r.get('fit_cpu')}; wall@48 {r.get('wall_at48')} (44-52 mean {r.get('wall_44_52')}, n {r.get('n_44_52')}); client {r.get('client')}")
    if srv is not None:
        srv.should_exit = True
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        client_main(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:
        main()
