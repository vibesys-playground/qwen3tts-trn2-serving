"""Streaming TTS server for Qwen3-TTS-12Hz-1.7B-CustomVoice on one Trainium2 chip.

    python -m engine.server --host 127.0.0.1 --port 8000

GET /health, POST /generate (raw s16le 24 kHz PCM streamed as frames are generated, or the
streamed decode of given `codes`), POST /score (teacher-forced log-probabilities).

The language model runs as continuous batches (engine.batch): one replica per visible NeuronCore
(engine.replica), each advancing all of its live streams by one frame per device step, admitting
new requests at step boundaries. With one visible core the replica is a thread of this process;
with several, the replicas are processes of their own, one per core but the last, and this
process (HTTP, routing to the replica with the fewest requests) opens only the codec's core. Each
stream's frames go to its own stream (sink) of the batched device codec (engine.codec_dev), which
owns the last core when two or more are visible and shares the core with the replica otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import tempfile
import threading
import time

from fastapi import FastAPI, Request  # module level: the route annotations must resolve
from fastapi.responses import JSONResponse, StreamingResponse

WEIGHTS = os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
CODEC_SELFTEST_DB = 20.0  # device codec vs host FP32 decode, per stream, at startup (uniform random
# codes: BF16 on the CPU gives ~30 dB there and 36-44 dB on real codes; the task gate is 15 dB)
STALL_S = float(os.environ.get("QWEN3TTS_STALL_S", "2400"))  # startup stage without progress -> exit (sized for a cold compile of every graph)
# First-audio hold: a uniform playout buffer. A text /generate releases its first PCM at max(first PCM ready,
# submit + HOLD); the codec anchors its frame deadlines at that release. 0 = no hold (control).
# Not applied to the codes path or /score.
HOLD_S = float(os.environ.get("QWEN3TTS_HOLD_MS", "150")) / 1000
# Async PCM delivery: stream PCM from async generators on the event loop, fed by one loop callback per codec
# call (CodecStream.bind_async / achunks), instead of sync generators that hop to a worker thread
# per chunk. 0 = the sync path (control).
ASYNC_OUT = os.environ.get("QWEN3TTS_ASYNC_OUT", "1") == "1"


def _loop_or_none():
    """The running event loop when called on one (an HTTP route), else None."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FairLock:
    """A FIFO lock: the language model and the codec thread take turns on a shared NeuronCore."""

    def __init__(self):
        self.cv = threading.Condition()
        self.next = self.serving = 0

    def __enter__(self):
        with self.cv:
            t = self.next
            self.next += 1
            while self.serving != t:
                self.cv.wait()

    def __exit__(self, *a):
        with self.cv:
            self.serving += 1
            self.cv.notify_all()


class Watchdog:
    """Exit non-zero when a startup stage makes no progress for STALL_S seconds."""

    def __init__(self, limit: float):
        self.limit, self.stage, self.t = limit, "start", time.time()
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()

    def beat(self, stage: str) -> None:
        self.stage, self.t = stage, time.time()
        log(f"[startup] {stage}")

    def _run(self) -> None:
        while self.active:
            time.sleep(5)
            if self.active and time.time() - self.t > self.limit:
                log(f"[startup] STALLED for {self.limit:.0f}s in stage: {self.stage}; exiting")
                os._exit(3)


def setup_env() -> None:
    """Compile caches under $VIBESYS_CACHE_DIR (when set, so a restart loads compiled graphs instead
    of compiling them); the compiler's logs in a fresh temp directory."""
    cache = os.environ.get("VIBESYS_CACHE_DIR")
    if cache:
        os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", os.path.join(cache, "neff"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(cache, "inductor"))
        os.environ.setdefault("TORCH_NEURONX_HLO_COMPILE_CACHE", "1")
    os.environ.setdefault("TORCH_NEURONX_DEBUG_DIR", tempfile.mkdtemp(prefix="neuron_backend_"))
    os.chdir(tempfile.mkdtemp(prefix="qwen3tts_cwd_"))


def pick_device() -> str:
    """The first visible NeuronCore; the CPU only when QWEN3TTS_DEVICE=cpu says so (tests)."""
    dev = os.environ.get("QWEN3TTS_DEVICE")
    if dev:
        return dev
    import torch
    import torch_neuronx  # noqa: F401

    if not (torch.neuron.is_available() and torch.neuron.device_count() > 0):
        raise RuntimeError("no Neuron device is visible")
    return "neuron:0"


def codec_selftest(dc, cw, seed: int = 0, frames: int | None = None) -> float:
    """Decode a stream on every slot of the codec (8 distinct code sequences, SELFTEST_CODES then
    random codes) and compare each with the host FP32 decoder. On the device the streams run
    `frames` > 72 frames, so every slot's key/value ring wraps; one stream skips every third call
    (its state must stay untouched while its group decodes), and one stream ends early and its
    slot is reopened by a fresh stream mid-run (stale ring columns and contexts must not leak).
    Then the smallest bucket on its own. Returns the worst SNR in dB; exits non-zero below the
    limit."""
    import numpy as np

    from .codec import StreamDecoder
    from .codec_dev import snr_db
    from .replica import SELFTEST_CODES

    if frames is None:  # the eager CPU path (tests) keeps it short
        frames = int(os.environ.get("QWEN3TTS_CODEC_SELFTEST_FRAMES", "6" if dc.device.type == "cpu" else "80"))
    rng = np.random.default_rng(seed)
    seqs = [np.concatenate([np.array(SELFTEST_CODES), rng.integers(0, 2048, size=(frames - 2, 16))]) for _ in range(8)]
    refs = []
    for q in seqs:
        d = StreamDecoder(cw)
        refs.append(np.concatenate([d.push(q[i : i + 1]) for i in range(frames)]))
    worst = float("inf")

    def check(st, nslots):
        nonlocal worst
        g = np.concatenate(st["got"])
        ref = refs[st["seq"]][: len(g)]
        v = snr_db(ref, g)
        worst = min(worst, v)
        if len(g) != st["n"] * 1920 or not v >= CODEC_SELFTEST_DB:
            log(f"[selftest] codec FAILED on slot {st['slot']} ({nslots} streams, {st['kind']}, {len(g) // 1920}/{st['n']} frames): SNR {v:.1f} dB < {CODEC_SELFTEST_DB}\n device: {g[:16]} ...\n host: {ref[:16]} ...")
            os._exit(2)

    for nslots in (dc.max_batch, 8):
        n = frames if nslots > 8 else min(frames, 6)
        short = n // 3 if nslots > 8 else n
        streams = [dict(slot=dc.open(), seq=k % 8, n=n, i=0, got=[], kind="stream") for k in range(nslots)]
        if nslots > 8:
            streams[1]["kind"] = "skips every third call"
            streams[2].update(n=short, kind="ends early")
        call, live = 0, list(streams)
        while live:
            batch = {st["slot"]: seqs[st["seq"]][st["i"]] for st in live if not (st["kind"].startswith("skips") and call % 3 == 0)}
            res = dc.decode(batch)
            call += 1
            for st in list(live):
                if st["slot"] in batch:
                    st["got"].append(res[st["slot"]])
                    st["i"] += 1
                if st["i"] == st["n"]:
                    live.remove(st)
                    check(st, nslots)
                    dc.close(st["slot"])
                    if st["kind"] == "ends early":  # a fresh stream takes the freed slot mid-run
                        nst = dict(slot=dc.open(), seq=3, n=n - short, i=0, got=[], kind=f"fresh in reused slot {st['slot']} after {short} frames")
                        live.append(nst)
    return worst


def visible_cores() -> list[str]:
    """The NeuronCores of NEURON_RT_VISIBLE_CORES ("0,1,2,3", "0-3" or "2"), without opening the
    runtime; [] when it is unset."""
    v = os.environ.get("NEURON_RT_VISIBLE_CORES", "").strip()
    out = []
    for part in filter(None, (x.strip() for x in v.split(","))):
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return out


class _Job:
    """A request in flight. A /generate job has the codec stream (`sink`) its frames go to; a
    /score job gets its result on `out`."""

    __slots__ = ("jid", "out", "sink", "replica", "t", "done", "est", "chars", "frames", "completed")

    def __init__(self, jid, replica, sink=None):
        self.jid, self.replica, self.t, self.done, self.sink = jid, replica, time.time(), False, sink
        self.est = self.chars = self.frames = 0  # admission estimate (token-bucket gate only), text length, frames pushed
        self.completed = False
        self.out: queue.Queue = queue.Queue()


class _Replica:
    """The server's handle on one LM replica: its inbox and the requests it holds."""

    def __init__(self, rid, core):
        self.rid, self.core = rid, core
        self.inbox = None
        self.proc = None
        self.outstanding = 0
        self.ready = threading.Event()


class _Direct:
    """The `out` of an in-process replica: messages go straight to the server's dispatcher."""

    def __init__(self, fn):
        self.put = fn


class Engine:
    """The front end: routes requests to the LM replicas (one per NeuronCore, each a continuous
    batch) and hands each stream's frames to its own stream of the device codec (one sink per
    stream; engine.codec_dev).

    Two or more visible cores: the codec owns the last one in this process and the LM replicas
    are processes on the others. One core: the replica is a thread of this process and it and
    the codec take turns on the core (FairLock around each batch step and each codec call)."""

    def __init__(self, device: str | None, wd: Watchdog):
        cores = visible_cores()
        procs = os.environ.get("QWEN3TTS_LM_PROCS", "1") == "1"
        self.multi = device is None and procs and len(cores) >= 2
        if self.multi:
            # This process opens only the codec's core; the replicas get theirs.
            lm_cores, codec_core = cores[:-1], cores[-1]
            os.environ["NEURON_RT_VISIBLE_CORES"] = codec_core
        import torch

        from .codec import CodecWeights

        self.torch = torch
        torch.set_num_threads(int(os.environ.get("QWEN3TTS_CPU_THREADS", "4")))
        self.wd = wd
        self.jobs: dict = {}
        self.lock = threading.Lock()
        self.next_id = 0
        self.rr = 0
        self.fatal = None
        self.codecs = None
        # The admission gate in front of the codec (engine.admit; QWEN3TTS_ADMIT=0 turns it off).
        self.gate = None
        # QWEN3TTS_ADMIT_MODE=deadline (the deadline-feasibility gate, default) or bucket (the
        # older codec frame token bucket, kept as an alternative).
        self.admit_mode = os.environ.get("QWEN3TTS_ADMIT_MODE", "deadline")
        if os.environ.get("QWEN3TTS_ADMIT", "1") == "1":
            from .admit import AdmissionGate, DeadlineGate

            self.gate = DeadlineGate() if self.admit_mode == "deadline" else AdmissionGate()
        wd.beat("loading the codec decoder")
        self.codec = CodecWeights(WEIGHTS)
        if self.multi:
            self.replicas = [_Replica(i, c) for i, c in enumerate(lm_cores)]
            cdev = os.environ.get("QWEN3TTS_CODEC_DEVICE", "neuron:0")
            log(f"[startup] LM replicas on NeuronCores {lm_cores}, codec on NeuronCore {codec_core} ({cdev} in this process)")
            self._start_procs(lambda: self._build_codec(cdev, None))
        else:
            if device is None:
                device = pick_device()
            cdev = os.environ.get("QWEN3TTS_CODEC_DEVICE", device)
            dev_lock = FairLock() if cdev == device else None
            r = _Replica(0, device)
            self.replicas = [r]
            r.inbox = queue.Queue()
            from .replica import replica_main

            threading.Thread(target=replica_main, args=(0, device, r.inbox, _Direct(self._on_msg)),
                             kwargs={"threads": None, "lock": dev_lock}, daemon=True).start()
            self._wait_ready([r])
            self._build_codec(cdev, dev_lock)
        log(f"[startup] {len(self.replicas)} LM replica(s) ready: {[(r.rid, r.core) for r in self.replicas]}")
        if self.gate is not None:
            log(f"[admit] gate on: {self._gate_stats()}")
            threading.Thread(target=self._admit_log, daemon=True).start()
        else:
            log("[admit] gate off (QWEN3TTS_ADMIT=0)")

    def _build_codec(self, cdev: str, lock) -> None:
        from .codec_dev import CodecService, DeviceCodec, scheduler_selftest

        wd = self.wd
        wd.beat(f"compiling the codec decoder on {cdev}")
        t0 = time.time()
        # Codec call buckets in groups of 8 streams: 16, 32 and 64 streams per call ("2,4" caps
        # calls at 32 streams).
        groups = tuple(int(g) for g in os.environ.get("QWEN3TTS_CODEC_GROUPS", "2,4,8").split(","))
        # concurrent streams with codec state: every LM row can be live at once
        slots = int(os.environ.get("QWEN3TTS_CODEC_SLOTS", str(max(128, 64 * len(self.replicas)))))
        dc = DeviceCodec(self.codec, cdev, groups=groups, n_groups=-(-slots // 8), log=log)
        wd.beat("NKI self-tests (xf ring attention, DAC tail) against FP32 CPU references")
        dc.nki_selftest(self.codec)
        for nb in dc.buckets:  # each bucket's compile is its own watchdog stage (B=32 takes ~115 s cold)
            wd.beat(f"compiling codec bucket of {nb * 8} streams")
            with self.torch.inference_mode():  # as decode() runs it: no dynamo recompile on the first requests
                dc._call(list(range(nb)), {})
            log(f"[startup] codec bucket of {nb * 8} streams ready ({time.time() - t0:.1f}s)")
        log(f"[startup] codec compile/load {time.time() - t0:.1f}s, {slots} stream slots")
        wd.beat("codec self-test against the host FP32 decoder")
        t0 = time.time()
        worst = codec_selftest(dc, self.codec)
        log(f"[selftest] device codec vs host FP32: worst SNR {worst:.1f} dB over {dc.max_batch} + 8 streams ({time.time() - t0:.1f}s)")
        self.codecs = CodecService(dc, lock=lock, log=log)
        if os.environ.get("QWEN3TTS_CODEC_SCHED_SELFTEST", "1") == "1":
            wd.beat("codec scheduler self-test (out-of-order streams vs plain-order decode)")
            scheduler_selftest(self.codecs, log=log)

    # ------------------------------------------------------------------ replicas
    def _start_procs(self, while_first_compiles=None) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.outq = ctx.Queue()
        threading.Thread(target=self._recv, daemon=True).start()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        threads = int(os.environ.get("QWEN3TTS_LM_THREADS", "2"))
        from .replica import child_main

        def start(r, refs):
            env = {"NEURON_RT_VISIBLE_CORES": r.core, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                r.inbox = ctx.Queue()
                r.proc = ctx.Process(target=child_main, args=(r.rid, r.core, r.inbox, self.outq, refs, threads), daemon=True)
                r.proc.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            log(f"[startup] replica {r.rid} started on NeuronCore {r.core} (pid {r.proc.pid})")

        # Replica 0 compiles every graph (or loads them from the cache); the others start after
        # it, in parallel, and load the compiled graphs.
        first = self.replicas[0]
        start(first, None)
        if while_first_compiles is not None:
            while_first_compiles()
        self._wait_ready([first])
        for r in self.replicas[1:]:
            start(r, self.refs)
        self._wait_ready(self.replicas[1:])
        threading.Thread(target=self._monitor, daemon=True).start()

    def _wait_ready(self, rs) -> None:
        while not all(r.ready.is_set() for r in rs):
            if self.fatal:
                log(f"[startup] replica failed: {self.fatal}; exiting")
                os._exit(2)
            for r in rs:
                if r.proc is not None and not r.proc.is_alive() and not r.ready.is_set():
                    log(f"[startup] replica {r.rid} exited with code {r.proc.exitcode}; exiting")
                    os._exit(2)
            time.sleep(0.2)

    def _monitor(self) -> None:
        while True:
            time.sleep(1)
            for r in self.replicas:
                if r.proc is not None and not r.proc.is_alive():
                    log(f"[server] replica {r.rid} exited with code {r.proc.exitcode}; exiting")
                    os._exit(3)

    def _recv(self) -> None:
        while True:
            self._on_msg(self.outq.get())

    def _on_msg(self, msg) -> None:
        kind, rid = msg[0], msg[1]
        if kind == "ev":
            for jid, k, payload in msg[2]:
                job = self.jobs.get(jid)
                if job is None:
                    continue
                if k == "frame":
                    job.frames += 1
                    job.sink.push(payload)
                    continue
                if k == "done":
                    job.completed = not (isinstance(payload, dict) and payload.get("cancelled"))
                    if job.sink is not None:
                        job.sink.finish()
                    else:
                        job.out.put(None)
                elif k == "score":
                    job.out.put(payload)
                elif k == "error":
                    if job.sink is not None:
                        job.sink.fail(RuntimeError(payload))
                        job.sink.abort()
                    else:
                        job.out.put(RuntimeError(payload))
                self._close(job)
        elif kind == "beat":
            self.wd.beat(f"replica {rid}: {msg[2]}")
        elif kind == "ready":
            if rid == 0:
                self.refs = msg[2]
            self.replicas[rid].ready.set()
        elif kind == "fatal":
            self.fatal = f"replica {rid}: {msg[2]}"
            log(f"[server] FATAL {self.fatal}")
            if all(r.ready.is_set() for r in self.replicas):
                os._exit(3)

    def _close(self, job) -> None:
        with self.lock:
            if job.done:
                return
            job.done = True
            self.jobs.pop(job.jid, None)
            job.replica.outstanding -= 1
        if job.est and self.gate is not None:
            self.gate.finish(job.est, job.chars, job.frames, job.completed)

    def admit(self, text: str):
        """(admitted, est) for a text /generate, from the codec's live capacity and backlog."""
        if self.gate is None:
            return True, 0
        c = self.codecs
        if self.admit_mode == "deadline":
            return self.gate.decide(c), 0  # reads only codec state, never `text` (length-neutral)
        return self.gate.admit(len(text), c.fps_cap, c.backlog, len(c.streams))

    def _gate_stats(self) -> str:
        if self.admit_mode == "deadline":
            return self.gate.stats(self.codecs)
        return self.gate.stats(self.codecs.fps_cap)

    def _admit_log(self) -> None:
        last = None
        while True:
            time.sleep(float(os.environ.get("QWEN3TTS_ADMIT_LOG_S", "10")))
            g = self.gate
            cur = (g.admitted, g.rejected)
            if cur != last:
                last = cur
                log(f"[admit] {self._gate_stats()}, backlog {self.codecs.backlog}, streams {len(self.codecs.streams)}")

    def submit(self, msg_kind, *args, est: int = 0) -> _Job:
        sink = self.codecs.open() if msg_kind == "gen" else None
        if sink is not None and ASYNC_OUT:
            loop = _loop_or_none()
            if loop is not None:  # before the replica can push a frame
                sink.bind_async(loop)
        with self.lock:
            self.next_id += 1
            jid = self.next_id
            n = len(self.replicas)
            r = min(self.replicas, key=lambda x: (x.outstanding, (x.rid - self.rr) % n))
            self.rr = (r.rid + 1) % n
            r.outstanding += 1
            job = _Job(jid, r, sink)
            if sink is not None and HOLD_S > 0:
                sink.hold_until = job.t + HOLD_S  # first PCM out no earlier than this
            if est:
                job.est, job.chars = est, len(args[0])
            self.jobs[jid] = job
        r.inbox.put((msg_kind, jid) + args)
        return job

    def cancel(self, job) -> None:
        if not job.done:
            job.replica.inbox.put(("cancel", job.jid))

    # ------------------------------------------------------------------ requests
    def generate_stream(self, text: str, speaker: str, language: str, job=None):
        if job is None:
            job = self.submit("gen", text, speaker, language)
        sink = job.sink
        first = True
        try:
            for pcm in sink.chunks():
                if first:
                    first = False
                    now = time.time()
                    log(f"[generate] first chunk: {sink.first_frames} frames, held {sink.held_ms:.0f} ms, device codec {sink.codec_ms:.1f} ms (incl. lock wait), "
                        f"first frame push to pcm {1000 * (now - sink.t_push):.0f} ms, submit to first pcm {1000 * (now - job.t):.0f} ms "
                        f"(replica {job.replica.rid})")
                yield pcm
        finally:
            self.cancel(job)

    async def agenerate_stream(self, job):
        """generate_stream as an async generator on the event loop (job bound async)."""
        sink = job.sink
        first = True
        try:
            async for pcm in sink.achunks():
                if first:
                    first = False
                    now = time.time()
                    log(f"[generate] first chunk: {sink.first_frames} frames, held {sink.held_ms:.0f} ms, device codec {sink.codec_ms:.1f} ms (incl. lock wait), "
                        f"first frame push to pcm {1000 * (now - sink.t_push):.0f} ms, submit to first pcm {1000 * (now - job.t):.0f} ms "
                        f"(replica {job.replica.rid})")
                yield pcm
        finally:
            self.cancel(job)

    async def acodes_stream(self, codes):
        """codes_stream as an async generator on the event loop."""
        sink = self.codecs.open()
        sink.bind_async(asyncio.get_running_loop())
        for fr in codes:
            sink.push(fr)
        sink.finish()
        async for pcm in sink.achunks():
            yield pcm

    def codes_stream(self, codes):
        """The `codes` mode: the given frames through the same streaming decode path."""
        sink = self.codecs.open()
        for fr in codes:
            sink.push(fr)
        sink.finish()
        yield from sink.chunks()

    def score(self, text, speaker, language, codes):
        job = self.submit("score", text, speaker, language, codes)
        res = job.out.get()
        if isinstance(res, Exception):
            raise res
        return res


def make_app(engine: Engine):
    import contextlib

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # Every live stream holds a worker thread while its sync generator runs (codec decode):
        # the default 40 would hold back requests past the 40th before they reach the LM.
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = int(os.environ.get("QWEN3TTS_HTTP_THREADS", "1024"))
        yield

    app = FastAPI(lifespan=lifespan)
    media = "audio/L16; rate=24000; channels=1"

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/generate")
    async def generate(req: Request):
        b = await req.json()
        speaker, language = b.get("speaker", "Ryan"), b.get("language", "English")
        if b.get("codes") is not None:
            codes = [[int(c) for c in fr] for fr in b["codes"]]
            body = engine.acodes_stream(codes) if ASYNC_OUT else engine.codes_stream(codes)
            return StreamingResponse(body, media_type=media)
        ok, est = engine.admit(b["text"])  # before any LM or codec work
        if not ok:
            return JSONResponse({"error": "overloaded"}, status_code=503)
        job = engine.submit("gen", b["text"], speaker, language, est=est)  # joins the LM batch now, not when the body starts
        if job.sink.aq is not None:  # ASYNC_OUT: bound to this loop in submit
            return StreamingResponse(engine.agenerate_stream(job), media_type=media)
        return StreamingResponse(engine.generate_stream(b["text"], speaker, language, job=job), media_type=media)

    @app.post("/score")
    async def score(req: Request):
        b = await req.json()
        codes = [[int(c) for c in fr] for fr in b["codes"]]
        lp, eos = await asyncio.to_thread(engine.score, b["text"], b.get("speaker", "Ryan"), b.get("language", "English"), codes)
        return JSONResponse({"logprobs": lp, "eos_logprob": eos})

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    t0 = time.time()
    setup_env()
    wd = Watchdog(STALL_S)
    log(f"[startup] NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')}, cache {os.environ.get('VIBESYS_CACHE_DIR')}")
    engine = Engine(os.environ.get("QWEN3TTS_DEVICE"), wd)
    wd.active = False
    log(f"[startup] ready in {time.time() - t0:.1f}s")
    import gc

    import uvicorn

    app = make_app(engine)
    # The ~750k GC-tracked objects alive after startup (module graphs, compile-time objects,
    # self-test leftovers) are static. A gen2 collection walks all of them (0.3-0.35 s with the
    # GIL held), freezing every main-process thread; move them to the permanent generation so
    # later collections walk only what serving allocates.
    t1 = time.perf_counter()
    gc.collect()
    gc.freeze()
    log(f"[startup] gc.freeze: {gc.get_freeze_count()} objects frozen in {1000 * (time.perf_counter() - t1):.0f} ms")
    thr = os.environ.get("QWEN3TTS_GC_THRESHOLD")  # optional fallback, e.g. "50000,50,100"; off by default
    if thr:
        gc.set_threshold(*(int(x) for x in thr.split(",")))
        log(f"[startup] gc thresholds {gc.get_threshold()}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
