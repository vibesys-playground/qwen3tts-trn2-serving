"""One language-model replica: one NeuronCore, one continuous batch (engine.batch.BatchLM).

`replica_main` loads the checkpoint, compiles and self-tests the graphs, then serves messages
from its inbox until the process ends:

    ("gen", jid, text, speaker, language)            sampled generation
    ("score", jid, text, speaker, language, codes)   teacher-forced scoring
    ("cancel", jid)

and reports through `out.put(msg)`:

    ("beat", rid, stage)        startup progress (the server's watchdog)
    ("ready", rid, refs)        startup done (refs: the host self-test references, for reuse)
    ("ev", rid, events)         once per step: [(jid, 'frame' | 'done' | 'score' | 'error', payload)]
    ("fatal", rid, message)     the replica stopped

It runs as a thread of the server process (one visible NeuronCore, CPU tests) or as a process of
its own with NEURON_RT_VISIBLE_CORES set to its core (several cores: one replica per core, so
the host work and the GIL of one replica never delay another).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

SELFTEST_TEXTS = ["Self test of the engine.", "A second stream, at another position in the batch.",
                  "Numbers such as one, two and three are spoken as words.",
                  "This sentence is long enough that the stream keeps talking for several seconds while the batch changes its size around it."]
SELFTEST_CODES = [[1995, 1059, 1213, 1660, 513, 1452, 1876, 1155, 228, 1517, 1617, 1210, 1450, 1015, 1585, 1736],
                  [215, 1059, 52, 1413, 1513, 852, 1087, 1155, 1903, 1517, 1617, 1004, 1450, 1015, 1585, 1736]]
SELFTEST_PATTERNS = ["AB" * 8, "BA" * 8, "AABB" * 4, "ABBBAABA" * 6]  # the last: 48 frames (lives across a shrink)
TOL_MEAN, TOL_MAX, TOL_EOS = 0.1, 1.5, 0.5  # device vs host BF16 (the serial engine's self-test limits)


def selftest_variants():
    A, B = SELFTEST_CODES
    return [(SELFTEST_TEXTS[k], [A if ch == "A" else B for ch in SELFTEST_PATTERNS[k]]) for k in range(4)]


def host_refs(lm):
    """Teacher-forced host BF16 scores of the self-test variants (the serial engine on the CPU)."""
    return [lm.score(text, "Ryan", "English", codes) for text, codes in selftest_variants()]


def lp_diff(a, b):
    import numpy as np

    d = np.abs(np.asarray(a[0]) - np.asarray(b[0]))
    return float(d.mean()), float(d.max()), abs(a[1] - b[1])


def warm(blm, beat, log):
    """Compile (or load) every graph the replica serves with, one watchdog stage each."""
    import torch

    from .batch import KP
    from .model import BF

    lm = blm.lm
    d = blm.device
    with torch.inference_mode():
        x = torch.zeros((KP, 10, blm.H), dtype=BF, device=d)
        chained = set()
        for B in sorted(blm.buckets, key=lambda b: -blm.chain_pad.get(b, b)):
            K, V = blm.caches[B]
            st = blm._small_state(B)
            beat(f"compiling the prefill graph into the B={B} batch ({KP} arrivals per call)")
            t0 = time.time()
            st = list(blm.prefill(x, torch.zeros(KP, dtype=torch.long, device=d), K, V, *st))
            st[0].cpu()
            log(f"[startup] prefill B={B}: {time.time() - t0:.1f}s")
            z = torch.zeros(B, dtype=torch.long, device=d)
            Bc = blm.chain_pad.get(B, B)  # a bucket may run its chain at a larger (padded) size
            if blm.spec and Bc not in chained:
                chained.add(Bc)
                beat(f"compiling the speculative predictor chain B={Bc} (16 positions, proposal sampler)")
                t0 = time.time()
                cd, sd = blm.chain(torch.zeros((Bc, blm.H), dtype=BF, device=d), torch.zeros(Bc, dtype=torch.long, device=d),
                                   torch.zeros((Bc, (blm.G - 1) * blm.V1), device=d), torch.full((Bc, blm.G - 1), -1, dtype=torch.long, device=d))
                sd.cpu()
                log(f"[startup] chain B={Bc}: {time.time() - t0:.1f}s")
            elif not blm.spec:
                beat(f"compiling the batched frame graphs B={B} (predictor first/next positions)")
                t0 = time.time()
                lg, Kp, Vp = blm.predfirst(st[0], z)
                lg, Kp, Vp = blm.prednext(z, blm.gpos[(B, 2)], blm.ghead[2], Kp, Vp)
                lg.cpu()
            if not (blm.pages and B in blm.page_buckets):
                beat(f"compiling the talker step B={B}")
                t0 = time.time()
                hn, l0 = blm.talkerin(torch.zeros((B, blm.G), dtype=torch.long, device=d), torch.zeros((B, blm.H), device=d), z, K, V)
                l0.cpu()
                log(f"[startup] talker step B={B}: {time.time() - t0:.1f}s")
            else:
                from .batch import S_BATCH
                from .model import page_inputs

                beat(f"compiling the paged talker step B={B} (TalkerInP: long streams' KV pages in batch rows, {blm.page_p} per stream)")
                t0 = time.time()
                ins = tuple(t.to(d) for t in page_inputs([10] * B, {0: [B - 1]}, P=blm.page_p, page_at=blm.page_at, S=S_BATCH, ps=blm.page_ps))
                hn, l0 = blm.talkerp(torch.zeros((B, blm.G), dtype=torch.long, device=d), torch.zeros((B, blm.H), device=d), *ins, K, V)
                l0.cpu()
                log(f"[startup] paged talker step B={B}: {time.time() - t0:.1f}s")
        for a in blm.buckets:
            for b in blm.buckets:
                if a != b:
                    beat(f"compiling the row move B={a} -> B={b}")
                    t0 = time.time()
                    m = min(a, b)
                    blm.move(*blm.caches[b], *blm.caches[a], torch.zeros(m, dtype=torch.long, device=d)).cpu()
                    log(f"[startup] move {a}->{b}: {time.time() - t0:.1f}s")
        beat("compiling the B=1 frame graph (2048-slot cache, streams past 512 positions)")
        t0 = time.time()
        cache = lm.new_talker_cache()
        h = torch.zeros((1, blm.H), dtype=BF, device=d)
        lg = torch.zeros((1, blm.V0), dtype=torch.float32, device=d)
        st = lm.frame_state(10)
        hin = lm.host_input(lm.emb.prompt("Warm up.", "Ryan", "English")[1], 0, torch.Generator().manual_seed(0))
        lm.frame_step(h, lg, cache, st, hin)[0].cpu()
        log(f"[startup] frame B=1 S=2048: {time.time() - t0:.1f}s")
        blm.st = blm._small_state(blm.B)  # every row is free: the warm-up's writes are never read


def selftest(blm, refs, log) -> list[str]:
    """The batch against the host reference, through the serving scheduler itself.

    Waves of teacher-forced rows (4 variants with different texts and codes, admitted at
    different steps so every row sits at its own position) and sampled rows whose codes are
    recomputed on the host (half with host Gumbel noise: exact match; half with in-graph noise:
    inside the top-k set) fill the smallest bucket, grow into the largest (rows move between
    caches), re-admit into freed slots, shrink back with rows alive, and move two streams to the
    B = 1 path. Every score row must match the host BF16 reference; every sampled code must
    match; each bucket and the B = 1 path must have been exercised. Returns the failures."""
    from .batch import Job
    from .runtime import PREFILL_LEN

    variants = selftest_variants()
    sink_events = []
    old_sink = blm.sink
    blm.sink = sink_events.extend
    jobs = {}
    n = [0]

    def add(kind, k, **kw):
        n[0] += 1
        text, codes = variants[k]
        j = Job(("selftest", n[0]), kind, text, codes=codes if kind == "score" else None, **kw)
        jobs[j.jid] = (j, k)
        blm.add(j)
        return j

    t0 = time.time()
    if blm.spec:
        # Speculative chain: every step reruns the chain with the final codes forced and
        # checks each code against the host draw (blm.recheck), and every third step corrupts one
        # proposed code per sampled row (a random codebook, a random code), so fix-up passes run in
        # every bucket and must still end on the host's draws.
        import torch

        blm.recheck = True
        cg = torch.Generator().manual_seed(19)
        nstep = [0]

        def inject(B, verify):
            nstep[0] += 1
            if nstep[0] % 3:
                return []
            out = []
            for r in verify.nonzero()[:, 0].tolist():
                g = int(torch.randint(1, blm.G, (1,), generator=cg))
                out.append((r, g, int(torch.randint(0, blm.V1, (1,), generator=cg))))
            return out

        blm.inject = inject
        blm.relaunch_check = blm.overlap  # with the overlap, every talker relaunch is checked bitwise
    waves = blm.maxB // 8 + 10  # enough sampled rows for thousands of checked codes per bucket
    for w in range(waves):
        for q in range(4):
            add("score", (w + q) % 3)
        for q in range(4):
            add("gen", (w + q) % 3, max_frames=24, trace=True, hostnoise=(q % 2 == 0))
        blm.step()
    while blm.pending:
        blm.step()
    survivors = [add("score", 3), add("gen", 3, max_frames=80, trace=True, hostnoise=True),
                 add("gen", 3, max_frames=80, trace=True, hostnoise=False)]
    blm.run_until_idle(2000)
    add("score", 0, migrate_at=5)
    add("gen", 2, max_frames=10, trace=True, hostnoise=True, migrate_at=4)
    blm.run_until_idle(2000)
    if blm.pages and blm.page_buckets:
        # Paged rows at a small split: streams take their first page at position 14 and pages hold
        # 12 positions, so the 16-frame variants (positions 10..25) end on one page, the 24-frame
        # sampled rows (to 33) take a second, fresh page, and the 48-frame variant (to 57) needs
        # more than page_p (3) pages at 50 and moves to the B = 1 path with its pages. Rows start
        # in the smallest page bucket and more arrive while pages are live, so the batch grows
        # with pages (rows and pages move) and shrinks back. Every page starts fresh (no valid
        # slot) at its first step.
        old_split = (blm.page_at, blm.page_ps)
        blm.page_at, blm.page_ps = PREFILL_LEN + 4, 12
        base = dict(blm.stats)
        n1 = max(2, blm.page_buckets[0] // 2)  # streams + their pages fill the smallest page bucket
        first = [("score", 3, {}), ("gen", 3, dict(max_frames=24, trace=True, hostnoise=True)),
                 ("gen", 1, dict(max_frames=24, trace=True, hostnoise=False))] + [("score", q % 3, {}) for q in range(n1)]
        for kind, k, kw in first[:n1]:
            add(kind, k, **kw)
        for _ in range(6):
            blm.step()
        for q in range(blm.maxB // 2):
            add("score", q % 3)
        blm.run_until_idle(4000)
        blm.page_at, blm.page_ps = old_split
        pst = {k: v - base.get(k, 0) for k, v in blm.stats.items() if v - base.get(k, 0)}
        log(f"[selftest] paged rows: {pst}")
        if not pst.get("pages taken"):
            fails_pg = ["no stream took a KV page"]
        else:
            fails_pg = []
            if not any(k.startswith("paged steps B=") and k.endswith(f"={blm.maxB}") for k in pst):
                fails_pg.append(f"no paged step at B={blm.maxB}")
            if len(blm.page_buckets) > 1 and not pst.get("switches with pages"):
                fails_pg.append("the batch never changed bucket with pages live")
            if not pst.get("to B=1 path with pages"):
                fails_pg.append("no stream moved to the B=1 path with its pages")
    else:
        fails_pg = []
    fails_ov = overlap_selftest(blm, log) if blm.spec and blm.overlap else []
    blm.sink = old_sink
    blm.inject, blm.recheck, blm.relaunch_check = None, False, False
    fails = list(fails_pg) + fails_ov
    worst = [0.0, 0.0, 0.0]
    nscore = 0
    for jid, (j, k) in jobs.items():
        if j.kind != "score":
            continue
        if j.result is None:
            fails.append(f"score row {jid} gave no result")
            continue
        nscore += 1
        mean, mx, eos = lp_diff(j.result, refs[k])
        worst = [max(worst[0], mean), max(worst[1], mx), max(worst[2], eos)]
        if not (mean < TOL_MEAN and mx < TOL_MAX and eos < TOL_EOS):
            fails.append(f"score row {jid} (variant {k}, steps {dict(j.bucket_steps)}): mean {mean:.4f} max {mx:.4f} eos {eos:.4f}\n"
                         f"  device: {j.result}\n  host: {refs[k]}")
    errors = [e for e in sink_events if e[1] == "error"]
    if errors:
        fails.append(f"error events: {errors[:4]}")
    if blm.trace_bad:
        fails.append(f"sampler mismatches (where, job, frame, codebook, device code, host code, margin): {blm.trace_bad[:20]}")
    for b in list(blm.buckets) + [1]:
        if blm.trace_checked[f"B={b}"] == 0:
            fails.append(f"no sampled code was checked at B={b}")
    st = blm.stats
    if len(blm.buckets) > 1:
        lo, hi = blm.buckets[0], blm.buckets[-1]
        if not (st[f"switch {lo}->{hi}"] or any(k.startswith(f"switch {lo}->") for k in st)):
            fails.append("the batch never grew")
        if not any(k.startswith("switch") and k.endswith(f"->{lo}") for k in st if st[k]):
            fails.append("the batch never shrank")
    if st["to B=1 path"] < 2:
        fails.append("the B=1 path was not exercised")
    if blm.spec and blm.overlap and not blm.trace_checked["relaunch"]:
        fails.append("no talker relaunch was checked (overlap)")
    if blm.spec:
        if not st["spec fix-up passes"]:
            fails.append("no fix-up pass of the speculative chain ran")
        if not blm.trace_checked["recheck"]:
            fails.append("no code was rechecked with the chain forced")
    log(f"[selftest] batch: {nscore} score rows vs host BF16, worst mean {worst[0]:.4f} max {worst[1]:.4f} eos {worst[2]:.4f}; "
        f"sampled codes checked {dict(blm.trace_checked)}, mismatches {len(blm.trace_bad)}; survivors' steps by bucket "
        f"{[dict(j.bucket_steps) for j in survivors]}; stats {dict(st)}; {time.time() - t0:.1f}s")
    blm.stats.clear()
    return fails


def overlap_selftest(blm, log) -> list[str]:
    """Overlap self-test: in every bucket, one frame of a live batch computed by the control order and by the
    overlap with every sampled row's proposal corrupted (fix-ups and a talker relaunch), on the
    same fixed noise: codes, h' and logits0' must be bitwise equal. Half the bucket's rows are
    streams that take a KV page at a small split (TalkerInP with groups live)."""
    from . import batch as eb
    from .batch import Job
    from .runtime import PREFILL_LEN

    fails = []
    old_split, old_shrink = (blm.page_at, blm.page_ps), eb.SHRINK_AFTER
    paged = blm.pages and bool(blm.page_buckets)
    if paged:
        blm.page_at, blm.page_ps = PREFILL_LEN + 3, 12
    eb.SHRINK_AFTER = 1 << 30
    try:
        for B in blm.buckets:
            blm._switch(B)  # idle: selects the bucket
            n = max(1, B // 2) if paged and B in blm.page_buckets else B
            js = [Job(("ovtest", B, i), "gen", SELFTEST_TEXTS[i % 4], max_frames=40) for i in range(n)]
            for j in js:
                blm.add(j)
            for _ in range(5):
                blm.step()
            groups = sum(1 for r in blm.rows if r is not None and r.pages)
            diffs, nr = blm.overlap_ab(seed=33 + B)
            log(f"[selftest] overlap vs control at B={blm.B} ({sum(r is not None for r in blm.rows)} streams, {groups} with a KV page): "
                f"{'identical' if not diffs else diffs}, {nr} talker relaunch")
            if diffs:
                fails.append(f"overlap differs from the control at B={blm.B}: {diffs}")
            if not nr:
                fails.append(f"overlap self-test at B={blm.B}: the corrupted proposals caused no talker relaunch")
            for j in js:
                blm.cancel(j.jid)
            blm.run_until_idle(200)
    finally:
        blm.page_at, blm.page_ps = old_split
        eb.SHRINK_AFTER = old_shrink
    return fails


def timing(blm, log, steps=16):
    """ms per step with every row of each bucket live (sampled rows, in-graph noise)."""
    from .batch import Job

    sink = blm.sink
    blm.sink = lambda ev: None
    text = SELFTEST_TEXTS[3]
    res = {}
    for B in blm.buckets:
        blm._switch(B)  # idle: just selects the bucket
        js = [Job(("timing", B, i), "gen", text, max_frames=10 * steps) for i in range(B)]
        for j in js:
            blm.add(j)
        while blm.pending:
            blm.step()
        ts = []
        for _ in range(steps):
            t0 = time.perf_counter()
            blm.step()
            ts.append(1000 * (time.perf_counter() - t0))
        live, bnow = blm.live(), blm.B
        for j in js:
            blm.cancel(j.jid)
        blm.run_until_idle(200)
        ts.sort()
        res[B] = ts[len(ts) // 2]
        log(f"[timing] batch step B={bnow} with {live} live rows: median {ts[len(ts) // 2]:.1f} ms, min {ts[0]:.1f} ms "
            f"({1000 * live / ts[len(ts) // 2]:.0f} frames/s)")
    blm.sink = sink
    blm.stats.clear()
    return res


def replica_main(rid: int, device: str, inbox, out, refs=None, threads: int | None = None, log=None, lock=None) -> None:
    """Build, self-test and serve. Never returns normally (the server process owns its life).
    `lock`: held around each batch step when the core is shared with the codec (one core)."""
    if log is None:
        def log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] [lm{rid}] {msg}", flush=True)
    try:
        _replica(rid, device, inbox, out, refs, threads, log, lock)
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        out.put(("fatal", rid, f"{type(e).__name__}: {e}"))
        raise


def _replica(rid, device, inbox, out, refs, threads, log, lock):
    import contextlib

    import torch

    from .batch import BatchLM, Job
    from .runtime import LM

    weights = os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    if threads:
        torch.set_num_threads(threads)
    if device.startswith("neuron"):
        import torch_neuronx  # noqa: F401

    def beat(stage):
        log(f"[startup] {stage}")
        out.put(("beat", rid, stage))

    beat("loading the checkpoint")
    lm = LM(weights, log=log)
    compile_ = device.startswith("neuron")
    if refs is None:
        beat("self-test references on the host CPU (BF16, 4 teacher-forced utterances)")
        t0 = time.time()
        refs = host_refs(lm)
        log(f"[startup] host references in {time.time() - t0:.1f}s")
    beat(f"moving weights to {device}")
    lm.to_device(device, compile_=compile_)

    def sink(events):
        out.put(("ev", rid, events))

    blm = BatchLM(lm, sink, compile_, log=log)
    if compile_:
        warm(blm, beat, log)
    beat("batch self-test against the host reference (scores, sampler, bucket moves, B=1 path)")
    fails = selftest(blm, refs, log)
    if fails:
        for f in fails:
            log(f"[selftest] FAILED: {f}")
        out.put(("fatal", rid, "self-test failed"))
        sys.exit(2)
    if os.environ.get("QWEN3TTS_TIMING", "1") == "1":
        beat("timing the batch at each bucket")
        timing(blm, log)
    out.put(("ready", rid, refs))
    log("[startup] ready")
    stop = False

    def handle(msg):
        nonlocal stop
        kind = msg[0]
        if kind == "gen":
            blm.add(Job(msg[1], "gen", msg[2], msg[3], msg[4]))
        elif kind == "score":
            blm.add(Job(msg[1], "score", msg[2], msg[3], msg[4], codes=msg[5]))
        elif kind == "cancel":
            blm.cancel(msg[1])
        elif kind == "stop":
            stop = True

    while not stop:
        if not blm.busy():
            handle(inbox.get())
        while True:
            try:
                handle(inbox.get_nowait())
            except queue.Empty:
                break
        with lock if lock is not None else contextlib.nullcontext():
            blm.step()


def child_main(rid, core, inbox, out, refs, threads):
    """Entry point of a replica process (multiprocessing spawn): NEURON_RT_VISIBLE_CORES is
    already set to this replica's core by the parent."""
    from . import server

    server.setup_env()
    ppid = os.getppid()

    def orphan_watch():  # the server went away: exit too
        while True:
            time.sleep(2)
            if os.getppid() != ppid:
                os._exit(0)

    threading.Thread(target=orphan_watch, daemon=True).start()
    replica_main(rid, os.environ.get("QWEN3TTS_REPLICA_DEVICE", "neuron:0"), inbox, out, refs, threads)  # cpu: tests
