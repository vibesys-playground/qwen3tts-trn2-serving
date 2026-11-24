"""Continuous batching of the language model on one NeuronCore.

`BatchLM` keeps a slot table of live streams and advances all of them by one frame per step with
one launch of the batched frame graph (engine.model.BatchFrame): every request in flight shares
each device step. Requests join at step boundaries: their prompts go through the prefill graph
(engine.model.PrefillInto, KP arrivals per call) straight into free rows of the batch, and their
first frame comes out of the same step. A stream leaves at EOS, at the position cap, or when its
client goes away.

Batch sizes are buckets (default: 16, 32 and 64, see BUCKETS): each step runs the smallest bucket that holds the
live streams. A bucket has its own talker cache [L, B, kvh, 512, hd] (written in place, one slot
per row per step); changing bucket copies the live rows into the other cache (MoveRows). Growing
happens at once, shrinking after SHRINK_AFTER steps at the lower load.

The batched cache has 512 positions (a B = 64 cache of 2048 cannot be built: its DMA offsets
exceed 4 GiB). A stream that reaches position 511 keeps its row (the leader) and borrows a free
row of the same batch as its next KV page of 511 slots (engine.model.TalkerInP: the page
row attends with the leader's query and the leader merges both attentions exactly by FP32
log-sum-exp; no copy). When it needs more pages than the graph has partners (PAGE_P), or no row
is free in the largest bucket, it continues exactly on the B = 1 path with a 2048-slot cache
(runtime.LM's FrameStep graph): its rows and pages are copied there and it runs one B = 1 launch
per step until it ends. Nothing is truncated: the cap stays prompt ids + frames < 2048, as on
the serial path.

/score requests are teacher-forced rows of the same batch (same graph, same numerics).
"""

from __future__ import annotations

import collections
import os
import time

import torch

from .model import BF, MoveRows, PredChain, PredFirst, PredNext, PrefillInto, TalkerIn, TalkerInP, page_inputs
from .runtime import LM, MAX_POSITIONS, PREFILL_LEN, TALKER_SLOTS
from .sampling import NEG, TEMPERATURE, TOP_K, code0_scores, gumbel_check, gumbel_noise, gumbel_pick

S_BATCH = 512  # talker cache slots of the batched graphs
KP = 4  # arrivals per prefill call (padded with duplicates of a real arrival)
MAX_ADMIT = int(os.environ.get("QWEN3TTS_MAX_ADMIT", "8"))  # arrivals admitted per step boundary
SHRINK_AFTER = int(os.environ.get("QWEN3TTS_SHRINK_AFTER", "12"))  # steps at lower load before a smaller bucket
# The predictor's 15 codes come from one speculative chain graph (engine.model.PredChain) whose
# proposals the host verifies code by code (exact Gumbel-max on the chain's own logits and noise),
# with fix-up passes from the first wrong code. QWEN3TTS_SPEC=0 selects the non-speculative path
# (15 host round trips per frame: PredFirst + PredNext with a host draw after each).
SPEC = os.environ.get("QWEN3TTS_SPEC", "1") == "1"
# Buckets: 16, 32 and 64 with the speculative chain (its B=64 step fits the 80 ms frame period);
# 16 and 32 on the non-speculative path. Every code is drawn or verified on the host at every B, and the
# startup self-test checks the sampled codes of every bucket. The admission limit follows the
# largest bucket (self.maxB).
BUCKETS = tuple(sorted(int(b) for b in os.environ.get("QWEN3TTS_BUCKETS", "16,32,64" if SPEC else "16,32").split(",")))
# Launch the talker on the chain's proposed codes before the host verifies them (it is relaunched
# with the corrected codes when a fix-up pass changed any: same cache slot, overwritten).
SPEC_TALKER = os.environ.get("QWEN3TTS_SPEC_TALKER", "0") == "1"
# Overlap the host verify with the talker. The talker (TalkerIn/TalkerInP, page buckets included)
# is launched on the chain's proposed codes right after the chain's outputs reach the host (never
# before: the in-order device queue would hold their device-to-host copy behind it), and the
# host verify and fix-ups run while it computes; when the final codes differ from the proposals
# it is relaunched on the final codes (same graph, same cache slot: the slot at pos is masked on
# read and overwritten). The next step's host inputs (trail, positions, page inputs) are built
# before the wait on the talker's logits. QWEN3TTS_OVERLAP=0: the serial order (verify, then
# talker), kept as the control for the startup self-test.
OVERLAP = os.environ.get("QWEN3TTS_OVERLAP", "1") == "1"
# Run a bucket's chain on a larger compiled chain, the extra rows padding ("32:64": the B=32 batch's
# chain runs as B=64, the default with the chain; measured on trn2: the compiled B=32 chain took
# 25.4 ms, the B=64 one 20.7 ms; a B=32 step took 45.3 ms padded vs 47.9 ms on its own graph).
CHAIN_PAD = dict(tuple(int(x) for x in kv.split(":")) for kv in os.environ.get("QWEN3TTS_CHAIN_PAD", "32:64" if SPEC else "").split(",") if kv)
# Paged rows (a long stream borrows free rows of its batch as KV pages) instead of the B = 1 path.
# With PAGES the talker step of every bucket is TalkerInP (measured on trn2: its B=64 graph alone
# takes 19.4 ms with 0-2 page groups vs TalkerIn's 22.7 ms; 19.7 ms at P=3), so TalkerIn is not
# compiled; PAGE_P pages per stream (3: up to position 511 + 3 * 511 - 1 = 2043).
# QWEN3TTS_PAGES=0: TalkerIn, with every stream moving to the B = 1 path at position 512.
PAGES = os.environ.get("QWEN3TTS_PAGES", "1") == "1"
PAGE_P = int(os.environ.get("QWEN3TTS_PAGE_P", "3"))
PAGE_BUCKETS = os.environ.get("QWEN3TTS_PAGE_BUCKETS", "")  # default: every bucket


class Job:
    """A request inside a replica. kind 'gen' (sampled frames) or 'score' (teacher-forced codes).

    Test-only fields (the startup self-test and timing; never set by the server): max_frames,
    trace (check each sampled code on the host), hostnoise (use host Gumbel noise for this row),
    migrate_at (move to the B = 1 path at this frame)."""

    __slots__ = ("jid", "kind", "text", "speaker", "language", "codes", "max_frames", "trace", "hostnoise", "migrate_at",
                 "t", "n_ids", "cap", "trailing", "lps", "cancelled", "t_in", "t_admit", "t_first", "result", "bucket_steps", "pages")

    def __init__(self, jid, kind, text, speaker="Ryan", language="English", codes=None, max_frames=None, trace=False,
                 hostnoise=False, migrate_at=None):
        self.jid, self.kind, self.text, self.speaker, self.language = jid, kind, text, speaker, language
        self.codes = codes
        self.max_frames, self.trace, self.hostnoise, self.migrate_at = max_frames, trace, hostnoise, migrate_at
        self.t = 0
        self.n_ids = self.cap = 0
        self.trailing = None
        self.lps = []
        self.cancelled = False
        self.t_in = time.time()
        self.t_admit = self.t_first = None
        self.result = None
        self.bucket_steps = collections.Counter()
        self.pages = []  # batch rows borrowed as KV pages, in position order


class _Long:
    """A stream on the B = 1 path (2048-slot cache)."""

    __slots__ = ("job", "cache", "h", "logits", "state")

    def __init__(self, job, cache, h, logits, state):
        self.job, self.cache, self.h, self.logits, self.state = job, cache, h, logits, state


class _NoisePump:
    """Gumbel noise for the next step, drawn on a background thread while the current step runs
    (torch's kernels release the GIL). Its own generator; a draw for a batch size that is not
    asked for is thrown away (the noise stays i.i.d. and independent of every logit)."""

    def __init__(self, v0: int, n_sub: int, v1: int):
        from concurrent.futures import ThreadPoolExecutor

        self.v0, self.n_sub, self.v1 = v0, n_sub, v1
        self.gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))
        self.ex = ThreadPoolExecutor(1, thread_name_prefix="noise")
        self.fut, self.B = None, None

    def _draw(self, B):
        return gumbel_noise((B, self.v0), self.gen), gumbel_noise((B, self.n_sub, self.v1), self.gen)

    def get(self, B):
        """(noise of code 0 [B, V0], noise of codes 1-15 [B, 15, V1]), FP32 on the host."""
        r = self.fut.result() if self.fut is not None else None
        if r is None or self.B != B:
            r = self._draw(B)
        self.B, self.fut = B, self.ex.submit(self._draw, B)
        return r


class BatchLM:
    """The continuous-batching language model of one replica: slot table, buckets, admission,
    the speculative code-predictor chain with host verification, paged KV rows and the B = 1
    path for long streams. `step()` advances every live stream by one frame."""

    def __init__(self, lm: LM, sink, compile_: bool, buckets=BUCKETS, log=print, spec: bool | None = None):
        """`lm` is already on its device. `sink(events)` receives, once per step, a list of
        (jid, kind, payload): 'frame' [16] codes, 'done' info dict, 'score' (logprobs, eos),
        'error' message."""
        self.lm, self.sink, self.log = lm, sink, log
        self.device = lm.device
        self.buckets = tuple(sorted(buckets))
        self.maxB = self.buckets[-1]
        fs = lm.frame_mod
        self.pf_mod = PredFirst(fs).to(self.device).eval()
        self.pn_mod = PredNext(fs).to(self.device).eval()
        self.ti_mod = TalkerIn(fs).to(self.device).eval()
        self.prefill_mod = PrefillInto(lm.prefill_mod).to(self.device).eval()
        self.move_mod = MoveRows()
        self.chain_mod = PredChain(fs).to(self.device).eval()
        self.tp_mod = TalkerInP(fs, "gather").to(self.device).eval()
        mods = (self.pf_mod, self.pn_mod, self.ti_mod, self.prefill_mod, self.move_mod, self.chain_mod, self.tp_mod)
        if compile_:
            from torch import _dynamo

            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            mods = tuple(torch.compile(m, backend="neuron", fullgraph=True, dynamic=False) for m in mods)
        self.predfirst, self.prednext, self.talkerin, self.prefill, self.move, self.chain, self.talkerp = mods
        # Paging: the page buckets (TalkerInP compiled for each), pages per stream, where a
        # stream's first page starts (page_at) and the page size (ps); the self-test sets small ones.
        self.pages = PAGES
        self.page_p = PAGE_P
        pb = [int(b) for b in PAGE_BUCKETS.split(",") if b] or list(self.buckets)
        self.page_buckets = tuple(sorted(b for b in pb if b in self.buckets))
        self.page_at = S_BATCH - 1
        self.page_ps = S_BATCH - 1
        self._pg = None  # this step's TalkerInP inputs on the device (None: TalkerIn)
        self._pg_static = {}  # (B, groups) -> the step-invariant TalkerInP inputs on the device
        self.spec = SPEC if spec is None else spec
        self.spec_talker = SPEC_TALKER
        self.overlap = OVERLAP
        self.relaunch_check = False  # self-test: after a relaunch, launch again and require bitwise-equal h'/logits0'
        self._l0wait = 0.0
        self.chain_pad = dict(CHAIN_PAD)  # bucket -> the chain batch size it runs at (padded)
        self.chain_buckets = {self.chain_pad.get(b, b) for b in self.buckets}  # the chain sizes compiled (one per bucket, padding included)
        self.inject = None  # tests: inject(B, verify) -> [(row, codebook 1..15, code)] forced into the first pass (corrupted proposals)
        self.phase = collections.Counter()  # last step's phase times (ms) and spec counters
        self.recheck = False  # self-test: rerun the chain with every final code forced and check each against the host draw
        t = lm.talker_mod
        self.V0, self.V1 = fs.V0, fs.V1
        self.H = fs.H
        self.G = fs.n_sub + 1
        self.n_noise = fs.o_trail  # V0 + 15 * V1 (one frame's Gumbel noise, as the B = 1 FrameStep's layout)
        d = self.device
        self.caches = {}
        for B in self.buckets:
            shape = (t.L, B, t.kvh, S_BATCH, t.hd)
            self.caches[B] = (torch.zeros(shape, dtype=BF, device=d), torch.zeros(shape, dtype=BF, device=d))
        # predictor positions / head indices as device tensors, made once (no per-step copy)
        self.gpos = {(B, g): torch.full((B,), g, dtype=torch.long, device=d) for B in self.buckets for g in range(2, self.G)}
        self.ghead = {g: torch.tensor([g - 1], dtype=torch.long, device=d) for g in range(2, self.G)}
        self.B = self.buckets[0]
        self.st = self._small_state(self.B)
        # Repetition-penalty presence and frame counts of each row, on the host (code 0 is drawn there)
        self.pres = torch.zeros((self.B, self.V0), dtype=torch.float32)
        self.nfr = torch.zeros((self.B,), dtype=torch.long)
        self.rows: list[Job | None] = [None] * self.B
        self.long: list[_Long] = []
        self.pending: collections.deque[Job] = collections.deque()
        self.low_steps = 0
        self.gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1))  # B = 1 path noise
        self.test_gen = torch.Generator().manual_seed(1234)  # host noise of self-test rows
        self.sup = lm.frame_mod.suppress.cpu()
        self.eos_id = lm.eos_id
        self.stats = collections.Counter()
        self.trace_bad: list = []
        self.trace_checked = collections.Counter()
        self._t_stats = time.time()
        self._acc = collections.Counter()
        self.pump = _NoisePump(self.V0, self.G - 1, self.V1)

    # ------------------------------------------------------------------ state
    def _small_state(self, B):
        d = self.device
        return [torch.zeros((B, self.H), dtype=BF, device=d), torch.zeros((B, self.V0), dtype=torch.float32, device=d),
                torch.zeros((B, self.V0), dtype=torch.float32, device=d),
                torch.zeros((B,), dtype=torch.long, device=d)]  # h, logits0, presence, nframes

    def live(self) -> int:
        """Occupied rows: streams and the pages they borrowed."""
        return sum(r is not None for r in self.rows) + sum(len(r.pages) for r in self.rows if r is not None)

    def page_rows(self) -> set:
        return {p for r in self.rows if r is not None for p in r.pages}

    def _pages_needed(self, j: Job) -> int:
        p = PREFILL_LEN + j.t
        return 0 if p < self.page_at else 1 + (p - self.page_at) // self.page_ps

    def busy(self) -> bool:
        return bool(self.pending) or self.live() > 0 or bool(self.long)

    def add(self, job: Job) -> None:
        job.t_in = time.time()
        self.pending.append(job)

    def cancel(self, jid) -> None:
        for j in self.pending:
            if j.jid == jid:
                j.cancelled = True
        for r in self.rows:
            if r is not None and r.jid == jid:
                r.cancelled = True
        for lr in self.long:
            if lr.job.jid == jid:
                lr.job.cancelled = True

    def _switch(self, target: int) -> None:
        """Make `target` the current bucket, moving the live rows into its rows 0..n-1."""
        if target == self.B:
            return
        live = sorted([i for i, r in enumerate(self.rows) if r is not None] + list(self.page_rows()))
        m = min(self.B, target)
        assert len(live) <= m, (len(live), self.B, target)
        new_of = {i: k for k, i in enumerate(live)}
        if live:
            idx = live + [live[0]] * (m - len(live))
            sK, sV = self.caches[self.B]
            dK, dV = self.caches[target]
            it = torch.tensor(idx, dtype=torch.long)
            self.move(dK, dV, sK, sV, it.to(self.device)).cpu()
            new = []
            for x in self.st:
                xh = x.cpu()
                y = torch.zeros((target,) + tuple(xh.shape[1:]), dtype=xh.dtype)
                y[:m] = xh[it]
                new.append(y.to(self.device))
            self.st = new
        else:
            self.st = self._small_state(target)
        pres = torch.zeros((target, self.V0), dtype=torch.float32)
        nfr = torch.zeros((target,), dtype=torch.long)
        if live:
            pres[: len(live)] = self.pres[live]
            nfr[: len(live)] = self.nfr[live]
        self.pres, self.nfr = pres, nfr
        self.rows = [self.rows[i] for i in live] + [None] * (target - len(live))  # (page rows: None, as padding)
        if any(r is not None and r.pages for r in self.rows):
            self.stats["switches with pages"] += 1
        for r in self.rows:
            if r is not None:
                r.pages = [new_of[p] for p in r.pages]
        self.stats[f"switch {self.B}->{target}"] += 1
        self.B = target
        self.low_steps = 0

    # ------------------------------------------------------------------ admission
    def _admit(self, ev) -> None:
        while self.pending and self.pending[0].cancelled:
            j = self.pending.popleft()
            ev.append((j.jid, "done", {"frames": 0, "cancelled": True}))
        live = self.live()
        n = min(MAX_ADMIT, self.maxB - live, len(self.pending))
        if n <= 0:
            return
        take = []
        while self.pending and len(take) < n:
            j = self.pending.popleft()
            if j.cancelled:
                ev.append((j.jid, "done", {"frames": 0, "cancelled": True}))
                continue
            try:
                prefill, trailing, n_ids = self.lm.emb.prompt(j.text, j.speaker, j.language)
                assert prefill.shape[0] == PREFILL_LEN, f"prompt prefill of {prefill.shape[0]} positions, expected {PREFILL_LEN}"
            except Exception as e:  # noqa: BLE001
                ev.append((j.jid, "error", f"{type(e).__name__}: {e}"))
                continue
            j.trailing, j.n_ids, j.cap = trailing, n_ids, MAX_POSITIONS - n_ids
            take.append((j, prefill))
        if not take:
            return
        need = live + len(take)
        target = next(b for b in self.buckets if b >= need)
        if target > self.B:
            self._switch(target)
        taken = self.page_rows()
        free = [i for i, r in enumerate(self.rows) if r is None and i not in taken]
        K, V = self.caches[self.B]
        now = time.time()
        for c in range(0, len(take), KP):
            chunk = take[c : c + KP]
            rows = []
            for j, _ in chunk:
                i = free.pop(0)
                self.rows[i] = j
                self.pres[i] = 0.0
                self.nfr[i] = 0
                j.t_admit = now
                rows.append(i)
            xs = [p for _, p in chunk]
            xs += [xs[-1]] * (KP - len(xs))
            rows += [rows[-1]] * (KP - len(rows))
            x = torch.stack(xs).to(self.device)
            self.st = list(self.prefill(x, torch.tensor(rows, dtype=torch.long).to(self.device), K, V, *self.st))
            self.stats["prefill calls"] += 1
        self.stats["admitted"] += len(take)

    # ------------------------------------------------------------------ the step
    @torch.inference_mode()
    def step(self) -> None:
        ev = []
        t0 = time.perf_counter()
        self._admit(ev)
        t1 = time.perf_counter()
        if any(r is not None for r in self.rows):
            self._batch_step(ev)
        t2 = time.perf_counter()
        for lr in list(self.long):
            self._long_step(lr, ev)
        live = self.live()
        smaller = next(b for b in self.buckets if b >= live)
        if self.page_buckets and any(r is not None and r.pages for r in self.rows):
            smaller = max(smaller, self.page_buckets[0])  # pages run on the page buckets' graphs only
        if smaller < self.B:
            self.low_steps += 1
            if self.low_steps >= SHRINK_AFTER:
                self._switch(smaller)
        else:
            self.low_steps = 0
        if ev:
            self.sink(ev)
        t3 = time.perf_counter()
        if live or self.long or t1 - t0 > 0:
            a = self._acc
            a["steps"] += 1
            a["admit_ms"] += 1000 * (t1 - t0)
            a["batch_ms"] += 1000 * (t2 - t1)
            a["total_ms"] += 1000 * (t3 - t0)
            a["rows"] += live
            a[f"B{self.B}"] += 1
        if time.time() - self._t_stats > 10 and self._acc["steps"]:
            a, self._acc = self._acc, collections.Counter()
            n = a["steps"]
            dt = time.time() - self._t_stats
            self._t_stats = time.time()
            self.log(f"[batch] {n} steps in {dt:.1f}s: step {a['total_ms'] / n:.1f} ms (admission {a['admit_ms'] / n:.1f}, batch {a['batch_ms'] / n:.1f}), "
                     f"live {a['rows'] / n:.1f}, {a['frames']} frames ({a['frames'] / dt:.0f}/s), buckets "
                     f"{ {k: v for k, v in a.items() if k.startswith('B')} }, pending {len(self.pending)}, long {len(self.long)}"
                     + ("" if not a["spec checked"] else
                        f"; spec per step: code0 {a['spec c0_ms'] / n:.1f}, chain {a['spec chain_ms'] / n:.1f}, verify {a['spec verify_ms'] / n:.1f}, "
                        f"fix-up {a['spec fix_ms'] / n:.1f} ms ({a['spec passes'] / n:.2f} passes), talker {a['spec talker_ms'] / n:.1f} ms, "
                        f"l0 wait {a['spec l0wait_ms'] / n:.1f}, overlap launch {a['spec launch_ms'] / n:.1f} ms ({a['spec relaunches'] / n:.3f} relaunches), "
                        f"device mismatches {a['spec mismatches']}/{a['spec checked']} codes"))

    def _row_host_in(self, j: Job, buf: torch.Tensor) -> None:
        tr = j.trailing
        buf[: self.H] = tr[j.t] if j.t < tr.shape[0] else self.lm.emb.tts_pad
        o = self.H
        if j.kind == "score":
            buf[o : o + self.G] = torch.tensor(j.codes[j.t], dtype=torch.float32)
            buf[o + self.G] = 1.0
        else:
            buf[o : o + self.G + 1] = 0.0
        buf[o + self.G + 1] = 1.0 if j.hostnoise else 0.0

    def _batch_step(self, ev) -> None:
        B = self.B
        for i, j in enumerate(self.rows):
            if j is None:
                continue
            if j.cancelled:
                self._finish(i, j, ev, cancelled=True)
            elif j.migrate_at is not None and j.t >= j.migrate_at:
                self._to_long(i)
            elif not self.pages and PREFILL_LEN + j.t >= S_BATCH:
                self._to_long(i)
        if self.pages:
            self._take_pages()
        B = self.B
        rows = self.rows
        scoring = any(j is not None and j.kind == "score" for j in rows)
        tracing = any(j is not None and j.trace for j in rows)
        # Overlap: the step's host inputs do not depend on l0, so they are built while the
        # previous talker still runs (not with score rows: those may finish below, after l0h).
        early = self.overlap and self.spec and not scoring and any(j is not None for j in rows)
        prep = self._step_inputs(B, rows) if early else None
        tw = time.perf_counter()
        l0h = prh = nfh = None
        if scoring or tracing:
            l0h = self.st[1].cpu()
            lsm = torch.log_softmax(l0h, -1) if scoring else None
            if tracing:
                prh, nfh = self.pres.clone(), self.nfr.clone()
            for i, j in enumerate(rows):
                if j is not None and j.kind == "score":
                    if j.t >= len(j.codes):
                        j.result = (j.lps, float(lsm[i, self.eos_id]))
                        ev.append((j.jid, "score", j.result))
                        self._finish(i, j, ev, report=False)
                    else:
                        j.lps.append([float(lsm[i, int(j.codes[j.t][0])])])
        if not any(j is not None for j in rows):
            return
        live = [i for i, j in enumerate(rows) if j is not None]
        if l0h is None:
            l0h = self.st[1].cpu()
        self._l0wait = 1000 * (time.perf_counter() - tw)
        if prep is None:
            prep = self._step_inputs(B, rows)
        trail, pos, trail_d = prep
        dev = self.device
        tnoise = {i: [] for i in live if rows[i].hostnoise}  # self-test rows: noise from test_gen, kept for the trace

        def draw(scores, V):
            """Exact Gumbel-max on the host for every row; teacher-forced rows take their code."""
            noise = gumbel_noise((B, V), self.gen)
            for i, lst in tnoise.items():
                noise[i] = gumbel_noise((V,), self.test_gen)
                lst.append(noise[i])
            return gumbel_pick(scores, noise)

        def force(c, g):
            for i in live:
                j = rows[i]
                if j.kind == "score":
                    c[i] = int(j.codes[j.t][g])
            return c

        K, V = self.caches[B]
        h, l0, pr, nf = self.st
        if self.spec:
            codes, subh, hostnoise, h, l0 = self._spec_frame(B, live, rows, l0h, h, trail, pos, K, V, trail_d)
        else:
            codes = torch.zeros((B, self.G), dtype=torch.long)
            codes[:, 0] = force(draw(code0_scores(l0h, self.pres, self.nfr, self.sup, self.eos_id), self.V0), 0)
            lg, Kp, Vp = self.predfirst(h, codes[:, 0].to(dev))
            subs = []
            for g in range(1, self.G):
                lgh = lg.cpu()
                subs.append(lgh)
                codes[:, g] = force(draw(lgh, self.V1), g)
                if g + 1 < self.G:
                    lg, Kp, Vp = self.prednext(codes[:, g].to(dev), self.gpos[(B, g + 1)], self.ghead[g + 1], Kp, Vp)
            h, l0 = self._talk(codes.to(dev), trail.to(dev), pos, K, V)
            subh = torch.stack(subs, 1)  # [B, 15, V1]
            hostnoise = {i: torch.cat(lst) for i, lst in tnoise.items()}
        self.st = [h, l0, pr, nf]
        self.pres[torch.arange(B), codes[:, 0]] = 1.0
        self.nfr += 1
        codes = codes.tolist()
        self.stats[f"steps B={B}"] += 1
        for i, j in enumerate(rows):
            if j is None:
                continue
            j.bucket_steps[B] += 1
            if j.trace:
                self._check_trace(j, f"B={B}", l0h[i], prh[i], nfh[i], hostnoise.get(i), subh[i], codes[i])
            self._after(i, j, codes[i], subh[i], ev)

    def _step_inputs(self, B, rows):
        """The step's host inputs: trail [B, H] FP32, positions [B] (host), and with the overlap the
        trail on the device; sets self._pg (TalkerInP's inputs on the device) on a page bucket."""
        dev = self.device
        trail = torch.zeros((B, self.H), dtype=torch.float32)
        for i, j in enumerate(rows):
            if j is None:
                continue
            tr = j.trailing
            t = j.t
            trail[i] = tr[t] if t < tr.shape[0] else self.lm.emb.tts_pad
        pos = [PREFILL_LEN + j.t if j is not None else 0 for j in rows]  # true positions (a leader's may be >= S_BATCH)
        groups = {i: list(j.pages) for i, j in enumerate(rows) if j is not None and j.pages}
        self._pg = None
        if self.pages and B in self.page_buckets:
            pi = page_inputs(pos, groups, P=self.page_p, page_at=self.page_at, S=S_BATCH, ps=self.page_ps)
            key = (B, tuple((k, tuple(v)) for k, v in sorted(groups.items())))
            st = self._pg_static.get(key)
            if st is None:  # snb, wsrc, qsrc, part, pw change only when a group does
                if len(self._pg_static) > 64:
                    self._pg_static.clear()
                st = self._pg_static[key] = tuple(x.to(dev) for x in pi[3:])
            self._pg = tuple(x.to(dev) for x in pi[:3]) + st
            if groups:
                self.stats[f"paged steps B={B}"] += 1
        else:
            assert not groups
        pos = torch.tensor(pos, dtype=torch.long)  # (TalkerIn only: no page live, all < page_at <= S_BATCH - 1)
        trail_d = trail.to(dev) if self.overlap and self.spec else None
        return trail, pos, trail_d

    def _talk(self, codes_d, trail_d, pos, K, V):
        """The talker step of the batch: TalkerInP on a page bucket (this step's inputs in
        self._pg), else TalkerIn. pos: host positions [B] (TalkerIn only)."""
        if self._pg is not None:
            return self.talkerp(codes_d, trail_d, *self._pg, K, V)
        return self.talkerin(codes_d, trail_d, pos.to(self.device), K, V)

    def _take_pages(self) -> None:
        """Before a step: every stream whose next position is past its rows takes a free row of
        the batch as its next page (growing to a page bucket, then to a larger bucket, when no
        row is free); one that needs more than page_p pages, or finds no row in the largest
        bucket, moves to the B = 1 path with its pages."""
        want = [j for j in self.rows if j is not None and self._pages_needed(j) > len(j.pages)
                and not (j.kind == "score" and j.t >= len(j.codes))]  # (a finished score row only reports this step)
        for j in want:
            t0 = time.perf_counter()
            if self._pages_needed(j) > self.page_p:
                self._to_long(self.rows.index(j))
                self.stats["page fallbacks to B=1 (pages full)"] += 1
                continue
            taken = self.page_rows()
            free = [i for i, r in enumerate(self.rows) if r is None and i not in taken]
            if self.B not in self.page_buckets or not free:
                bigger = [b for b in self.buckets if b > self.B and b in self.page_buckets and b >= self.live() + 1]
                if self.B in self.page_buckets and not free and not bigger:
                    self._to_long(self.rows.index(j))
                    self.stats["page fallbacks to B=1 (no free row)"] += 1
                    continue
                if not bigger:
                    self._to_long(self.rows.index(j))
                    self.stats["page fallbacks to B=1 (no page bucket)"] += 1
                    continue
                self._switch(bigger[0])
                taken = self.page_rows()
                free = [i for i, r in enumerate(self.rows) if r is None and i not in taken]
            pr = free[-1]  # from the top: admissions fill from the bottom
            j.pages.append(pr)
            self.stats["pages taken"] += 1
            self.log(f"[batch] stream {j.jid} long: row {self.rows.index(j)} takes row {pr} as KV page {len(j.pages)} at position "
                     f"{PREFILL_LEN + j.t} (B={self.B}, {1000 * (time.perf_counter() - t0):.1f} ms)")

    # ------------------------------------------------------------------ speculate and verify
    def _spec_frame(self, B, live, rows, l0h, h, trail, pos, K, V, trail_d=None):
        """One frame of the batch with the speculative chain: code 0 drawn on the host (as
        before), codes 1-15 proposed by PredChain and verified on the host (`_spec_codes`), then
        the talker step on the verified codes. Returns (codes [B, 16] host int64, sub-logits
        [B, 15, V1] host FP32, {row: the row's noise [V0 + 15 V1]} for traced rows, h', logits0')."""
        ph = self.phase
        ph.clear()
        ph["l0wait_ms"] = self._l0wait
        t0 = time.perf_counter()
        n0, n1 = self.pump.get(B)
        c0 = gumbel_pick(code0_scores(l0h, self.pres, self.nfr, self.sup, self.eos_id), n0)
        forced = torch.full((B, self.G - 1), -1, dtype=torch.long)
        verify = torch.zeros(B, dtype=torch.bool)
        for i in live:
            j = rows[i]
            if j.kind == "score":
                fr = j.codes[j.t]
                c0[i] = int(fr[0])
                forced[i] = torch.tensor([int(c) for c in fr[1:]], dtype=torch.long)
            else:
                verify[i] = True
        t1 = time.perf_counter()
        ph["c0_ms"] += 1000 * (t1 - t0)
        dev = self.device
        spec_t = ov = None
        if self.overlap:
            if trail_d is None:
                trail_d = trail.to(dev)
            ov = (trail_d, pos)
        elif self.spec_talker and self._pg is None:
            trail_d, pos_d = trail.to(dev), pos.to(dev)
            spec_t = (trail_d, pos_d)
        sub_codes, subs = self._spec_codes(B, h, c0, n1, forced, verify, K, V, spec_t, ov=ov)
        codes = torch.cat([c0[:, None], sub_codes], 1)
        if self.recheck:
            self._recheck(B, h, c0, n1, sub_codes, verify)
        t2 = time.perf_counter()
        if ph.pop("_no_spec_talker", 0):
            spec_t = None
        if ov is not None:
            codes_p, (h2, l2) = ph.pop("_ov")
            if not torch.equal(codes_p, codes):
                # a fix-up changed a code: the talker again on the final codes (same graph and
                # cache slot; the speculative launch's write at pos is masked on read and replaced)
                h2, l2 = self._talk(codes.to(dev), trail_d, pos, K, V)
                self.stats["talker relaunches"] += 1
                ph["relaunches"] += 1
                if self.relaunch_check:
                    self._check_relaunch(B, codes, trail_d, pos, K, V, h2, l2)
        elif spec_t is not None and ph["mismatches"] == 0:
            h2, l2 = ph.pop("_spec_out")
        else:
            ph.pop("_spec_out", None)
            h2, l2 = self._talk(codes.to(dev), trail.to(dev), pos, K, V)
            if spec_t is not None:
                self.stats["talker relaunches"] += 1
        t3 = time.perf_counter()
        ph["talker_ms"] += 1000 * (t3 - t2)
        noise = {i: torch.cat([n0[i], n1[i].reshape(-1)]) for i in live if rows[i].trace}
        self.stats["spec codes proposed"] += int(ph["checked"])
        self.stats["spec device mismatches"] += int(ph["mismatches"])
        self.stats["spec fix-up passes"] += int(ph["passes"])
        for k in ("l0wait_ms", "c0_ms", "chain_ms", "launch_ms", "verify_ms", "fix_ms", "talker_ms", "passes", "mismatches", "checked", "relaunches"):
            self._acc["spec " + k] += ph[k]
        return codes, subs, noise, h2, l2

    def _check_relaunch(self, B, codes, trail_d, pos, K, V, h2, l2) -> None:
        """Self-test of a relaunch: the talker launched once more on the final codes reads
        the slot at pos as written by the relaunch, where the relaunch read it as written by the
        speculative launch; both outputs must be bitwise equal (the slot is masked on read)."""
        h3, l3 = self._talk(codes.to(self.device), trail_d, pos, K, V)
        self.trace_checked["relaunch"] += 1
        a, b, c, d = h2.cpu(), h3.cpu(), l2.cpu(), l3.cpu()
        if torch.equal(a, b) and torch.equal(c, d):
            return
        # Only the rows of streams are served (their h' and logits0' feed their next frame);
        # page rows and free rows compute values nothing reads. Every differing row is logged.
        rd = ((a.float() - b.float()).abs().amax(1) > 0) | ((c - d).abs().amax(1) > 0)
        pages = self.page_rows()
        bad, other = [], []
        for i in rd.nonzero()[:, 0].tolist():
            j = self.rows[i]
            dh, dl = float((a[i].float() - b[i].float()).abs().max()), float((c[i] - d[i]).abs().max())
            if j is not None:
                bad.append((i, j.kind, j.t, int(pos[i]), list(j.pages), round(dh, 4), round(dl, 4)))
            else:
                other.append((i, "page" if i in pages else "free", int(pos[i]), round(dh, 4), round(dl, 4)))
        self.stats["relaunch check: non-stream rows differ"] += bool(other)
        self.log(f"[batch] relaunch check B={B}: stream rows differing {bad[:8]}; page/free rows differing {other[:8]} "
                 f"(groups {[(i, r.pages) for i, r in enumerate(self.rows) if r is not None and r.pages]})")
        if bad:
            self.trace_bad.append(("relaunch", f"B={B}", bad[:8]))

    def overlap_ab(self, seed: int, inject_all: bool = True):
        """Self-test and bench of the overlap: one frame of the current batch computed twice from the
        same state and the same fixed noise, by the control order (overlap off) and by the overlap
        with a corrupted proposal in every sampled row (forcing fix-ups and a talker relaunch).
        Neither result is kept (both write the same cache slots with the same final codes).
        Returns a list of differences (empty: codes, h' and logits0' bitwise equal) and the
        relaunch count of the overlap run."""
        B, rows = self.B, self.rows
        live = [i for i, j in enumerate(rows) if j is not None]
        assert live and not any(rows[i].kind == "score" for i in live)
        g = torch.Generator().manual_seed(seed)
        n0, n1 = gumbel_noise((B, self.V0), g), gumbel_noise((B, self.G - 1, self.V1), g)
        get, inj, ov = self.pump.get, self.inject, self.overlap
        self.pump.get = lambda B_: (n0, n1)
        out = {}
        try:
            l0h = self.st[1].cpu()
            h = self.st[0]
            K, V = self.caches[B]
            for mode in (False, True):
                self.overlap = mode
                self.inject = None
                if mode and inject_all:
                    cg = torch.Generator().manual_seed(seed + 1)
                    self.inject = lambda B_, ver: [(r, int(torch.randint(1, self.G, (1,), generator=cg)), int(torch.randint(0, self.V1, (1,), generator=cg)))
                                                   for r in ver.nonzero()[:, 0].tolist()]
                trail, pos, trail_d = self._step_inputs(B, rows)
                codes, _, _, h2, l2 = self._spec_frame(B, live, rows, l0h, h, trail, pos, K, V, trail_d)
                out[mode] = (codes, h2.cpu(), l2.cpu(), int(self.phase["relaunches"]))
        finally:
            self.pump.get, self.inject, self.overlap = get, inj, ov
        diffs = []
        (ca, ha, la, _), (cb, hb, lb, nr) = out[False], out[True]
        if not torch.equal(ca, cb):
            diffs.append(f"codes differ at {(ca != cb).nonzero().tolist()[:8]}")
        if not torch.equal(ha, hb):
            diffs.append(f"h' differs (max |d| {float((ha.float() - hb.float()).abs().max()):.3g})")
        if not torch.equal(la, lb):
            diffs.append(f"logits0' differ (max |d| {float((la - lb).abs().max()):.3g})")
        return diffs, nr

    def _recheck(self, B, h, c0, n1, codes, verify, tol: float = 1e-3):
        """Independent check of the verify loop's result: the chain rerun with every code forced to
        the final ones gives the logits of each code on its final prefix; each final code of a
        verified row must be the host's Gumbel-max draw on them (a rounding tie whose margin is
        < tol is excused: a fix-up pass ran at another batch size). Failures go to trace_bad."""
        _, sd = self._chain_host(B, h, c0, n1, codes)
        idx = verify.nonzero()[:, 0]
        if idx.numel() == 0:
            return
        n_sub, V1 = self.G - 1, self.V1
        sc = sd[idx].reshape(-1, V1)
        nz = n1[idx].reshape(-1, V1)
        want = gumbel_pick(sc, nz).view(-1, n_sub)
        self.trace_checked["recheck"] += want.numel()
        for k, g in (want != codes[idx]).nonzero().tolist():
            s = sc.view(-1, n_sub, V1)[k, g] / TEMPERATURE
            kth = torch.topk(s, TOP_K).values[-1]
            v = torch.where(s >= kth, s, NEG) + nz.view(-1, n_sub, V1)[k, g]
            t2 = torch.topk(v, 2).values
            margin = float(t2[0] - t2[1])
            if margin >= tol:
                self.trace_bad.append(("recheck", int(idx[k]), g + 1, int(codes[idx[k], g]), int(want[k, g]), round(margin, 4)))

    def _chain_host(self, B, h, c0, n1, forced):
        """The chain on B rows -> host (codes [B, 16], logits [B, 15, V1]); run at the padded size
        chain_pad[B] when one is set (h goes through the host: the pad rows are zeros)."""
        dev = self.device
        Bp = self.chain_pad.get(B, B)
        if Bp == B:
            cd, sd = self.chain(h, c0.to(dev), n1.reshape(B, -1).to(dev), forced.to(dev))
            return cd.cpu(), sd.cpu()
        hh = h.cpu() if h.device.type != "cpu" else h
        pad = Bp - B
        hp = torch.cat([hh, torch.zeros((pad,) + tuple(hh.shape[1:]), dtype=hh.dtype)])
        c0p = torch.cat([c0, torch.zeros(pad, dtype=c0.dtype)])
        n1p = torch.cat([n1.reshape(B, -1), torch.zeros((pad, n1[0].numel()), dtype=n1.dtype)])
        fp = torch.cat([forced, torch.full((pad, forced.shape[1]), -1, dtype=forced.dtype)])
        cd, sd = self.chain(hp.to(dev), c0p.to(dev), n1p.to(dev), fp.to(dev))
        return cd.cpu()[:B], sd.cpu()[:B]

    def _spec_codes(self, B, h, c0, n1, forced, verify, K=None, V=None, spec_t=None, ov=None):
        """Codes 1-15 of B rows: PredChain proposes them; the host verifies every proposal of a
        row with `verify` set (exact Gumbel-max `gumbel_pick` on the chain's logits for that
        position and the row's noise n1 [B, 15, V1]); a row whose first wrong code is at g gets
        the host's code there and is rerun (a fix-up pass, on the same chain graph as the first
        pass) with codes 1..g forced, until every code is the
        host's draw on logits computed from the final codes before it. Rows without `verify` take
        their forced codes (teacher forcing). Returns (codes [B, 15] int64, logits [B, 15, V1]).

        ov = (trail on the device, positions), with the overlap: once the chain's outputs are on the host,
        the talker is launched on the proposed codes (teacher-forced rows: their codes) before the
        verify; phase["_ov"] = (those codes [B, 16] host, (h', logits0'))."""
        ph = self.phase
        dev = self.device
        n_sub, V1 = self.G - 1, self.V1
        f_in = forced
        if self.inject is not None:
            f_in = forced.clone()
            for r, g, c in self.inject(B, verify):
                f_in[r, g - 1] = c
        t0 = time.perf_counter()
        if self.chain_pad.get(B, B) != B:
            spec_t = None  # (a padded chain's codes are not a [B, 16] device tensor)
            ph["_no_spec_talker"] = 1
            cd, subs = self._chain_host(B, h, c0, n1, f_in)
            codes = cd[:, 1:].clone()
        else:
            cd, sd = self.chain(h, c0.to(dev), n1.reshape(B, -1).to(dev), f_in.to(dev))
            if spec_t is not None:
                ph["_spec_out"] = self.talkerin(cd, spec_t[0], spec_t[1], K, V)
            codes = cd.cpu()[:, 1:].clone()
            subs = sd.cpu()
        t1 = time.perf_counter()
        ph["chain_ms"] += 1000 * (t1 - t0)
        if ov is not None:
            cp = codes.clone()
            tf = ~verify
            cp[tf] = torch.where(forced[tf] >= 0, forced[tf], cp[tf])
            cp = torch.cat([c0[:, None], cp], 1)
            ph["_ov"] = (cp, self._talk(cp.to(dev), ov[0], ov[1], K, V))
            t1b = time.perf_counter()
            ph["launch_ms"] += 1000 * (t1b - t1)
        start = torch.zeros(B, dtype=torch.long)  # codes before start[r] (0-based codebook-1 index) are final
        check = verify.clone()
        gpos = torch.arange(n_sub)[None, :]
        h_host = None
        while True:
            tv = time.perf_counter()
            idx = check.nonzero()[:, 0]
            if idx.numel() == 0:
                ph["verify_ms"] += 1000 * (time.perf_counter() - tv)
                break
            ok = gumbel_check(subs[idx].reshape(-1, V1), n1[idx].reshape(-1, V1), codes[idx].reshape(-1)).view(-1, n_sub)
            st = start[idx][:, None]
            bad = ~ok & (gpos >= st)
            anyb = bad.any(1)
            first = torch.where(anyb, bad.float().argmax(1), torch.full_like(start[idx], n_sub))
            ph["checked"] += int((first - start[idx] + anyb.long()).clamp(min=0).sum())
            ph["mismatches"] += int(anyb.sum())
            check[idx[~anyb]] = False
            bidx = idx[anyb].tolist()
            fg = first[anyb].tolist()
            if bidx:  # the host's draw at each row's first wrong code
                bt, gt = torch.tensor(bidx), torch.tensor(fg)
                wb = gumbel_pick(subs[bt, gt], n1[bt, gt])
                for k, (r, g) in enumerate(zip(bidx, fg)):
                    codes[r, g] = wb[k]
                    start[r] = g + 1
            done = [r for r in bidx if int(start[r]) >= n_sub]  # the wrong code was the last one
            for r in done:
                check[r] = False
            bidx = [r for r in bidx if int(start[r]) < n_sub]
            ph["verify_ms"] += 1000 * (time.perf_counter() - tv)
            if not bidx:  # (a wrong last code needs no rerun)
                break
            # a fix-up pass over the rows whose codes after the corrected one are on a wrong prefix
            tf = time.perf_counter()
            ph["passes"] += 1
            if ph["passes"] > 4 * n_sub:
                raise RuntimeError(f"speculative chain: no convergence after {ph['passes']} fix-up passes")
            nb = len(bidx)
            # The same chain graph as the first pass (never a smaller bucket's): every logit of
            # a row then comes from one compiled graph, so rerunning it gives the same values
            # (fix-ups on the B=16 graph were observed to give logits that differ in rounding from
            # the B=64 graph's, and the self-test's forced recheck saw other draws on 9 of 28080
            # codes).
            Bf = self.chain_pad.get(B, B)
            if h_host is None:
                h_host = h.cpu()
            sel = bidx + [bidx[0]] * (Bf - nb)
            ff = torch.full((Bf, n_sub), -1, dtype=torch.long)
            for k, r in enumerate(sel):
                ff[k, : int(start[r])] = codes[r, : int(start[r])]
            st_t = torch.tensor(sel, dtype=torch.long)
            cd2, sd2 = self.chain(h_host[st_t].to(dev), c0[st_t].to(dev), n1[st_t].reshape(Bf, -1).to(dev), ff.to(dev))
            c2, s2 = cd2.cpu()[:, 1:], sd2.cpu()
            for k, r in enumerate(bidx):
                g = int(start[r])
                codes[r, g:] = c2[k, g:]
                subs[r, g:] = s2[k, g:]
            ph["fix_ms"] += 1000 * (time.perf_counter() - tf)
        # rows that take forced codes (teacher forcing) keep them
        tf_rows = ~verify
        codes[tf_rows] = torch.where(forced[tf_rows] >= 0, forced[tf_rows], codes[tf_rows])
        return codes, subs

    def _after(self, i, j: Job, codes, sub, ev) -> None:
        """Per-row bookkeeping after a step: i is the batch row, or None on the B = 1 path."""
        if j.kind == "score":
            lps = torch.log_softmax(sub.float(), -1)
            fr = j.codes[j.t]
            j.lps[-1] += [float(lps[g - 1, int(fr[g])]) for g in range(1, self.G)]
            j.t += 1
            return
        if codes[0] == self.eos_id:
            self._finish(i, j, ev)
            return
        if j.t_first is None:
            j.t_first = time.time()
        ev.append((j.jid, "frame", codes))
        self._acc["frames"] += 1
        j.t += 1
        if j.t >= j.cap or (j.max_frames is not None and j.t >= j.max_frames):
            self._finish(i, j, ev)

    def _finish(self, i, j: Job, ev, cancelled=False, report=True) -> None:
        if i is None:
            self.long = [lr for lr in self.long if lr.job is not j]
        else:
            self.rows[i] = None
            j.pages = []  # (its page rows are free again: their slots are rewritten or masked)
        if report:
            ev.append((j.jid, "done", {"frames": j.t, "cancelled": cancelled}))
        if j.kind == "gen":
            now = time.time()
            first = f"{1000 * (j.t_first - j.t_in):.0f}" if j.t_first else "-"
            self.log(f"[generate] {len(j.text)} chars, {j.t} frames, queued {1000 * ((j.t_admit or now) - j.t_in):.0f} ms, first frame {first} ms after arrival, "
                     f"{now - j.t_in:.2f}s, steps by bucket {dict(j.bucket_steps)}{' cancelled' if cancelled else ''}")

    # ------------------------------------------------------------------ the B = 1 path
    def _to_long(self, i: int) -> None:
        """Move batch row i to the B = 1 path with a 2048-slot cache (same graph as the serial
        engine), copying its cache slots and state."""
        j = self.rows[i]
        K, V = self.caches[self.B]
        L, _, kvh, S, hd = K.shape
        p = PREFILL_LEN + j.t
        cache = []
        for X in (K, V):
            src = [i] + list(j.pages)
            try:
                xr = [X[:, r : r + 1].cpu() for r in src]
            except Exception:  # noqa: BLE001  (an eager slice the backend refuses: copy the whole cache)
                xa = X.cpu()
                xr = [xa[:, r : r + 1] for r in src]
            y = torch.zeros((L, 1, kvh, TALKER_SLOTS, hd), dtype=BF)
            if not j.pages:
                y[:, :, :, :S] = xr[0]
            else:  # the leader's slots < page_at, then each page's slots < ps (positions < p only)
                n = min(p, self.page_at)
                y[:, :, :, :n] = xr[0][:, :, :, :n]
                for k in range(len(j.pages)):
                    base = self.page_at + k * self.page_ps
                    n = max(0, min(p - base, self.page_ps))
                    y[:, :, :, base : base + n] = xr[1 + k][:, :, :, :n]
            cache.append(y.to(self.device))
        h, l0 = (x.cpu()[i : i + 1].to(self.device) for x in self.st[:2])
        pr, nf = self.pres[i : i + 1].to(self.device), self.nfr[i : i + 1].to(self.device)
        pos = torch.tensor([PREFILL_LEN + j.t], dtype=torch.long).to(self.device)
        self.long.append(_Long(j, cache, h, l0, [pos, pr, nf]))
        self.rows[i] = None
        npg = len(j.pages)
        j.pages = []
        self.stats["to B=1 path"] += 1
        if npg:
            self.stats["to B=1 path with pages"] += 1
        self.log(f"[batch] stream {j.jid} moves to the B=1 path at frame {j.t} (position {PREFILL_LEN + j.t}, {npg} pages)")

    def _long_step(self, lr: _Long, ev) -> None:
        j = lr.job
        lm = self.lm
        if j.cancelled:
            self._finish(None, j, ev, cancelled=True)
            return
        l_in = lr.logits.cpu() if (j.kind == "score" or j.trace) else None
        if j.kind == "score":
            lsm = torch.log_softmax(l_in[0].float(), -1)
            if j.t >= len(j.codes):
                j.result = (j.lps, float(lsm[self.eos_id]))
                ev.append((j.jid, "score", j.result))
                self._finish(None, j, ev, report=False)
                return
            j.lps.append([float(lsm[int(j.codes[j.t][0])])])
            hin = lm.host_input(j.trailing, j.t, forced=[int(c) for c in j.codes[j.t]])
        else:
            hin = lm.host_input(j.trailing, j.t, self.test_gen if j.hostnoise else self.gen)
        p_in, n_in = (lr.state[1].cpu(), lr.state[2].cpu()) if j.trace else (None, None)
        codes, lr.h, lr.logits, sub = lm.frame_step(lr.h, lr.logits, lr.cache, lr.state, hin)
        codes = codes[0].cpu().tolist()
        subh = sub[0].cpu() if (j.kind == "score" or j.trace) else None
        j.bucket_steps[1] += 1
        if j.trace:
            self._check_trace(j, "B=1", l_in[0], p_in[0], n_in[0], hin[0, : lm.frame_mod.o_trail] if j.hostnoise else None, subh, codes)
        self._after(None, j, codes, subh, ev)

    # ------------------------------------------------------------------ self-test helpers
    def _check_trace(self, j, where, l_in, p_in, n_in, noise, sub, codes) -> None:
        """Recompute a sampled row's codes on the host from the graph's own inputs and outputs:
        with host noise, the exact host Gumbel-max (full vocabulary) must give the same codes
        (a rounding tie whose margin is < 1e-3 is excused); with in-graph noise, every code must
        lie in the host's top-k set."""
        f = self.lm.frame_mod
        scores = [code0_scores(l_in[None].float(), p_in[None], n_in[None], self.sup, self.eos_id)]
        scores += [sub[g - 1][None].float() for g in range(1, self.G)]
        for g, sc in enumerate(scores):
            self.trace_checked[where] += 1
            s = sc / TEMPERATURE
            kth = torch.topk(s, TOP_K, dim=-1).values[..., -1:]
            if noise is None:
                if not bool(s[0, codes[g]] >= kth[0, 0]):
                    self.trace_bad.append((where, j.jid, j.t, g, codes[g], "outside the top-k set"))
                continue
            o = 0 if g == 0 else f.o_g1 + (g - 1) * f.V1
            nz = noise[o : o + (f.V0 if g == 0 else f.V1)][None]
            want = int(gumbel_pick(sc, nz)[0])
            if want != codes[g]:
                v = torch.where(s >= kth, s, NEG) + nz
                top2 = torch.topk(v[0], 2).values
                margin = float(top2[0] - top2[1])
                if margin >= 1e-3:
                    # diagnosis: ranks of both codes in the scores, the k-th value, both scores
                    rk = lambda c: int((s[0] > s[0, c]).sum())  # noqa: E731
                    self.trace_bad.append((where, j.jid, j.t, g, codes[g], want, round(margin, 4), f"ranks {rk(codes[g])}/{rk(want)}",
                                           f"s {float(s[0, codes[g]]):.4f}/{float(s[0, want]):.4f} kth {float(kth[0, 0]):.4f}"))

    def run_until_idle(self, max_steps=100000) -> int:
        n = 0
        while self.busy() and n < max_steps:
            self.step()
            n += 1
        return n
