"""Talker-launch overlap benchmark (needs one NeuronCore; run inside the server image).

The batched LM step with the talker launched on the chain's proposals right after the chain's D2H
(before the host verify) and the next step's inputs built before the codebook-0 logits wait
(BatchLM.overlap, the default), against the plain order (overlap off), on the serving graphs
(replica.warm: pages on, B=32 chain padded to 64).

(a) idle, closed loop with rows topped up, B=16, 32 and 64, plain and overlap back to back; one
    stream of each run holds a KV page (TalkerInP with a group live, the page split at
    PREFILL_LEN + 60): batch step median / p99, phases (l0wait, c0, chain, launch, verify, fix,
    talker), uninjected relaunches per step;
(b) B=64 under 2 x 3-thread CPU burners: plain, overlap, plain, overlap;
(c) exactness: (c1) traced + rechecked rows with the relaunch check (the talker launched once more
    on the final codes after every relaunch: bitwise-equal h'/logits0'), clean and with a corrupted
    proposal in every row every 2nd step, at B=32 and B=64 with the page row; (c2) BatchLM.overlap_ab:
    the same frame by the plain order and by the overlap with every proposal corrupted (relaunch),
    on the same state and fixed noise: codes, h', logits0' bitwise equal; also uncorrupted.

The summary's acceptance lines: K1 B=64 idle median <= 50 ms; K2 B=64 under burners <= 0.88 x
plain; K3 no difference in (c); K4 uninjected relaunches <= 0.10 per step at B=64; K5 overlap chain
phase <= plain + 3 ms at B=64. Exit 1 on an exactness failure or error event, 3 on a stall.

    NEURON_RT_VISIBLE_CORES=0 python benchmarks/overlap_bench.py    (env: STEPS=200 QWEN3TTS_LM_THREADS=2)
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QWEN3TTS_MAX_ADMIT", "64")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "1000000")

from engine import server  # noqa: E402

server.setup_env()  # compiler caches under $VIBESYS_CACHE_DIR, cwd in a fresh temp dir

import torch  # noqa: E402

from engine import replica  # noqa: E402
from engine.batch import S_BATCH, BatchLM, Job  # noqa: E402
from engine.runtime import LM, PREFILL_LEN  # noqa: E402

STEPS = int(os.environ.get("STEPS", "200"))
D_STEPS = int(os.environ.get("D_STEPS", "40"))
AB_TRIALS = int(os.environ.get("AB_TRIALS", "8"))
HANG_S = float(os.environ.get("HANG_S", "300"))
WARM_HANG_S = float(os.environ.get("WARM_HANG_S", "1500"))
PAGE_AT_FRAME = int(os.environ.get("PAGE_AT_FRAME", "60"))
T0 = time.time()
last = [time.time()]
limit = [WARM_HANG_S]
TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a lantern, a thermos of "
        "strong tea and a worn notebook in which he recorded the color of the sky, the direction of the wind, the "
        "ships that passed along the horizon and the names of the birds that rested on the railing before they "
        "continued their long journey south across the gray and restless sea.")


def log(m):
    print(f"[ov {time.time() - T0:7.1f}s] {m}", flush=True)


def beat(stage=None):
    last[0] = time.time()
    if stage:
        log(stage)


def hang_watch():
    while True:
        time.sleep(5)
        if time.time() - last[0] > limit[0]:
            log(f"STALL: no progress for {limit[0]:.0f}s; exiting")
            os._exit(3)


threading.Thread(target=hang_watch, daemon=True).start()

device = os.environ.get("QWEN3TTS_DEVICE") or server.pick_device()
compile_ = device.startswith("neuron")
torch.set_num_threads(int(os.environ.get("QWEN3TTS_LM_THREADS", "2")))
weights = os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
BUCKETS = tuple(int(b) for b in os.environ.get("BENCH_BUCKETS", "16,32,64").split(","))
BIG = max(BUCKETS)
beat("loading the checkpoint")
lm = LM(weights, log=log)
lm.to_device(device, compile_=compile_)
events = []
blm = BatchLM(lm, events.extend, compile_, buckets=BUCKETS, log=log, spec=True)
log(f"buckets {blm.buckets}, chain pad {blm.chain_pad}, pages {blm.pages} (page buckets {blm.page_buckets}, P={blm.page_p})")
replica.warm(blm, beat, log)
limit[0] = HANG_S
beat("warm-up done")
blm.page_at, blm.page_ps = PREFILL_LEN + PAGE_AT_FRAME, S_BATCH - 1  # the page row takes its page after 60 frames
fails = []
res = {}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


bt = []
orig_bs = blm._batch_step


def timed_bs(ev):
    t0 = time.perf_counter()
    orig_bs(ev)
    bt.append(1000 * (time.perf_counter() - t0))


blm._batch_step = timed_bs


def drain():
    for r in blm.rows:
        if r is not None:
            blm.cancel(r.jid)
    for j in blm.pending:
        j.cancelled = True
    blm.run_until_idle(500)
    beat()


def topup(tag, trace=False):
    for i in range(blm.B - blm.live() - len(blm.pending)):
        blm.add(Job((tag, "top", time.time(), i), "gen", TEXT, max_frames=PAGE_AT_FRAME - 10, trace=trace))


def setup(B, tag, page=True, trace=False):
    """Bucket B, idle; a long stream that takes a KV page; the rest of the rows topped up."""
    assert not blm.busy()
    blm._switch(B)
    lj = None
    if page:
        lj = Job((tag, "long", time.time()), "gen", " ".join([TEXT] * 4))
        blm.add(lj)
        for _ in range(PAGE_AT_FRAME + 2):
            blm.step()
            beat()
        if not lj.pages:
            log(f"WARNING {tag}: the long stream holds no page (t={lj.t}, rows {lj in blm.rows})")
    topup(tag, trace)
    while blm.pending:
        blm.step()
        beat()
    assert blm.B == B, (blm.B, B)
    return lj


PH = ("l0wait_ms", "c0_ms", "chain_ms", "launch_ms", "verify_ms", "fix_ms", "talker_ms")


def run(name, B, steps, overlap, page=True, trace=False, recheck=False, inject_every=0, rcheck=False):
    blm.overlap, blm.recheck, blm.relaunch_check = overlap, recheck, rcheck
    cg = torch.Generator().manual_seed(B)
    cnt = [0]

    def inject(B_, ver):
        cnt[0] += 1
        if not inject_every or cnt[0] % inject_every:
            return []
        return [(r, int(torch.randint(1, 16, (1,), generator=cg)), int(torch.randint(0, blm.V1, (1,), generator=cg))) for r in ver.nonzero()[:, 0].tolist()]

    lj = setup(B, name, page, trace)
    blm.inject = inject if inject_every else None
    blm.step()
    blm.stats.clear()
    bt.clear()
    live, ph, ts = [], [], []
    for _ in range(steps):
        topup(name, trace)
        t0 = time.perf_counter()
        blm.step()
        ts.append(1000 * (time.perf_counter() - t0))
        live.append(blm.live())
        ph.append(dict(blm.phase))
        beat()
    blm.inject = None
    st = dict(blm.stats)
    btm = list(bt[-steps:])
    drain()
    blm.recheck = blm.relaunch_check = False
    out = {"B": B, "overlap": overlap, "med": pct(btm, 0.5), "p90": pct(btm, 0.9), "p99": pct(btm, 0.99), "step_med": pct(ts, 0.5),
           "live": sum(live) / len(live), "relaunch_per_step": st.get("talker relaunches", 0) / steps,
           "passes_per_step": st.get("spec fix-up passes", 0) / steps, "paged_steps": st.get(f"paged steps B={B}", 0),
           "long_pages": len(lj.pages) if lj is not None else 0, "phases": {k[:-3]: pct([p.get(k, 0.0) for p in ph], 0.5) for k in PH},
           "mism": st.get("spec device mismatches", 0), "proposed": st.get("spec codes proposed", 0)}
    log(f"{name}: B={B} {'OVERLAP' if overlap else 'control'} live {out['live']:.1f}: batch step median {out['med']:.1f} p90 {out['p90']:.1f} "
        f"p99 {out['p99']:.1f} ms (whole step {out['step_med']:.1f}); {1000 * out['live'] / out['med']:.0f} frames/s; phases median ms "
        + ", ".join(f"{k} {v:.2f}" for k, v in out["phases"].items())
        + f"; passes/step {out['passes_per_step']:.3f}, relaunches/step {out['relaunch_per_step']:.3f}, paged steps {out['paged_steps']}/{steps}, "
        f"device mismatches {out['mism']}/{out['proposed']}")
    return out


with torch.inference_mode():
    # ------------------------------------------------------------------ (a) idle
    for B in BUCKETS:
        for ov in (False, True):
            res[f"a{B}{'o' if ov else 'c'}"] = run("(a) idle", B, STEPS, ov)

    # ------------------------------------------------------------------ (c1) traced + rechecked + relaunch check
    blm.trace_bad.clear()
    blm.trace_checked.clear()
    rel = 0
    for B in sorted(BUCKETS)[-2:]:
        run("(c1) traced+rechecked clean", B, D_STEPS, True, trace=True, recheck=True, rcheck=True)
        o = run("(c1) traced+rechecked corrupted every 2nd step", B, D_STEPS, True, trace=True, recheck=True, inject_every=2, rcheck=True)
        rel += o["relaunch_per_step"] * D_STEPS
    log(f"(c1) checked {dict(blm.trace_checked)}, relaunches {rel:.0f}, failures {len(blm.trace_bad)}: {blm.trace_bad[:10]}")
    if blm.trace_bad:
        fails.append(f"(c1) {len(blm.trace_bad)} failures: {blm.trace_bad[:10]}")
    if not blm.trace_checked["relaunch"]:
        fails.append("(c1) no relaunch was checked")

    # ------------------------------------------------------------------ (c2) overlap vs control on the same state and noise
    nab = nrel = 0
    for B in BUCKETS:
        lj = setup(B, f"c2-{B}")
        groups = sum(1 for r in blm.rows if r is not None and r.pages)
        for t in range(AB_TRIALS):
            for inj in (True, False):
                diffs, nr = blm.overlap_ab(seed=1000 * B + t, inject_all=inj)
                nab += 1
                nrel += nr
                if diffs:
                    fails.append(f"(c2) B={B} trial {t} injected={inj}: {diffs}")
                if inj and not nr:
                    fails.append(f"(c2) B={B} trial {t}: no relaunch after corrupted proposals")
            topup(f"c2-{B}")
            blm.step()
            beat()
        log(f"(c2) B={B} ({groups} stream(s) with a page): {2 * AB_TRIALS} A/B frames, failures so far {len(fails)}")
        drain()
    log(f"(c2) overlap vs control: {nab} frames compared, {nrel} relaunches, {'all identical' if not any(f.startswith('(c2)') for f in fails) else 'DIFFERENCES'}")

    # ------------------------------------------------------------------ (b) burners
    if True:
        burn = ("import torch, time\ntorch.set_num_threads(3)\na = torch.randn(1024, 1024)\n"
                "t = time.time()\nwhile time.time() - t < 900:\n    a = (a @ a).tanh_()\n")
        procs = [subprocess.Popen([sys.executable, "-c", burn]) for _ in range(2)]
        time.sleep(3)
        n = int(os.environ.get("B_STEPS", str(max(60, STEPS // 2))))
        res["b_c1"] = run("(b) burners", BIG, n, False)
        res["b_o1"] = run("(b) burners", BIG, n, True)
        res["b_c2"] = run("(b) burners (repeat)", BIG, n, False)
        res["b_o2"] = run("(b) burners (repeat)", BIG, n, True)
        for p in procs:
            p.kill()

errs = [e for e in events if e[1] == "error"]
if errs:
    fails.append(f"error events: {errs[:4]}")

K = {}
if True:
    c64, o64 = res[f"a{BIG}c"], res[f"a{BIG}o"]
    lc = (res["b_c1"]["med"] + res["b_c2"]["med"]) / 2
    lo = (res["b_o1"]["med"] + res["b_o2"]["med"]) / 2
    K["K1 B=64 idle median <= 50 ms"] = (o64["med"], o64["med"] <= 50)
    K["K2 B=64 burners overlap/control <= 0.88"] = (lo / lc, lo / lc <= 0.88)
    K["K4 uninjected relaunches/step <= 0.10"] = (o64["relaunch_per_step"], o64["relaunch_per_step"] <= 0.10)
    K["K5 chain phase overlap - control <= 3 ms"] = (o64["phases"]["chain"] - c64["phases"]["chain"],
                                                     o64["phases"]["chain"] - c64["phases"]["chain"] <= 3)
K["K3 exactness"] = (len(fails), not fails)
for B in BUCKETS:
    c, o = res[f"a{B}c"], res[f"a{B}o"]
    log(f"[summary] idle B={B}: control {c['med']:.1f} (p99 {c['p99']:.1f}) -> overlap {o['med']:.1f} (p99 {o['p99']:.1f}) ms, "
        f"ratio {o['med'] / c['med']:.3f}; l0wait {c['phases']['l0wait']:.1f} -> {o['phases']['l0wait']:.1f}, chain {c['phases']['chain']:.1f} -> "
        f"{o['phases']['chain']:.1f}, verify {c['phases']['verify']:.1f} -> {o['phases']['verify']:.1f}, talker {c['phases']['talker']:.1f} -> "
        f"{o['phases']['talker']:.1f}; relaunches/step {o['relaunch_per_step']:.3f}; paged steps {c['paged_steps']}/{o['paged_steps']}")
if True:
    log(f"[summary] burners B={BIG}: control {res['b_c1']['med']:.1f} / {res['b_c2']['med']:.1f}, overlap {res['b_o1']['med']:.1f} / {res['b_o2']['med']:.1f} ms")
for k, (v, ok) in K.items():
    log(f"[K] {k}: {v:.3f} {'PASS' if ok else 'TRIPPED'}")
for f in fails:
    log(f"FAILED: {f}")
art = os.environ.get("VIBESYS_ARTIFACTS_DIR")
if art:
    Path(art).mkdir(parents=True, exist_ok=True)
    Path(art, "overlap_bench.json").write_text(json.dumps({"res": res, "K": {k: [v, ok] for k, (v, ok) in K.items()}, "fails": fails}, indent=1, default=str))
log(f"[summary] kill lines: {'PASS' if all(ok for _, ok in K.values()) else 'TRIPPED'}")
sys.exit(1 if fails else 0)
