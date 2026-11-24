"""Speculative code-predictor chain benchmark (needs one NeuronCore; run inside the server image).

Loads the LM on the first visible core and one BatchLM (buckets 16, 32, 64; chain also at 8),
then, all on real hidden states with the loop closed (the talker's h fed back each step):

(a) baseline: the non-speculative BatchLM step (PredFirst + 14 PredNext, host draw per code) at
    B=32 with 32 live rows: step ms (batch part and whole step) and host CPU ms per step;
(b) the chain alone (PredChain with the topk-free proposal sampler) at B=8, 16, 32, 64: compile s,
    synced ms per call (incl. the D2H of codes and logits) and pipelined ms per call;
(c) the speculative step (host code 0, chain, D2H, host verify, fix-up, talker) at B=32 and B=64:
    median / p90 / p99 ms, phase split, fix-up passes per step, device mismatches per code; also
    with the talker launched speculatively on the proposals (SPEC_TALKER);
(d) exactness: (1) every row traced (each final code recomputed on the host from the step's own
    logits and noise, incl. code 0) and every step rechecked (chain rerun with the final codes
    forced: each code must be the host draw on its final prefix), at B=32 and B=64, clean and with
    a corrupted proposal per row every 2nd step; (2) on real states at B=32, spec codes vs the
    sequential host path (PredFirst/PredNext + host Gumbel-max) on the same noise, clean and
    corrupted (rounding ties < 1e-3 excused and counted);
(e) (c) at B=64 again with 2 CPU-burner processes x 3 threads (approximating the host load of
    three replicas).

The final "kill line" compares against the latency budget the chain was designed for (B=32 median
<= 46 ms, B=64 <= 72 ms, <= 76 ms under load, <= 0.5 fix-up passes per step, exact).
Exit 1 on any exactness failure or error event; 3 on a stall (no progress for HANG_S s).

    NEURON_RT_VISIBLE_CORES=0 python benchmarks/spec_chain_bench.py    (env: STEPS=200 QWEN3TTS_LM_THREADS=2)
"""
import os, sys, threading, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QWEN3TTS_MAX_ADMIT", "64")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "1000000")

from engine import server  # noqa: E402

server.setup_env()  # compiler caches under $VIBESYS_CACHE_DIR, cwd in a fresh temp dir

import torch  # noqa: E402

from engine.batch import BatchLM, Job  # noqa: E402
from engine.model import BF  # noqa: E402
from engine.batch import KP  # noqa: E402
from engine.runtime import LM  # noqa: E402
from engine.sampling import NEG, TEMPERATURE, TOP_K, gumbel_noise, gumbel_pick  # noqa: E402

STEPS = int(os.environ.get("STEPS", "200"))
D_STEPS = int(os.environ.get("D_STEPS", "40"))  # steps per (d1) phase
D_TRIALS = int(os.environ.get("D_TRIALS", "30"))  # (d2) trials
HANG_S = float(os.environ.get("HANG_S", "300"))
WARM_HANG_S = float(os.environ.get("WARM_HANG_S", "1500"))  # one cold compile stage (measured: B=64 chain 243 s, frame graph 489 s)
T0 = time.time()
last = [time.time()]
limit = [WARM_HANG_S]
TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a lantern, a thermos of "
        "strong tea and a worn notebook in which he recorded the color of the sky, the direction of the wind, the "
        "ships that passed along the horizon and the names of the birds that rested on the railing before they "
        "continued their long journey south across the gray and restless sea.")


def log(m):
    print(f"[sc {time.time() - T0:7.1f}s] {m}", flush=True)


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
beat("loading the checkpoint")
lm = LM(weights, log=log)
lm.to_device(device, compile_=compile_)
events = []
blm = BatchLM(lm, events.extend, compile_, buckets=(16, 32, 64), log=log, spec=False)
CHAIN_B = (8, 16, 32, 64)
blm.chain_buckets = set(CHAIN_B)
d = blm.device
n_sub, V1 = blm.G - 1, blm.V1
fails = []
res = {}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


# ------------------------------------------------------------------ warm-up / compile
with torch.inference_mode():
    x = torch.zeros((KP, 10, blm.H), dtype=BF, device=d)
    for B in (32, 64):
        K, V = blm.caches[B]
        st = blm._small_state(B)
        beat(f"[warm] prefill B={B}")
        t0 = time.time()
        blm.prefill(x, torch.zeros(KP, dtype=torch.long, device=d), K, V, *st)[0].cpu()
        log(f"[warm] prefill B={B}: {time.time() - t0:.1f}s")
        beat(f"[warm] talker B={B}")
        t0 = time.time()
        z = torch.zeros(B, dtype=torch.long, device=d)
        blm.talkerin(torch.zeros((B, blm.G), dtype=torch.long, device=d), torch.zeros((B, blm.H), device=d), z, K, V)[1].cpu()
        log(f"[warm] talker B={B}: {time.time() - t0:.1f}s")
    B = 32
    beat("[warm] PredFirst/PredNext B=32 (control)")
    t0 = time.time()
    z = torch.zeros(B, dtype=torch.long, device=d)
    lg, Kp, Vp = blm.predfirst(torch.zeros((B, blm.H), dtype=BF, device=d), z)
    blm.prednext(z, blm.gpos[(B, 2)], blm.ghead[2], Kp, Vp)[0].cpu()
    log(f"[warm] PredFirst/PredNext B=32: {time.time() - t0:.1f}s")
    compile_s = {}
    for B in CHAIN_B:
        beat(f"[warm] chain B={B}")
        t0 = time.time()
        blm.chain(torch.zeros((B, blm.H), dtype=BF, device=d), torch.zeros(B, dtype=torch.long, device=d),
                  torch.zeros((B, n_sub * V1), device=d), torch.full((B, n_sub), -1, dtype=torch.long, device=d))[1].cpu()
        compile_s[B] = time.time() - t0
        log(f"[warm] chain B={B}: {compile_s[B]:.1f}s")
limit[0] = HANG_S
beat("warm-up done")

# ------------------------------------------------------------------ (b) the chain alone
gen = torch.Generator().manual_seed(7)
with torch.inference_mode():
    for B in CHAIN_B:
        h = (torch.randn((B, blm.H), generator=gen) * 0.5).to(BF).to(d)
        c0 = torch.randint(0, 2048, (B,), generator=gen).to(d)
        nz = gumbel_noise((B, n_sub * V1), gen).to(d)
        fo = torch.full((B, n_sub), -1, dtype=torch.long).to(d)
        for _ in range(3):
            blm.chain(h, c0, nz, fo)[1].cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            cd, sd = blm.chain(h, c0, nz, fo)
            cd.cpu(), sd.cpu()
            ts.append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        outs = [blm.chain(h, c0, nz, fo) for _ in range(10)]
        outs[-1][1].cpu()
        piped = 100 * (time.perf_counter() - t0)
        res[f"chain{B}"] = pct(ts, 0.5)
        log(f"(b) chain B={B}: compile {compile_s[B]:.1f}s, synced median {pct(ts, 0.5):.2f} ms (min {min(ts):.2f}, incl. D2H of codes + "
            f"{B * n_sub * V1 * 4 / 1e6:.1f} MB logits), pipelined {piped:.2f} ms/call")
        beat()


# ------------------------------------------------------------------ batch helpers
def fill(B, n_rows, trace=False, tag="r"):
    """Make B the bucket (idle) and admit n_rows long sampled rows."""
    assert not blm.busy()
    blm._switch(B)
    js = []
    for i in range(n_rows):
        j = Job((tag, B, i, time.time()), "gen", TEXT, max_frames=440, trace=trace)
        blm.add(j)
        js.append(j)
    while blm.pending:
        blm.step()
        beat()
    return js


def topup(B, tag, trace=False):
    free = blm.B - blm.live()
    for i in range(free):
        blm.add(Job((tag, "top", time.time(), i), "gen", TEXT, max_frames=440, trace=trace))


def drain():
    for r in blm.rows:
        if r is not None:
            blm.cancel(r.jid)
    for j in blm.pending:
        j.cancelled = True
    blm.run_until_idle(500)
    beat()


bt = []
orig_bs = blm._batch_step


def timed_bs(ev):
    t0 = time.perf_counter()
    orig_bs(ev)
    bt.append(1000 * (time.perf_counter() - t0))


blm._batch_step = timed_bs


def run(name, B, steps, spec, spec_talker=False, trace=False, recheck=False, inject_every=0):
    blm.spec, blm.spec_talker, blm.recheck = spec, spec_talker, recheck
    cg = torch.Generator().manual_seed(B)
    cnt = [0]

    def inject(B_, ver):
        cnt[0] += 1
        if not inject_every or cnt[0] % inject_every:
            return []
        return [(r, int(torch.randint(1, 16, (1,), generator=cg)), int(torch.randint(0, V1, (1,), generator=cg))) for r in ver.nonzero()[:, 0].tolist()]

    blm.inject = inject if inject_every else None
    fill(B, B, trace=trace, tag=name)
    blm.step()
    blm.stats.clear()
    bt.clear()
    ts, cpu, live, ph = [], [], [], []
    for _ in range(steps):
        topup(B, name, trace)
        t0, c0 = time.perf_counter(), time.thread_time()
        blm.step()
        ts.append(1000 * (time.perf_counter() - t0))
        cpu.append(1000 * (time.thread_time() - c0))
        live.append(blm.live())
        ph.append(dict(blm.phase))
        beat()
    blm.inject = None
    st = dict(blm.stats)
    btm = list(bt[-steps:])  # the timed steps' batch part (before the drain's steps)
    drain()
    blm.recheck = False
    out = {"med": pct(btm, 0.5), "p90": pct(btm, 0.9), "p99": pct(btm, 0.99), "step_med": pct(ts, 0.5), "cpu": sum(cpu) / len(cpu),
           "live": sum(live) / len(live), "stats": st}
    msg = (f"{name}: B={B} live {out['live']:.1f}: batch step median {out['med']:.1f} p90 {out['p90']:.1f} p99 {out['p99']:.1f} max {max(btm):.1f} ms; "
           f"whole step median {out['step_med']:.1f} ms; host CPU {out['cpu']:.1f} ms/step; {1000 * out['live'] / out['med']:.0f} frames/s")
    if spec:
        keys = ("c0_ms", "chain_ms", "verify_ms", "fix_ms", "talker_ms")
        msg += "; phases median ms " + ", ".join(f"{k[:-3]} {pct([p.get(k, 0.0) for p in ph], 0.5):.2f}" for k in keys)
        msg += f" (mean fix {sum(p.get('fix_ms', 0.0) for p in ph) / len(ph):.2f})"
        npass = st.get("spec fix-up passes", 0)
        mism, prop = st.get("spec device mismatches", 0), max(1, st.get("spec codes proposed", 0))
        out["passes"] = npass / steps
        out["mism_rate"] = mism / prop
        msg += f"; fix-up passes {npass} ({npass / steps:.3f}/step), device mismatches {mism}/{prop} codes ({mism / prop:.2e})"
        if spec_talker:
            msg += f", talker relaunches {st.get('talker relaunches', 0)}"
    log(msg)
    return out


# ------------------------------------------------------------------ (a) control
with torch.inference_mode():
    a = run("(a) control (PredFirst/PredNext + host draws)", 32, min(STEPS, 80), spec=False)
    res["a"] = a["med"]

    # ------------------------------------------------------------------ (c) speculative step
    c32 = run("(c) spec", 32, STEPS, spec=True)
    c64 = run("(c) spec", 64, STEPS, spec=True)
    c32t = run("(c) spec + speculative talker", 32, STEPS // 2, spec=True, spec_talker=True)
    c64t = run("(c) spec + speculative talker", 64, STEPS // 2, spec=True, spec_talker=True)

    # ------------------------------------------------------------------ (d1) traced + rechecked
    blm.trace_bad.clear()
    blm.trace_checked.clear()
    for B in (32, 64):
        run("(d1) traced+rechecked clean", B, D_STEPS, spec=True, trace=True, recheck=True)
        run("(d1) traced+rechecked corrupted every 2nd step", B, D_STEPS, spec=True, trace=True, recheck=True, inject_every=2)
    log(f"(d1) codes checked {dict(blm.trace_checked)}, mismatches {len(blm.trace_bad)}: {blm.trace_bad[:10]}")
    if blm.trace_bad:
        fails.append(f"(d1) {len(blm.trace_bad)} final codes differ from the host draw: {blm.trace_bad[:10]}")
    if not sum(blm.trace_checked.values()):
        fails.append("(d1) nothing checked")

    # ------------------------------------------------------------------ (d2) vs the sequential host path
    blm.spec = True
    fill(32, 32, tag="d2")
    for _ in range(5):
        blm.step()
    B = blm.B
    h = blm.st[0]
    tot = diff = excused = 0
    gen = torch.Generator().manual_seed(11)

    def seq_codes(h, c0, n1):
        lg, Kp, Vp = blm.predfirst(h, c0.to(d))
        codes, subs = [], []
        for g in range(1, blm.G):
            lgh = lg.cpu()
            subs.append(lgh)
            c = gumbel_pick(lgh, n1[:, g - 1])
            codes.append(c)
            if g + 1 < blm.G:
                lg, Kp, Vp = blm.prednext(c.to(d), blm.gpos[(B, g + 1)], blm.ghead[g + 1], Kp, Vp)
        return torch.stack(codes, 1), torch.stack(subs, 1)

    maxd = meand = 0.0
    for trial in range(D_TRIALS):
        c0 = torch.randint(0, 2048, (B,), generator=gen)
        n1 = gumbel_noise((B, n_sub, V1), gen)
        ref, ref_sub = seq_codes(h, c0, n1)
        _, sd = blm.chain(h, c0.to(d), n1.reshape(B, -1).to(d), ref.to(d))
        sd = sd.cpu()
        maxd, meand = max(maxd, float((sd - ref_sub).abs().max())), max(meand, float((sd - ref_sub).abs().mean()))
        got = {}
        for corrupt in (False, True):
            if corrupt:
                cg = torch.Generator().manual_seed(100 + trial)
                blm.inject = lambda B_, ver: [(r, int(torch.randint(1, 16, (1,), generator=cg)), int(torch.randint(0, V1, (1,), generator=cg)))
                                              for r in ver.nonzero()[:, 0].tolist()]
            blm.phase.clear()
            got[corrupt] = blm._spec_codes(B, h, c0, n1, torch.full((B, n_sub), -1, dtype=torch.long), torch.ones(B, dtype=torch.bool))[0]
            blm.inject = None
        tot += got[True].numel()
        diff += int((got[True] != got[False]).sum())
        excused += int(((got[False] != ref).cumsum(1) > 0).any(1).sum())
        beat()
    drain()
    log(f"(d2) corrupted proposals + fix-up vs clean speculative codes at B=32 (same noise): {tot} codes, {diff} differ; "
        f"diagnostic vs the sequential PredFirst/PredNext path (other op order): forced logits max |diff| {maxd:.3f}, worst mean {meand:.2e}, "
        f"rows with any differing code {excused}/{D_TRIALS * B}")
    if diff:
        fails.append(f"(d2) {diff} codes differ between the corrupted and the clean speculative pass")

    # ------------------------------------------------------------------ (e) contention
    burn = ("import torch, time\ntorch.set_num_threads(3)\na = torch.randn(1024, 1024)\n"
            "t = time.time()\nwhile time.time() - t < 600:\n    a = (a @ a).tanh_()\n")
    procs = [subprocess.Popen([sys.executable, "-c", burn]) for _ in range(2)]
    time.sleep(3)
    e64 = run("(e) spec under 2x3-thread CPU burners", 64, STEPS // 2, spec=True)
    e64t = run("(e) spec + speculative talker under CPU burners", 64, STEPS // 2, spec=True, spec_talker=True)
    e32 = run("(e) spec under CPU burners", 32, STEPS // 2, spec=True)
    for p in procs:
        p.kill()

errs = [e for e in events if e[1] == "error"]
if errs:
    fails.append(f"error events: {errs[:4]}")

best32 = min(c32["med"], c32t["med"])
best64 = min(c64["med"], c64t["med"])
beste = min(e64["med"], e64t["med"])
kill = []
if best32 > 46:
    kill.append(f"B=32 median {best32:.1f} > 46")
if best64 > 72:
    kill.append(f"B=64 median {best64:.1f} > 72")
if beste > 76:
    kill.append(f"B=64 under load {beste:.1f} > 76")
if c64["passes"] > 0.5:
    kill.append(f"fix-up passes {c64['passes']:.3f}/step at B=64 > 0.5")
if fails:
    kill.append("exactness")
log(f"[summary] (a) control B=32 {a['med']:.1f} ms; (b) chain " + ", ".join(f"B={b} {res[f'chain{b}']:.1f}" for b in CHAIN_B) +
    f" ms; (c) B=32 {c32['med']:.1f} (spec talker {c32t['med']:.1f}) p99 {c32['p99']:.1f}, B=64 {c64['med']:.1f} (spec talker {c64t['med']:.1f}) "
    f"p99 {c64['p99']:.1f} ms, passes/step B=32 {c32['passes']:.3f} B=64 {c64['passes']:.3f}, mismatch/code {c64['mism_rate']:.2e}; "
    f"(e) B=64 {e64['med']:.1f} (spec talker {e64t['med']:.1f}), B=32 {e32['med']:.1f} ms; exactness {'OK' if not fails else 'FAILED'}; "
    f"kill line: {'TRIPPED: ' + '; '.join(kill) if kill else 'PASS'}")
for f in fails:
    log(f"FAILED: {f}")
sys.exit(1 if fails else 0)
