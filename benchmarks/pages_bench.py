"""Paged KV rows benchmark (needs one NeuronCore; run inside the server image).

Paged rows (TalkerInP: a stream past the 512-slot batch cache borrows free rows of the batch as
extra KV pages) against TalkerIn and against the B=1 2048-slot path:

(a) baseline: TalkerIn graph alone at B=64 and B=32 (synced, median of ITERS calls, two rounds);
(b) TalkerInP ("gather" form, P=1 partner) at B=64 with 0, 1, 2 groups (leaders at 700, 980), at
    B=32 with 1 group; P=3 at B=64 with 1 group at 1400 (2 pages in use + 1 fresh); the "mm" form
    (host-built [B, B] 0/1 matrices instead of index_select) at B=64 with 0 and 2 groups;
(c) teacher-forced long rows: one stream generated on the B=1 path (FrameStep, 2048 slots),
    snapshots copied into a B=64 batch cache (leader row + page rows, garbage in every slot that
    is not valid); NT frames with the B=1 path's codes forced through the B=64 chain and
    TalkerInP; |logprob diff| vs the B=1 path for a baseline row (position 110, no group) and long
    rows: P=1: one crossing 511 during the run (its page starts fresh, valid 0) and one at 980;
    P=3: one crossing 1022 (second page fresh) and one at 1400 (third page unused, valid 0);
(d) TalkerInP B=64 compile time.

The final "kill lines" are the acceptance limits the design was held to: (b, B=64, 2 groups) <=
(a, B=64) + 4 ms; (c) all finite, long-row mean |d| <= baseline + 0.02, max <= 0.5; (d) <= 600 s.
Exit 3 on a stall.

    NEURON_RT_VISIBLE_CORES=0 python benchmarks/pages_bench.py      (env: ITERS=50 NT=26; CPU smoke: QWEN3TTS_DEVICE=cpu LR_S=64 LR_B=8 ...)
"""
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import server  # noqa: E402

server.setup_env()  # compiler caches under $VIBESYS_CACHE_DIR, cwd in a fresh temp dir

import torch  # noqa: E402

from engine.model import BF, PredChain, TalkerIn, TalkerInP, page_inputs  # noqa: E402
from engine.runtime import LM, TALKER_SLOTS  # noqa: E402

ITERS = int(os.environ.get("ITERS", "50"))
S = int(os.environ.get("LR_S", "512"))
SP = S - 1  # page size; page_at = SP (the serving split)
B = int(os.environ.get("LR_B", "64"))
B2 = int(os.environ.get("LR_B2", str(B // 2)))
NT = int(os.environ.get("NT", "26"))
CTRL = int(os.environ.get("LR_CTRL", "110"))
# (c) start positions: P=1 crossing, P=1 inside page 1, P=3 crossing page 1->2, P=3 inside page 2
STARTS = tuple(int(x) for x in os.environ.get("LR_STARTS", f"{SP - 6},{SP + 469},{2 * SP - 12},{2 * SP + 378}").split(","))
TPOS = tuple(int(x) for x in os.environ.get("LR_TPOS", "700,980,1400").split(","))  # (b) leader positions
MM = os.environ.get("MM", "1") == "1"
HANG_S = float(os.environ.get("HANG_S", "300"))
WARM_HANG_S = float(os.environ.get("WARM_HANG_S", "1500"))
T0 = time.time()
last = [time.time()]
limit = [WARM_HANG_S]
TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a lantern, a thermos of "
        "strong tea and a worn notebook in which he recorded the color of the sky, the direction of the wind, the "
        "ships that passed along the horizon and the names of the birds that rested on the railing before they "
        "continued their long journey south across the gray and restless sea.")


def log(m):
    print(f"[pg {time.time() - T0:7.1f}s] {m}", flush=True)


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


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2]


device = os.environ.get("QWEN3TTS_DEVICE") or server.pick_device()
compile_ = device.startswith("neuron")
torch.set_num_threads(int(os.environ.get("QWEN3TTS_LM_THREADS", "2" if compile_ else "8")))
weights = os.environ.get("QWEN3TTS_WEIGHTS", "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
beat("loading the checkpoint")
lm = LM(weights, log=log)
lm.to_device(device, compile_=compile_)
d = lm.device
fs = lm.frame_mod
G, H, V1 = fs.n_sub + 1, fs.H, fs.V1
if compile_:
    from torch import _dynamo

    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
    torch._logging.set_logs(recompiles=True)


def comp(m):
    m = m.to(d).eval()
    return torch.compile(m, backend="neuron", fullgraph=True, dynamic=False) if compile_ else m


talkerin = comp(TalkerIn(fs))
chain = comp(PredChain(fs))
tp_g1 = comp(TalkerInP(fs, "gather"))  # P=1 and P=3 are separate graphs (shapes) of this module
tp_mm = comp(TalkerInP(fs, "mm")) if MM else None
t = fs.talker
L, kvh, hd = t.L, t.kvh, t.hd
kill, notes, compile_s = [], [], {}

with torch.inference_mode():  # every graph input below is made in inference mode (avoids dispatch-key recompiles)
    caches = {b: (torch.zeros((L, b, kvh, S, hd), dtype=BF, device=d), torch.zeros((L, b, kvh, S, hd), dtype=BF, device=d))
              for b in (B, B2)}

    def dev_inputs(b, groups, P, mode, pos=None):
        if pos is None:
            pos = [10 + (r * 7) % (S - 20) for r in range(b)]  # normal rows: positions < S - 1
            for ld, p in groups.items():
                pos[ld] = p[0]
            groups = {ld: p[1] for ld, p in groups.items()}
        return tuple(x.to(d) for x in page_inputs(pos, groups, P=P, mode=mode, page_at=SP, S=S))

    zc = {b: torch.zeros((b, G), dtype=torch.long, device=d) for b in (B, B2)}
    zt = {b: torch.zeros((b, H), dtype=torch.float32, device=d) for b in (B, B2)}
    zp = {b: torch.tensor([10 + (r * 7) % (S - 20) for r in range(b)], dtype=torch.long).to(d) for b in (B, B2)}

    def call_ti(b):
        return talkerin(zc[b], zt[b], zp[b], *caches[b])

    def call_tp(mod, b, ins):
        return mod(zc[b], zt[b], *ins, *caches[b])

    # ------------------------------------------------------------------ compile / load
    def warm(name, fn):
        limit[0] = WARM_HANG_S
        beat(f"[warm] {name}")
        t0 = time.time()
        out = fn()
        out[1].cpu() if isinstance(out, tuple) else out.cpu()
        compile_s[name] = time.time() - t0
        log(f"[warm] {name}: {compile_s[name]:.1f}s")

    for b in (B, B2):
        warm(f"TalkerIn B={b}", lambda b=b: call_ti(b))
    warm(f"chain B={B}", lambda: chain(torch.zeros((B, H), dtype=BF, device=d), torch.zeros(B, dtype=torch.long, device=d),
                                       torch.zeros((B, (G - 1) * V1), device=d), torch.full((B, G - 1), -1, dtype=torch.long, device=d)))

    def warm_b1():
        cache = lm.new_talker_cache()
        st = lm.frame_state(10)
        hin = lm.host_input(lm.emb.prompt("Warm up.", "Ryan", "English")[1], 0, torch.Generator().manual_seed(0))
        return lm.frame_step(torch.zeros((1, H), dtype=BF, device=d), torch.zeros((1, fs.V0), dtype=torch.float32, device=d), cache, st, hin)[0]

    warm("FrameStep B=1 S=2048", warm_b1)
    IN = {  # (b) inputs, built once
        "g P=1 B=64 0 groups": (tp_g1, B, dev_inputs(B, {}, 1, "gather")),
        "g P=1 B=64 1 group": (tp_g1, B, dev_inputs(B, {0: (TPOS[0], [B - 1])}, 1, "gather")),
        "g P=1 B=64 2 groups": (tp_g1, B, dev_inputs(B, {0: (TPOS[0], [B - 1]), 1: (TPOS[1], [B - 2])}, 1, "gather")),
        f"g P=1 B={B2} 1 group": (tp_g1, B2, dev_inputs(B2, {0: (TPOS[0], [B2 - 1])}, 1, "gather")),
        "g P=3 B=64 1 group (1400)": (tp_g1, B, dev_inputs(B, {0: (TPOS[2], [B - 1, B - 2, B - 3])}, 3, "gather")),
    }
    if MM:
        IN["mm P=1 B=64 0 groups"] = (tp_mm, B, dev_inputs(B, {}, 1, "mm"))
        IN["mm P=1 B=64 2 groups"] = (tp_mm, B, dev_inputs(B, {0: (TPOS[0], [B - 1]), 1: (TPOS[1], [B - 2])}, 1, "mm"))
    for name in ("g P=1 B=64 0 groups", f"g P=1 B={B2} 1 group", "g P=3 B=64 1 group (1400)", "mm P=1 B=64 0 groups"):
        if name in IN:
            mod, b, ins = IN[name]
            warm(f"TalkerInP {name.split(' 0 ')[0].split(' 1 ')[0]}", lambda mod=mod, b=b, ins=ins: call_tp(mod, b, ins))
    tp_compile = compile_s.get("TalkerInP g P=1 B=64", float("nan"))
    if tp_compile > 600:
        kill.append(f"(d) TalkerInP B=64 compile {tp_compile:.0f}s > 600")
    limit[0] = HANG_S
    beat("warm-up done")

    # ------------------------------------------------------------------ (c) B=1 reference stream
    def lps_of(l_in, sub, codes):
        l0 = torch.log_softmax(l_in.float(), -1)
        ls = torch.log_softmax(sub.float(), -1)
        return [float(l0[codes[0]])] + [float(ls[g - 1, codes[g]]) for g in range(1, 16)]

    cache, h, logits, trailing, n_ids, pos0 = lm.prefill(TEXT, "Ryan", "English")
    state = lm.frame_state(pos0)
    gen = torch.Generator().manual_seed(27)
    starts = (CTRL,) + STARTS
    snap_f = {p - pos0 for p in starts}
    last_f = max(starts) - pos0 + NT
    assert pos0 + last_f < TALKER_SLOTS and last_f < TALKER_SLOTS - n_ids, (pos0, last_f, n_ids)
    snaps, ref, b1_ms = {}, {}, []
    for f in range(last_f):
        if f in snap_f:
            snaps[f] = (cache[0].cpu(), cache[1].cpu(), h.cpu(), logits.cpu())
        win = any(p - pos0 <= f < p - pos0 + NT for p in starts)
        l_in = logits.cpu() if win else None
        hin = lm.host_input(trailing, f, gen)
        t0 = time.perf_counter()
        codes, h, logits, sub = lm.frame_step(h, logits, cache, state, hin)
        c = codes[0].cpu().tolist()
        b1_ms.append(1000 * (time.perf_counter() - t0))
        if win:
            ref[f] = (c, lps_of(l_in[0], sub[0].cpu(), c))
        if f % 100 == 0:
            beat()
    log(f"(c) B=1 reference stream: {last_f} frames from position {pos0}, FrameStep B=1 launch+sync median {med(b1_ms):.2f} ms")
    del cache

    gcpu = torch.Generator().manual_seed(5)

    def run_c(mod, mode, P, rows):
        """rows: {batch row: (start position, [page rows])}. Returns {row: [NT, 16] |d|}."""
        Kh = torch.randn((L, B, kvh, S, hd), generator=gcpu).to(BF)  # garbage in every slot that is not valid
        Vh = torch.randn((L, B, kvh, S, hd), generator=gcpu).to(BF)
        hh = torch.zeros((B, H), dtype=BF)
        lh = {}
        for r, (p0, pages) in rows.items():
            k1, v1, h1, l1 = snaps[p0 - pos0]
            n = min(p0, SP)
            Kh[:, r, :, :n] = k1[:, 0, :, :n]
            Vh[:, r, :, :n] = v1[:, 0, :, :n]
            for j, pr in enumerate(pages):
                base = SP + j * SP
                n = min(max(p0 - base, 0), SP)
                Kh[:, pr, :, :n] = k1[:, 0, :, base : base + n]
                Vh[:, pr, :, :n] = v1[:, 0, :, base : base + n]
            hh[r] = h1[0]
            lh[r] = l1[0]
        caches[B] = None
        K, V = Kh.to(d), Vh.to(d)  # (made in inference mode, as every graph input)
        caches[B] = (K, V)
        del Kh, Vh
        hd_ = hh.to(d)
        diffs = {r: [] for r in rows}
        ngroups = []
        for tt in range(NT):
            codes = torch.zeros((B, G), dtype=torch.long)
            forced = torch.full((B, G - 1), -1, dtype=torch.long)
            trail = torch.zeros((B, H), dtype=torch.float32)
            pos = [0] * B
            groups = {}
            for r, (p0, pages) in rows.items():
                f = p0 - pos0 + tt
                c = ref[f][0]
                codes[r] = torch.tensor(c)
                forced[r] = torch.tensor(c[1:])
                trail[r] = trailing[f] if f < trailing.shape[0] else lm.emb.tts_pad
                pos[r] = p0 + tt
                need = 0 if pos[r] < SP else 1 + (pos[r] - SP) // SP
                if r != CROW and need:  # pages join when the stream needs them (P=3, 1400: all 3, the last one unused)
                    groups[r] = pages if P == 3 and p0 > 2 * SP else pages[:need]
            ngroups.append(len(groups))
            cd, sd = chain(hd_, codes[:, 0].contiguous().to(d), torch.zeros((B, (G - 1) * V1), device=d), forced.to(d))
            sdh = sd.cpu()
            for r, (p0, _) in rows.items():
                f = p0 - pos0 + tt
                got = lps_of(lh[r], sdh[r], ref[f][0])
                diffs[r].append([abs(a - b) for a, b in zip(got, ref[f][1])])
            ins = tuple(x.to(d) for x in page_inputs(pos, groups, P=P, mode=mode, page_at=SP, S=S))
            tc = time.time()
            hd_, l2 = mod(codes.to(d), trail.to(d), *ins, K, V)
            l2h = l2.cpu()
            if time.time() - tc > 5:
                log(f"(c) TalkerInP call {tt} took {time.time() - tc:.1f}s (a recompile?)")
            for r in rows:
                lh[r] = l2h[r]
            beat()
        return diffs, ngroups

    CROW = B // 6
    results_c = {}
    R1, R2 = 3, B * 5 // 8
    p1rows = {CROW: (CTRL, []), R1: (STARTS[0], [B - 1]), R2: (STARTS[1], [B - 2])}
    cases = [("gather P=1", tp_g1, "gather", 1, p1rows),
             ("gather P=3", tp_g1, "gather", 3, {CROW: (CTRL, []), R1: (STARTS[2], [B - 1, B - 2]), R2: (STARTS[3], [B - 3, B - 4, B - 5])})]
    if MM:
        cases.append(("mm P=1", tp_mm, "mm", 1, p1rows))
    for name, mod, mode, P, rows in cases:
        if mode == "mm" and P == 3:
            continue
        diffs, ng = run_c(mod, mode, P, rows)
        n_nan, longs = 0, []
        ctrl = None
        for r, (p0, pages) in rows.items():
            dd = torch.tensor(diffs[r])
            bad = ~dd.isfinite()
            n_nan += int(bad.sum())
            fin = dd[~bad]
            mean_, max_ = (float(fin.mean()), float(fin.max())) if fin.numel() else (float("nan"), float("nan"))
            c0d = dd[1:, 0]
            log(f"(c) {name}: row {r} positions {p0}..{p0 + NT - 1} ({'control' if r == CROW else 'long, pages ' + str(pages)}): "
                f"{dd.numel()} logprobs, non-finite {int(bad.sum())}, mean |d| {mean_:.4f} max {max_:.4f}; code 0 after a TalkerInP step "
                f"mean {float(c0d[c0d.isfinite()].mean()):.4f} max {float(c0d[c0d.isfinite()].max()):.4f}; per-frame mean "
                f"{[round(float(x), 3) for x in dd.nan_to_num(9.0).mean(1)]}")
            if r == CROW:
                ctrl = mean_
            else:
                longs.append(fin.flatten())
        lo = torch.cat(longs)
        c_mean, c_max = float(lo.mean()), float(lo.max())
        log(f"(c) {name}: groups per step {ng}; long rows mean |d| {c_mean:.4f} max {c_max:.4f}, control mean {ctrl:.4f}, non-finite {n_nan}")
        results_c[name] = (c_mean, c_max, ctrl, n_nan)
        if n_nan:
            kill.append(f"(c) {name}: {n_nan} non-finite logprob differences")
        if not (c_mean <= ctrl + 0.02 and c_max <= 0.5):
            kill.append(f"(c) {name}: long rows mean {c_mean:.4f} (control {ctrl:.4f} + 0.02) max {c_max:.4f} (limit 0.5)")

    # ------------------------------------------------------------------ (a), (b) graphs alone
    def time_fn(fn):
        for _ in range(5):
            fn()[1].cpu()
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            fn()[1].cpu()
            ts.append(1000 * (time.perf_counter() - t0))
            beat()
        return med(ts)

    rounds = []
    for rnd in range(2):
        res = {f"TalkerIn B={B}": time_fn(lambda: call_ti(B)), f"TalkerIn B={B2}": time_fn(lambda: call_ti(B2))}
        for name, (mod, b, ins) in IN.items():
            res[name] = time_fn(lambda mod=mod, b=b, ins=ins: call_tp(mod, b, ins))
        log(f"(a/b) round {rnd}: " + ", ".join(f"{k} {v:.2f}" for k, v in res.items()) + " ms (graph alone, synced, median)")
        rounds.append(res)
    best = {k: min(r[k] for r in rounds) for k in rounds[0]}
    a64, a32 = best[f"TalkerIn B={B}"], best[f"TalkerIn B={B2}"]
    for k, v in best.items():
        ref_ = a32 if f"B={B2}" in k and B2 != B else a64
        log(f"(a/b) {k}: {v:.2f} ms ({v - ref_:+.2f} vs TalkerIn at the same B)")
    g2 = best["g P=1 B=64 2 groups"] - a64
    m2 = best["mm P=1 B=64 2 groups"] - a64 if MM else float("inf")
    if min(g2, m2) > 4:
        kill.append(f"(b) B=64 2 groups: gather {g2:+.2f}, mm {m2:+.2f} ms vs TalkerIn {a64:.2f} (limit +4)")
    elif g2 > 4:
        notes.append(f"gather form {g2:+.2f} > +4 ms; mm form {m2:+.2f} passes")
    g4 = best["g P=3 B=64 1 group (1400)"] - a64
    if g4 > 6:
        notes.append(f"P=3 {g4:+.2f} ms > +6: ship P=1 with the B=1 fallback above {2 * SP} positions")

log(f"[summary] (a) TalkerIn B={B} {a64:.2f} ms, B={B2} {a32:.2f} ms; (b) gather P=1 B=64 0/1/2 groups "
    f"{best['g P=1 B=64 0 groups'] - a64:+.2f}/{best['g P=1 B=64 1 group'] - a64:+.2f}/{g2:+.2f} ms, B={B2} 1 group "
    f"{best[f'g P=1 B={B2} 1 group'] - a32:+.2f}, P=3 1 group {g4:+.2f}"
    + (f", mm 0/2 groups {best['mm P=1 B=64 0 groups'] - a64:+.2f}/{m2:+.2f}" if MM else "")
    + f"; (c) {', '.join(f'{k}: long mean {v[0]:.4f} max {v[1]:.4f} control {v[2]:.4f} non-finite {v[3]}' for k, v in results_c.items())}; "
    f"(d) compiles {', '.join(f'{k} {v:.0f}s' for k, v in compile_s.items())}; notes {notes}; kill lines: "
    f"{'TRIPPED: ' + '; '.join(kill) if kill else 'PASS'}")
sys.exit(0)
