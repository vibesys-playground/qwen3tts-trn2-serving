"""The scored benchmark: VoxServe v2's goodput sweep (arXiv 2602.00269v2, section 4 and Appendix B).

    python bench.py --output-json FILE      (vibesys-cli appends the flag; the server is at
                                             VIBESYS_SERVICE_URL)

Workload: the normalized text of the 4,837 LibriTTS test-clean utterances
(data/libritts_test_clean.jsonl), speaker Ryan, English, the checkpoint's sampling defaults.

For each arrival seed s in 0, 1, 2 (one sweep each, back to back against the same server):
  - a discarded burst of 64 concurrent requests, then four warm-up requests that are not counted;
  - offered rates 10, 12, 14, ... req/s; each rate is a 60 s arrival window of Poisson arrivals
    with a fixed seed, kept on an absolute schedule (a slow server does not lower the offered
    load), after which the client waits until every request finishes or times out (900 s);
  - the seed draws its own order of the utterances and its own arrival times;
  - the sweep stops after a rate at which fewer than half of the requests receive audio within
    1 s (the paper's rule), or after PLATEAU_RATES (3) consecutive rates none of which sets a new
    best SLO-met count for this seed, whichever comes first.

Deviation from the paper's protocol (which starts at 2 req/s and has only the TTFA rule): the
start at 10 req/s and the plateau stop cut measurement time; approved by the user on 2026-11-24.
Rates below 10 req/s are far under this system's capacity (goodput ~22 req/s), so they never set
the max. Each rate keeps its own arrival seed, so the arrivals at any rate are the same as before.

A request meets the SLO if it completes (HTTP 200, the stream ends cleanly, some audio), its
TTFA (send to first audio byte, HTTP and queueing included) is at most 200 ms, and it does not
stall (a player starting at the first byte pauses less than 10 ms in total). Rejected and failed
requests are misses.

Score: `goodput` (req/s, max) = max over offered rates r of
    (requests meeting the SLO at rate r, summed over the three seeds) / (3 x 60 s);
a seed whose sweep stopped below r contributes no requests at r. Also reported (diagnostic):
SLO capacity at 90% and 99% pooled attainment (the highest rate at which attainment stays at
or above the threshold at that rate and every lower one; since the sweep starts at 10, these
capacity diagnostics consider rates from 10 req/s up), TTFA percentiles and per-rate tables.

Integrity guard: across the counted requests, audio seconds per text character must be at least
MIN_AUDIO_RATIO of the reference's (golden/sampled_reference.json); a server that speaks less of
the text than the model does fails the sample instead of scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402

HERE = Path(__file__).parent
SEEDS = (0, 1, 2)
START_RATE = 10.0  # deviation from the paper's 2 req/s, approved by the user on 2026-11-24
RATE_STEP = 2.0
PLATEAU_RATES = 3  # stop a seed after this many consecutive rates without a new best slo_met
WINDOW_S = 60.0
WARMUP_BURST = 64
WARMUP_SEQUENTIAL = 4
MAX_RATE = 400.0  # a guard against a runaway sweep; far above any single chip
STOP_TTFA_S = 1.0
MIN_AUDIO_RATIO = 0.75


def texts() -> list[str]:
    return [json.loads(line)["text"] for line in (HERE / "data" / "libritts_test_clean.jsonl").read_text().splitlines()]


def log(*a) -> None:
    print(*a, flush=True)


async def run_rate(url: str, rate: float, order: list[str], cursor: int, rng: np.random.Generator):
    """One 60 s window at `rate`, drained. Returns the results and the advanced text cursor."""
    gaps = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= WINDOW_S:
            break
        gaps.append(t)
    results = []
    async with client.session() as sess:
        start = time.monotonic() + 0.05
        tasks = []
        for off in gaps:
            res = client.Result(text=order[cursor % len(order)], sent=start + off)
            cursor += 1
            results.append(res)
            delay = res.sent - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(client.request(sess, url, res)))
        deadline = start + WINDOW_S + client.REQUEST_TIMEOUT_S
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline - time.monotonic()))
            for p in pending:
                p.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    return results, cursor


async def warmup(url: str, order: list[str]) -> None:
    async with client.session() as sess:
        burst = [client.Result(text=order[i % len(order)], sent=time.monotonic()) for i in range(WARMUP_BURST)]
        await asyncio.gather(*(client.request(sess, url, r) for r in burst))
        for i in range(WARMUP_SEQUENTIAL):
            r = client.Result(text=order[(WARMUP_BURST + i) % len(order)], sent=time.monotonic())
            await client.request(sess, url, r)
    failed = sum(1 for r in burst if not r.complete)
    log(f"  warm-up: burst of {WARMUP_BURST} ({failed} incomplete), {WARMUP_SEQUENTIAL} sequential")


def rate_stats(results: list[client.Result]) -> dict:
    n = len(results)
    ttfas = sorted(r.ttfa for r in results if r.ttfa is not None)
    return {
        "requests": n,
        "slo_met": sum(r.meets_slo for r in results),
        "completed": sum(r.complete for r in results),
        "audio_within_1s": sum(1 for r in results if r.ttfa is not None and r.ttfa <= STOP_TTFA_S),
        "stalled": sum(1 for r in results if r.complete and r.pause >= client.STALL_SLO_S),
        "ttfa_p50": float(np.percentile(ttfas, 50)) if ttfas else None,
        "ttfa_p99": float(np.percentile(ttfas, 99)) if ttfas else None,
    }


async def sweep(url: str, seed: int, corpus: list[str], artifacts: Path | None) -> dict[float, dict]:
    order = list(corpus)
    random.Random(seed).shuffle(order)
    log(f"seed {seed}:")
    await warmup(url, order)
    cursor = WARMUP_BURST + WARMUP_SEQUENTIAL
    out: dict[float, dict] = {}
    rate = START_RATE
    best = -1
    since_best = 0
    while rate <= MAX_RATE:
        rng = np.random.default_rng([seed, int(round(rate * 1000))])
        t0 = time.monotonic()
        results, cursor = await run_rate(url, rate, order, cursor, rng)
        st = rate_stats(results)
        st["audio_s"] = sum(r.audio_seconds for r in results if r.complete)
        st["chars"] = sum(len(r.text) for r in results if r.complete)
        out[rate] = st
        log(
            f"  rate {rate:g}: {st['requests']} req, {st['slo_met']} met SLO, {st['completed']} completed, "
            f"{st['audio_within_1s']} audio<=1s, ttfa p50 {st['ttfa_p50']}, {time.monotonic() - t0:.0f}s"
        )
        if artifacts:
            with open(artifacts / f"requests-seed{seed}.jsonl", "a") as f:
                for r in results:
                    f.write(json.dumps({"rate": rate, **r.summary()}) + "\n")
        if st["slo_met"] > best:
            best, since_best = st["slo_met"], 0
        else:
            since_best += 1
        if st["audio_within_1s"] < st["requests"] / 2:
            log(f"  seed {seed} stops at rate {rate:g}: fewer than half the requests got audio within {STOP_TTFA_S:g} s")
            break
        if since_best >= PLATEAU_RATES:
            log(f"  seed {seed} stops at rate {rate:g}: plateau, {PLATEAU_RATES} rates without a new best ({best} met SLO)")
            break
        rate += RATE_STEP
    return out


def capacity(pooled: dict[float, dict], threshold: float) -> float:
    cap = 0.0
    for rate in sorted(pooled):
        p = pooled[rate]
        if p["requests"] == 0 or p["slo_met"] / p["requests"] < threshold:
            break
        cap = rate
    return cap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--url", default=os.environ.get("VIBESYS_SERVICE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="for local tries only; the score uses 0,1,2")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    seeds = [int(s) for s in args.seeds.split(",")]
    art = os.environ.get("VIBESYS_ARTIFACTS_DIR")
    artifacts = Path(art) if art else None
    corpus = texts()
    assert len(corpus) == 4837, f"expected the 4,837 LibriTTS test-clean utterances, got {len(corpus)}"

    t0 = time.monotonic()
    per_seed = {s: asyncio.run(sweep(url, s, corpus, artifacts)) for s in seeds}
    rates = sorted({r for v in per_seed.values() for r in v})
    pooled = {}
    for r in rates:
        keys = ("requests", "slo_met", "completed", "audio_within_1s", "stalled", "audio_s", "chars")
        pooled[r] = {k: sum(per_seed[s][r][k] for s in seeds if r in per_seed[s]) for k in keys}
        pooled[r]["goodput"] = pooled[r]["slo_met"] / (len(seeds) * WINDOW_S)
    goodput = max((p["goodput"] for p in pooled.values()), default=0.0)
    peak_rate = max(pooled, key=lambda r: pooled[r]["goodput"]) if pooled else None

    audio_s = sum(p["audio_s"] for p in pooled.values())
    chars = sum(p["chars"] for p in pooled.values())
    ref = json.loads((HERE / "golden" / "sampled_reference.json").read_text())
    ref_rate = ref["audio_seconds_per_char"]
    ratio = (audio_s / chars) / ref_rate if chars else None
    report = {
        "goodput": goodput,
        "peak_rate": peak_rate,
        "slo_capacity_90": capacity(pooled, 0.90),
        "slo_capacity_99": capacity(pooled, 0.99),
        "audio_per_char_ratio": ratio,
        "seeds": seeds,
        "pooled": {f"{r:g}": p for r, p in pooled.items()},
        "per_seed": {str(s): {f"{r:g}": v for r, v in d.items()} for s, d in per_seed.items()},
        "wall_s": time.monotonic() - t0,
    }
    log(json.dumps({k: report[k] for k in ("goodput", "peak_rate", "slo_capacity_90", "slo_capacity_99", "audio_per_char_ratio", "wall_s")}))
    if ratio is not None and ratio < MIN_AUDIO_RATIO:
        log(f"FAIL: the server produced {ratio:.2f}x the reference's audio per character (minimum {MIN_AUDIO_RATIO})")
        return 1
    if artifacts:
        (artifacts / "goodput.json").write_text(json.dumps(report, indent=1))
    Path(args.output_json).write_text(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
