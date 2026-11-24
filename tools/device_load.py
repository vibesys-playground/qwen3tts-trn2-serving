"""Device load test (needs a Trainium2 instance; run inside the server image): start the server as
the check and benchmark expect it (`python -m engine.server`, all visible NeuronCores), then

1. /score on the ten golden utterances at once (the check's scoring gate numbers),
2. a few idle /generate requests one at a time (unloaded TTFA),
3. bursts of N concurrent /generate requests (LibriTTS texts, a fixed shuffle): per burst the
   completed count, TTFA percentiles, the player's pauses, audio seconds per character, and that
   every body is whole frames (1920 samples each).

The server's own log (its replicas' startup self-tests, [timing] per bucket and [batch] step
statistics under load) is printed in line. Exits non-zero when a request fails.

    python tools/device_load.py [--bursts 32,64] [--idle 3] [--poisson 20:75,24:75 --seed 0]

--poisson RATE:SECONDS,...: after the bursts, open-loop Poisson arrivals at each rate for that
long (texts drawn at random from the LibriTTS list, an absolute schedule), then wait for every
request; per rate: sent, 200s, rejections (non-200), SLO met (200, whole frames, TTFA <= 200 ms,
player pauses < 10 ms), TTFA percentiles, and the wall-clock span (to match server log lines).
"""
import argparse, http.client, json, os, random, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "eval"
T0 = time.time()


def log(m):
    print(f"[load {time.time() - T0:7.1f}s] {m}", flush=True)


def post(port, path, body, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    return c, c.getresponse()


def gen(port, text, out):
    t = time.monotonic()
    r = {"chars": len(text), "ttfa": None, "bytes": 0, "pause": 0.0, "ok": False}
    try:
        c, resp = post(port, "/generate", {"text": text, "speaker": "Ryan", "language": "English", "stream": True})
        r["status"] = resp.status
        play_end = 0.0
        while True:
            b = resp.read1(65536)
            now = time.monotonic()
            if not b:
                break
            if r["ttfa"] is None:
                r["ttfa"] = now - t
                play_end = now
            elif now > play_end:
                r["pause"] += now - play_end
                play_end = now
            play_end += len(b) / 48000
            r["bytes"] += len(b)
        r["ok"] = resp.status == 200 and r["bytes"] > 0 and r["bytes"] % 3840 == 0
        c.close()
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
    r["total"] = time.monotonic() - t
    out.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", default="32,64")
    ap.add_argument("--idle", type=int, default=3)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--poisson", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    env = dict(os.environ)
    srv = subprocess.Popen([sys.executable, "-m", "engine.server", "--host", "127.0.0.1", "--port", str(a.port)], cwd=ROOT, env=env)
    fails = []
    try:
        while True:
            if srv.poll() is not None:
                log(f"server exited with {srv.returncode} during startup")
                sys.exit(2)
            try:
                c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=5)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            time.sleep(2)
        log(f"server healthy after {time.time() - T0:.0f}s")

        G = json.loads((TASK / "golden/golden.json").read_text())
        res = [None] * len(G["cases"])

        def score(i, cs):
            c, resp = post(a.port, "/score", {"text": cs["text"], "speaker": "Ryan", "language": "English", "codes": cs["codes"]})
            res[i] = json.loads(resp.read())

        t = time.time()
        ths = [threading.Thread(target=score, args=(i, cs)) for i, cs in enumerate(G["cases"])]
        [x.start() for x in ths]
        [x.join() for x in ths]
        d = np.concatenate([np.abs(np.array(r["logprobs"]) - np.array(cs["logprobs"])).ravel() for r, cs in zip(res, G["cases"])])
        eos = max(abs(r["eos_logprob"] - cs["eos_logprob"]) for r, cs in zip(res, G["cases"]))
        ok = d.mean() <= 0.08 and np.quantile(d, 0.99) <= 0.40 and d.max() <= 2.0 and eos <= 0.5
        log(f"SCORE 10 at once in {time.time() - t:.1f}s: mean {d.mean():.4f} p99 {np.quantile(d, .99):.4f} max {d.max():.4f} eos {eos:.4f} -> {'pass' if ok else 'FAIL'}")
        if not ok:
            fails.append("score gate")

        texts = [json.loads(x)["text"] for x in (TASK / "data/libritts_test_clean.jsonl").read_text().splitlines()]
        random.Random(5).shuffle(texts)
        k = 0
        for _ in range(a.idle):
            out = []
            gen(a.port, texts[k], out)
            k += 1
            r = out[0]
            log(f"IDLE {r['chars']} chars: ttfa {1000 * (r['ttfa'] or 0):.0f} ms, {r['bytes'] / 48000:.2f}s audio in {r['total']:.2f}s, pause {1000 * r['pause']:.0f} ms, ok {r['ok']}")
            if not r["ok"]:
                fails.append(f"idle request {r}")
        for n in [int(x) for x in a.bursts.split(",") if x]:
            out = []
            ths = [threading.Thread(target=gen, args=(a.port, texts[k + i], out)) for i in range(n)]
            k += n
            t = time.time()
            [x.start() for x in ths]
            [x.join() for x in ths]
            wall = time.time() - t
            okn = sum(r["ok"] for r in out)
            tt = np.array([r["ttfa"] for r in out if r["ttfa"] is not None]) * 1000
            audio = sum(r["bytes"] for r in out) / 48000
            spc = [r["bytes"] / 48000 / r["chars"] for r in out if r["ok"]]
            log(f"BURST {n} concurrent: {okn}/{n} ok in {wall:.1f}s; ttfa ms p50 {np.median(tt):.0f} p90 {np.quantile(tt, .9):.0f} max {tt.max():.0f}; "
                f"pause s p50 {np.median([r['pause'] for r in out]):.2f} max {max(r['pause'] for r in out):.2f}; "
                f"{audio:.1f}s audio ({audio / wall:.1f}x real time, {audio * 12.5 / wall:.0f} frames/s); s/char min {min(spc):.3f} median {np.median(spc):.3f} max {max(spc):.3f}")
            bad = [r for r in out if not r["ok"]]
            if bad:
                fails.append(f"burst {n}: {len(bad)} failed: {bad[:3]}")
        for spec in [x for x in a.poisson.split(",") if x]:
            rate, dur = (float(v) for v in spec.split(":"))
            rng = random.Random(a.seed * 1000 + int(rate))
            out, ths = [], []
            t0 = time.monotonic()
            w0 = time.strftime("%H:%M:%S")
            at = rng.expovariate(rate)
            while at < dur:
                d = t0 + at - time.monotonic()
                if d > 0:
                    time.sleep(d)
                th = threading.Thread(target=gen, args=(a.port, rng.choice(texts), out), daemon=True)
                th.start()
                ths.append(th)
                at += rng.expovariate(rate)
            w1 = time.strftime("%H:%M:%S")
            for th in ths:
                th.join(900)
            n = len(ths)
            st200 = [r for r in out if r.get("status") == 200]
            rej = sum(1 for r in out if r.get("status") not in (None, 200))
            err = [r for r in out if "error" in r]
            met = sum(1 for r in st200 if r["ok"] and r["ttfa"] is not None and r["ttfa"] <= 0.2 and r["pause"] < 0.01)
            tt = np.array([r["ttfa"] for r in st200 if r["ttfa"] is not None] or [0.0]) * 1000
            log(f"POISSON rate {rate:g} for {dur:g}s ({w0}-{w1}): sent {n}, 200 {len(st200)}, rejected {rej}, errors {len(err)}, "
                f"SLO met {met} ({met / dur:.2f} req/s); ttfa ms p50 {np.median(tt):.0f} p90 {np.quantile(tt, .9):.0f} p99 {np.quantile(tt, .99):.0f}; "
                f"stalled {sum(1 for r in st200 if r['pause'] >= 0.01)}")
            rj = [r["chars"] for r in out if r.get("status") not in (None, 200)]
            log(f"POISSON rate {rate:g} audit: attainment among admitted {100 * met / max(len(st200), 1):.1f}%; mean chars admitted "
                f"{np.mean([r['chars'] for r in st200]) if st200 else float('nan'):.1f} rejected {np.mean(rj) if rj else float('nan'):.1f} "
                f"(all sent {np.mean([r['chars'] for r in out]) if out else float('nan'):.1f}); audio s {sum(r['bytes'] for r in st200) / 48000:.0f}")
            bad = [r for r in st200 if not r["ok"]] + err
            if bad:
                fails.append(f"poisson {rate:g}: {len(bad)} failed: {bad[:3]}")
        time.sleep(12)  # one more [batch] stats line from the replicas
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    log(f"DONE {'FAILED ' + str(fails) if fails else 'all ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
