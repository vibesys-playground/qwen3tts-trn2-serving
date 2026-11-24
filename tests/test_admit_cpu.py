"""CPU unit test of the token-bucket admission gate and the deadline-feasibility gate (engine.admit) with
a fake clock; the deadline gate runs against the real CodecService.urgent / call_full_s on a
simulated codec (EDF calls of up to 64 slots, cycle a + b x active).

    python tests/test_admit_cpu.py
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.admit import AdmissionGate, DeadlineGate  # noqa: E402
import engine.codec_dev as CD  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def simulate(rate: float, secs: float = 600.0, cap: float = 1000 / 0.9, frames: int = 100, dur: float = 8.0, seed: int = 0,
             lengths=None):
    """Poisson arrivals of `frames`-frame requests (100 characters at f = 1) that stay open `dur`
    seconds; the codec is a fluid queue draining `cap` frames/s, fed each admitted request's frames
    evenly over its lifetime. With `lengths` (lo, hi), each request has uniform lo..hi characters,
    one frame per character, open dur x chars / 100 seconds. Returns admitted req/s, rejections and
    the gate; the gate also gets .adm_chars and .rej_chars (the lengths of each side)."""
    clk = Clock()
    g = AdmissionGate(rho=0.9, nmin=64, clock=clk)
    rng = random.Random(seed)
    t, backlog, live = 0.0, 0.0, []  # live: (end time, est, chars)
    adm_chars, rej_chars = [], []
    dt_sim = 0.01
    arrivals, a = [], 0.0
    while a < secs + 1:
        a += rng.expovariate(rate)
        arrivals.append(a)
    i = 0
    while t < secs:
        t += dt_sim
        clk.t = t
        backlog = max(0.0, backlog + len(live) * frames / dur * dt_sim - cap * dt_sim)
        for e in [x for x in live if x[0] <= t]:
            live.remove(e)
            g.finish(e[1], e[2], e[2] * frames // 100, True)
        while i < len(arrivals) and arrivals[i] <= t:
            chars = 100 if lengths is None else rng.randint(*lengths)
            ok, est = g.admit(chars, cap, backlog, len(live))
            if ok:
                live.append((t + dur * chars / 100, est, chars))
                adm_chars.append(chars)
            else:
                rej_chars.append(chars)
            i += 1
    g.adm_chars, g.rej_chars = adm_chars, rej_chars
    return g.admitted / secs, g.rejected, g


class FStream:
    """A codec stream for the simulation (the fields CodecService.urgent and deadline() read)."""
    deadline = CD.CodecStream.deadline
    anchor = CD.CodecStream.anchor
    hold_until = None

    def __init__(self, t, nframes, chars):
        self.t_arrival, self.t_first_out, self.delivered = t, None, 0
        self.frames, self.ended, self.aborted = [], False, False
        self.total, self.produced, self.chars = nframes, 0, chars
        self.late = 0


class FSvc:
    """CodecService's state for the deadline gate, on a fake clock; urgent and call_full_s are
    CodecService's own methods."""

    def __init__(self, clk, a=0.0172, b=0.000172):
        import collections, threading
        self.clk, self.a, self.b = clk, a, b  # full 64-slot call 28.2 ms: 2270 frames/s
        self.cv = threading.Condition()
        self.streams = []
        self.calls = collections.deque(maxlen=64)
        self.max_groups, self.fps_seed = 8, 64 / (a + 64 * b)
        self.dc = type("DC", (), {"buckets": [2, 4, 8]})()

    def urgent(self, h):
        return CD.CodecService.urgent(self, h, now=self.clk.t)

    call_full_s = CD.CodecService.call_full_s


def sim_deadline(rate, secs=120.0, seed=0, lengths=(20, 200), slow=None, prefill=0.1, lm_fps=20.0, gate=None):
    """Poisson arrivals (chars uniform in `lengths`, one frame per character) at a deadline gate;
    admitted streams get frames from the LM at lm_fps after `prefill`, and the codec makes
    back-to-back EDF calls of up to 64 streams with a pending frame, each lasting a + b x active
    (x 1.5 after `slow` seconds, if given). Returns a dict of results."""
    clk = Clock()
    svc = FSvc(clk)
    g = gate or DeadlineGate(h=0.5, u=0.9, nmin=64)
    rng = random.Random(seed)
    arr, x = [], 0.0
    while x < secs:
        x += rng.expovariate(rate)
        arr.append(x)
    i, t, t_call_end = 0, 0.0, 0.0
    adm_c, rej_c, late, deliv, done, met = [], [], 0, 0, 0, 0
    tl = []  # (t, admitted so far, late frames pending)
    dt = 0.002
    while t < secs + 20:
        t += dt
        clk.t = t
        for st in svc.streams:  # the LM
            if t - st.t_arrival >= prefill and st.produced < st.total:
                want = min(st.total, int((t - st.t_arrival - prefill) * lm_fps) + 1)
                while st.produced < want:
                    st.frames.append(0)
                    st.produced += 1
                if st.produced == st.total:
                    st.ended = True
        if t >= t_call_end:  # the codec: one call at a time
            work = sorted([s for s in svc.streams if s.frames], key=lambda s: s.deadline())[:64]
            if work:
                cyc = (svc.a + svc.b * len(work)) * (1.5 if slow is not None and t >= slow else 1.0)
                t_call_end = t + cyc
                svc.calls.append((len(work), len({j // 8 for j in range(len(work))}), cyc))
                for st in work:
                    st.frames.pop(0)
                    if st.t_first_out is None:
                        st.t_first_out = t_call_end
                        if t_call_end > st.t_arrival + 0.2:
                            st.late += 1
                    elif t_call_end > st.deadline() + 1e-9:
                        st.late += 1
                    st.delivered += 1
                    deliv += 1
        for st in [s for s in svc.streams if s.ended and not s.frames]:
            svc.streams.remove(st)
            late += st.late
            done += 1
            met += st.late == 0
        while i < len(arr) and arr[i] <= t:
            chars = rng.randint(*lengths)
            if g.decide(svc):
                svc.streams.append(FStream(t, chars, chars))
                adm_c.append(chars)
            else:
                rej_c.append(chars)
            i += 1
        if int(t / 0.5) != int((t - dt) / 0.5):
            pend_late = sum(max(0, math.floor((t - s.deadline()) / CD.FRAME_S) + 1) for s in svc.streams if s.frames or not s.ended)
            tl.append((t, len(adm_c), pend_late))
    return dict(adm=len(adm_c) / secs, rej=len(rej_c), adm_c=adm_c, rej_c=rej_c, late_frac=late / max(deliv, 1),
                attain=met / max(done, 1), fps=deliv / (secs + 20), tl=tl, gate=g, svc=svc)


def test_deadline():
    # the decision does not depend on the request: it never sees one (same state, same answer)
    import inspect
    assert list(inspect.signature(DeadlineGate.decide).parameters) == ["self", "svc"]
    clk = Clock()
    clk.t = 100.0
    svc = FSvc(clk)
    g = DeadlineGate(h=0.5, u=0.9, nmin=64)
    assert g.allow == 6.25, g.allow  # 0.5 s of playback (first frame at TTFA 0.2 s, then 0.28, 0.36, 0.44)
    for k in range(63):
        svc.streams.append(FStream(clk.t - 5, 100, 100))
        svc.streams[-1].t_first_out = clk.t - 10  # 10 s behind: hugely late
    assert all(g.decide(svc) for _ in range(10))  # below NMIN: always admitted
    svc.streams.append(FStream(clk.t, 100, 100))
    answers = set()
    for chars in (1, 100, 10000):  # the arriving request's length cannot enter; same state -> same answer
        answers.add(g.decide(svc))
    assert answers == {False}, answers
    for st in svc.streams:
        st.t_first_out, st.delivered = clk.t + 5, 0  # everyone 5 s ahead: nothing due
    assert all(g.decide(svc) for chars in (1, 100, 10000))
    # urgent(): a fresh stream counts its 6.25-frame playback rate within 0.5 s, a stream 1 s late
    # 13 + 6 frames, an ended stream at most its pending frames
    svc.streams = [FStream(clk.t, 100, 100)]
    assert svc.urgent(0.5) == 6.25, svc.urgent(0.5)
    st = FStream(clk.t - 3, 100, 100)
    st.t_first_out, st.delivered = clk.t - 1.0, 0
    svc.streams = [st]
    assert svc.urgent(0.5) == 19, svc.urgent(0.5)
    st = FStream(clk.t - 3, 100, 100)
    st.t_first_out, st.delivered, st.ended, st.frames = clk.t - 1.0, 0, True, [0, 0]
    svc.streams = [st]
    assert svc.urgent(0.5) == 2
    # call_full_s: the fit recovers a + 64 b from calls at 40-64 active slots; ignores small buckets
    for n in range(40, 65):
        svc.calls.append((n, 8, 0.0172 + 0.000172 * n))
    svc.calls.append((8, 1, 0.005))
    pc, a, b, n = svc.call_full_s()
    assert abs(pc - 0.0282) < 1e-4 and abs(b - 0.000172) < 1e-6, (pc, a, b)

    # load: 2270 frames/s at full fill, 20-200 chars (110 frames mean, ~8.8 s of audio)
    res = {}
    for rate in (10.0, 15.0, 20.0, 30.0):
        r = sim_deadline(rate, seed=int(rate))
        res[rate] = r
        ma = sum(r["adm_c"]) / len(r["adm_c"])
        mr = sum(r["rej_c"]) / len(r["rej_c"]) if r["rej_c"] else float("nan")
        print(f"deadline gate, {rate:g} req/s offered: admitted {r['adm']:.2f} req/s ({r['adm'] * ma / 1:.0f} frames/s), "
              f"rejected {r['rej']}, delivered {r['fps']:.0f} fps, late frames {100 * r['late_frac']:.2f}%, "
              f"attainment {100 * r['attain']:.1f}%; mean chars admitted {ma:.1f} rejected {mr:.1f}")
    assert res[10.0]["rej"] == 0 and res[15.0]["rej"] == 0, (res[10.0]["rej"], res[15.0]["rej"])
    r = res[30.0]
    assert r["rej"] > 100 and 1800 <= r["adm"] * 110 <= 2270, r["adm"]
    assert r["late_frac"] < 0.01 and r["attain"] > 0.95, (r["late_frac"], r["attain"])
    for rate in (20.0, 30.0):
        r = res[rate]
        if r["rej"] > 300:  # enough rejections for the means to settle (sd of 20-200 is 52 chars)
            ma = sum(r["adm_c"]) / len(r["adm_c"])
            mr = sum(r["rej_c"]) / len(r["rej_c"])
            assert abs(mr / ma - 1) <= 0.10, (rate, ma, mr)

    # a 1.5x host slowdown at t = 60 s: admission is cut within 2 s and the late backlog drains
    r = sim_deadline(30.0, seed=7, slow=60.0)
    tl = r["tl"]
    before = [x for x in tl if 50 <= x[0] < 60]
    after = [x for x in tl if 62 <= x[0] < 72]
    rate_b = (before[-1][1] - before[0][1]) / (before[-1][0] - before[0][0])
    rate_a = (after[-1][1] - after[0][1]) / (after[-1][0] - after[0][0])
    peak = max(x[2] for x in tl if x[0] >= 60)
    print(f"slowdown at 60 s: admitted {rate_b:.1f} -> {rate_a:.1f} req/s; pending late frames peak {peak}, "
          f"at 63-66 s {[x[2] for x in tl if 63 <= x[0] <= 66]}; late {100 * r['late_frac']:.2f}%")
    assert rate_a < 0.8 * rate_b, (rate_b, rate_a)
    assert all(x[2] <= 0.5 * 2270 / 1.5 for x in tl if 63 <= x[0] < 120), [x for x in tl if x[0] >= 60][:20]
    print("deadline gate: OK")


def main():
    # rho x C = 1000 frames/s, 100-frame requests, 15 req/s offered: admits ~10 req/s
    adm, rej, g = simulate(15.0)
    print(f"15 req/s offered: admitted {adm:.2f} req/s, rejected {rej}; {g.stats(1000 / 0.9)}")
    assert 9.5 <= adm <= 10.5, adm
    # 5 req/s: everything admitted
    adm, rej, g = simulate(5.0)
    print(f"5 req/s offered: admitted {adm:.2f} req/s, rejected {rej}")
    assert rej == 0, rej
    # 9 req/s (0.81 C): still everything admitted (bursts absorbed by the bucket depth)
    adm, rej, g = simulate(9.0)
    print(f"9 req/s offered: admitted {adm:.2f} req/s, rejected {rej}")
    assert rej == 0, rej
    # length neutrality: mixed lengths (20-200 characters) under overload; rejected and admitted
    # requests have the same mean length (within 10%), and every rate is admitted at low load
    for rate, seed in ((20.0, 1), (30.0, 2)):
        adm, rej, g = simulate(rate, lengths=(20, 200), seed=seed)
        ma = sum(g.adm_chars) / len(g.adm_chars)
        mr = sum(g.rej_chars) / len(g.rej_chars)
        print(f"{rate:.0f} req/s mixed lengths: admitted {adm:.2f} req/s, rejected {rej}; mean chars admitted {ma:.1f} rejected {mr:.1f}")
        assert rej > 100, rej
        assert abs(mr / ma - 1) <= 0.10, (ma, mr)
    for rate in (5.0, 8.0):
        adm, rej, g = simulate(rate, lengths=(20, 200), seed=3)
        print(f"{rate:.0f} req/s mixed lengths: admitted {adm:.2f} req/s, rejected {rej}")
        assert rej == 0, rej
    # the decision does not depend on the request: same state, any length, same answer
    for chars in (1, 100, 10000):
        clk = Clock()
        g = AdmissionGate(rho=0.9, nmin=64, clock=clk)
        g.admit(10, 100.0, 1e9, 64)
        g.tokens = -1.0
        assert g.admit(chars, 100.0, 1e9, 64)[0] is False
        g.tokens = 0.0
        assert g.admit(chars, 100.0, 1e9, 64)[0] is True
    # never below NMIN open streams, whatever the tokens and backlog
    clk = Clock()
    g = AdmissionGate(rho=0.9, nmin=64, clock=clk)
    for k in range(200):
        ok, _ = g.admit(1000, 100.0, 1e9, 63)
        assert ok
    assert g.tokens < 0
    # at NMIN with no tokens and a big backlog: rejected; with an idle codec (no backlog): admitted
    assert g.admit(100, 100.0, 1e9, 64)[0] is False
    assert g.admit(100, 100.0, 0, 64)[0] is True
    # refund on finish and the frames-per-char estimate adapts
    clk = Clock()
    g = AdmissionGate(rho=1.0, nmin=0, clock=clk)
    t0 = g.tokens if g.tokens is not None else None
    ok, est = g.admit(100, 1000.0, 0, 0)
    assert ok and est == 100 and g.tokens == 1500 - 100, (est, g.tokens, t0)
    g.finish(est, 100, 40, True)
    assert g.tokens == 1500 - 40, g.tokens
    for _ in range(300):
        g.finish(0, 100, 50, True)
    assert abs(g.f - 0.5) < 0.01, g.f
    assert g.admit(100, 1000.0, 0, 0)[1] in (50, 51)
    # cancelled requests do not update f
    f = g.f
    g.finish(0, 100, 1, False)
    assert g.f == f
    print("admission gate: OK")
    test_deadline()


if __name__ == "__main__":
    main()
