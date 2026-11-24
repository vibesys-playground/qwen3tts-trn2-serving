"""CPU test: the first-audio release hold and the release-anchored codec deadlines.

- CodecStream.deadline: TTFA budget before the first PCM; then anchored at max(first PCM, hold).
- CodecStream.chunks: with a hold, the first chunk waits until `hold_until` and carries every PCM
  already queued (and an end marker or error right behind it ends the stream); without a hold,
  chunks pass through one by one, unchanged.
- The late histogram binning of the [codec] stats line.
- Engine.submit sets the hold on text /generate jobs only.

    python tests/test_hold_cpu.py
"""
import sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import codec_dev
from engine.codec_dev import FRAME_S, TTFA_S, CodecStream, late_hist_add, late_hist_str


class FakeSvc:
    def __init__(self):
        self.cv = threading.Condition()
        self.backlog = 0


def test_deadline():
    st = CodecStream(FakeSvc())
    assert st.deadline() == st.t_arrival + TTFA_S
    st.t_first_out = 100.0
    st.delivered = 3
    assert abs(st.deadline() - (100.0 + 3 * FRAME_S)) < 1e-9  # no hold
    st.hold_until = 100.05  # release after the PCM was ready: anchored at the release
    assert abs(st.deadline() - (100.05 + 3 * FRAME_S)) < 1e-9
    st.hold_until = 99.9  # PCM later than the hold: anchored at the PCM
    assert abs(st.deadline() - (100.0 + 3 * FRAME_S)) < 1e-9
    assert st.anchor() == 100.0


def drain(st):
    got = []
    for c in st.chunks():
        got.append((time.time(), c))
    return got


def test_hold_merges():
    st = CodecStream(FakeSvc())
    t0 = time.time()
    st.hold_until = t0 + 0.1
    st.out.put(b"a")
    st.out.put(b"bb")

    def late():
        time.sleep(0.05)
        st.out.put(b"ccc")  # decoded during the hold: goes out with the first chunk
        time.sleep(0.15)
        st.out.put(b"dd")  # after the release: its own chunk
        st.out.put(None)

    threading.Thread(target=late).start()
    got = drain(st)
    assert [c for _, c in got] == [b"abbccc", b"dd"], got
    assert got[0][0] >= t0 + 0.1 - 1e-3, got[0][0] - t0
    assert got[0][0] < t0 + 0.15
    assert st.first_frames == 3 and 90 < st.held_ms <= 101, (st.first_frames, st.held_ms)
    assert st.aborted


def test_hold_end_and_error():
    st = CodecStream(FakeSvc())
    st.hold_until = time.time() + 0.02
    for x in (b"a", b"b", None):
        st.out.put(x)
    assert [c for _, c in drain(st)] == [b"ab"]
    st = CodecStream(FakeSvc())
    st.hold_until = time.time() + 0.02
    for x in (b"a", RuntimeError("boom")):
        st.out.put(x)
    got = []
    try:
        for c in st.chunks():
            got.append(c)
        raise AssertionError("no error")
    except RuntimeError as e:
        assert str(e) == "boom"
    assert got == [b"a"]
    st = CodecStream(FakeSvc())  # an error before any PCM is raised as before
    st.hold_until = time.time() + 10
    st.out.put(RuntimeError("early"))
    t = time.time()
    try:
        list(st.chunks())
        raise AssertionError("no error")
    except RuntimeError:
        assert time.time() - t < 1  # not held


def test_hold_passed():
    st = CodecStream(FakeSvc())
    st.hold_until = time.time() - 1  # PCM later than the hold: no wait
    for x in (b"a", b"b", None):
        st.out.put(x)
    t = time.time()
    assert [c for _, c in drain(st)] == [b"ab"]  # whatever is queued still goes out together
    assert time.time() - t < 0.01 and st.held_ms == 0


def test_no_hold_unchanged():
    st = CodecStream(FakeSvc())
    for x in (b"a", b"b", b"c", None):
        st.out.put(x)
    assert [c for _, c in drain(st)] == [b"a", b"b", b"c"]


def test_hist():
    s = codec_dev.CodecService._zero_stats()
    st = CodecStream(FakeSvc())
    late_hist_add(s, st)
    for k, ms in ((1, 3), (2, 12), (2, 12), (4, 25), (9, 7)):
        st.max_late, st.max_late_k = ms / 1000, k
        late_hist_add(s, st)
    assert s["ok"] == 1
    assert s["hist"].tolist() == [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 1, 0, 0]], s["hist"].tolist()
    line = late_hist_str(s["hist"], s["ok"])
    assert "streams ended 6, never late 1" in line and "k2 0/0/2/0" in line, line


def test_server_sets_hold():
    from engine import server

    class R:
        def __init__(self):
            import queue
            self.rid, self.outstanding, self.inbox = 0, 0, queue.Queue()

    class Codecs:
        def open(self):
            return CodecStream(FakeSvc())

    eng = server.Engine.__new__(server.Engine)
    eng.lock, eng.next_id, eng.replicas, eng.rr, eng.jobs, eng.codecs = threading.Lock(), 0, [R()], 0, {}, Codecs()
    job = eng.submit("gen", "hello", "Ryan", "English")
    assert server.HOLD_S == 0.15 and abs(job.sink.hold_until - (job.t + 0.15)) < 1e-9
    assert eng.submit("score", "hello", "Ryan", "English", []).sink is None
    assert eng.codecs.open().hold_until is None  # the codes path opens its own stream: no hold


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all ok")
