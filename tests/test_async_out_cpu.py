"""CPU test: async per-call PCM delivery.

- CodecService._put: sync streams get their items on `out`; async streams' items go to their
  loop as ONE callback per call, in order, onto their asyncio queues; a closed loop is ignored.
- CodecStream.achunks yields what chunks() yields for the same items: pass-through without a
  hold; with the first-audio hold the first chunk waits (asyncio) until `hold_until` and carries every
  PCM already queued; an end marker or error right behind it ends the stream; errors raise; the
  stream is aborted at the end and when the consumer stops early.
- CodecStream.fail puts an error from another thread.
- Engine.submit binds a text /generate sink to the running loop only with ASYNC_OUT, on a loop.

    python tests/test_async_out_cpu.py
"""
import asyncio, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import codec_dev
from engine.codec_dev import CodecService, CodecStream


class FakeSvc:
    def __init__(self):
        self.cv = threading.Condition()
        self.backlog = 0


def svc_put():
    s = CodecService.__new__(CodecService)  # only _put
    return s


async def collect(st, limit=None):
    got = []
    async for c in st.achunks():
        got.append((time.time(), c))
        if limit and len(got) >= limit:
            break
    return got


def sync_collect(items, hold=None):
    st = CodecStream(FakeSvc())
    st.hold_until = hold
    for x in items:
        st.out.put(x)
    out = []
    try:
        for c in st.chunks():
            out.append(c)
    except RuntimeError as e:
        out.append(("err", str(e)))
    return out


def async_collect(items, hold=None, delay=0.0):
    async def main():
        loop = asyncio.get_running_loop()
        st = CodecStream(FakeSvc())
        st.bind_async(loop)
        st.hold_until = hold
        svc = svc_put()

        def feed():
            time.sleep(delay)
            for x in items:  # one call per item, as the delivery thread would
                svc._put([(st, x)])

        threading.Thread(target=feed).start()
        out = []
        try:
            async for c in st.achunks():
                out.append(c)
        except RuntimeError as e:
            out.append(("err", str(e)))
        assert st.aborted
        return out

    return asyncio.run(main())


def test_same_as_sync():
    cases = [[b"a", b"b", b"c", None], [b"a", RuntimeError("boom")], [RuntimeError("early")], [None]]
    for items in cases:
        assert async_collect(items) == sync_collect(items), items
        h = time.time() + 0.03
        assert async_collect(items, hold=h) == sync_collect(items, hold=h), items


def test_put_one_callback_per_call():
    async def main():
        loop = asyncio.get_running_loop()
        a, b, c = (CodecStream(FakeSvc()) for _ in range(3))
        a.bind_async(loop)
        b.bind_async(loop)  # c stays sync
        calls = []
        orig = loop.call_soon_threadsafe

        def spy(fn, *args):
            calls.append(fn)
            return orig(fn, *args)

        loop.call_soon_threadsafe = spy
        svc = svc_put()
        t = threading.Thread(target=lambda: [svc._put([(a, b"a%d" % i), (b, b"b%d" % i), (c, b"c%d" % i)]) for i in range(5)] + [svc._put([(a, None), (b, None), (c, None)])])
        t.start()
        t.join()
        ga = [x async for x in a.achunks()]
        gb = [x async for x in b.achunks()]
        assert ga == [b"a%d" % i for i in range(5)] and gb == [b"b%d" % i for i in range(5)], (ga, gb)
        assert [c.out.get_nowait() for _ in range(6)] == [b"c%d" % i for i in range(5)] + [None]
        assert len(calls) == 6 and all(f is codec_dev._fanout for f in calls), calls

    asyncio.run(main())


def test_hold_merges():
    async def main():
        loop = asyncio.get_running_loop()
        st = CodecStream(FakeSvc())
        st.bind_async(loop)
        t0 = time.time()
        st.hold_until = t0 + 0.1
        svc = svc_put()
        svc._put([(st, b"a")])
        svc._put([(st, b"bb")])

        def late():
            time.sleep(0.05)
            svc._put([(st, b"ccc")])
            time.sleep(0.15)
            svc._put([(st, b"dd")])
            svc._put([(st, None)])

        threading.Thread(target=late).start()
        got = await collect(st)
        assert [c for _, c in got] == [b"abbccc", b"dd"], got
        assert t0 + 0.1 - 1e-3 <= got[0][0] < t0 + 0.15, got[0][0] - t0
        assert st.first_frames == 3 and 90 < st.held_ms <= 101, (st.first_frames, st.held_ms)

    asyncio.run(main())


def test_hold_does_not_block_loop():
    async def main():
        loop = asyncio.get_running_loop()
        sts = []
        for _ in range(50):
            st = CodecStream(FakeSvc())
            st.bind_async(loop)
            st.hold_until = time.time() + 0.1
            codec_dev.CodecService._put(None, [(st, b"x"), (st, None)])
            sts.append(st)
        t = time.time()
        res = await asyncio.gather(*(collect(s) for s in sts))
        assert all([c for _, c in r] == [b"x"] for r in res)
        assert time.time() - t < 0.2, time.time() - t  # 50 holds of 0.1 s run concurrently

    asyncio.run(main())


def test_early_stop_aborts_and_fail():
    async def main():
        loop = asyncio.get_running_loop()
        st = CodecStream(FakeSvc())
        st.bind_async(loop)
        for x in (b"a", b"b", b"c"):
            codec_dev.CodecService._put(None, [(st, x)])
        gen = st.achunks()
        assert await gen.__anext__() == b"a"
        await gen.aclose()  # the client went away
        assert st.aborted
        st = CodecStream(FakeSvc())
        st.bind_async(loop)
        codec_dev.CodecService._put(None, [(st, b"a")])
        threading.Thread(target=lambda: (time.sleep(0.02), st.fail(RuntimeError("lm error")))).start()
        got = []
        try:
            async for c in st.achunks():
                got.append(c)
            raise AssertionError("no error")
        except RuntimeError as e:
            assert str(e) == "lm error"
        assert got == [b"a"]

    asyncio.run(main())
    st = CodecStream(FakeSvc())  # sync fail
    st.fail(RuntimeError("x"))
    assert isinstance(st.out.get_nowait(), RuntimeError)


def test_closed_loop_ignored():
    loop = asyncio.new_event_loop()
    st = CodecStream(FakeSvc())
    st.aq, st.loop = None, loop
    st.bind_async(loop)
    loop.close()
    codec_dev.CodecService._put(None, [(st, b"a")])  # no raise
    st.fail(RuntimeError("x"))


def test_server_binds_only_async_on_loop():
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
    assert server.ASYNC_OUT  # default on
    assert eng.submit("gen", "hi", "Ryan", "English").sink.aq is None  # no running loop: sync

    async def on_loop():
        j = eng.submit("gen", "hi", "Ryan", "English")
        assert j.sink.aq is not None and j.sink.loop is asyncio.get_running_loop()
        assert abs(j.sink.hold_until - (j.t + server.HOLD_S)) < 1e-9  # first-audio hold kept
        server.ASYNC_OUT = False
        try:
            assert eng.submit("gen", "hi", "Ryan", "English").sink.aq is None
        finally:
            server.ASYNC_OUT = True

    asyncio.run(on_loop())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all ok")
