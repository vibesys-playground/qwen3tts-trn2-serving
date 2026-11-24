"""In-process HTTP round trip on CPU: uvicorn in a thread, urllib requests to each route."""
import json, sys, threading, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import uvicorn
import os
os.environ.setdefault("QWEN3TTS_BUCKETS", "2,4")
os.environ.setdefault("QWEN3TTS_SHRINK_AFTER", "3")
os.environ.setdefault("QWEN3TTS_TIMING", "0")
os.environ.setdefault("QWEN3TTS_CODEC_GROUPS", "1")  # one 8-slot bucket: the CPU runs the codec graph eagerly
from engine import server

G = json.loads((ROOT / "eval/golden/golden.json").read_text())
eng = server.Engine("cpu", server.Watchdog(1e9))
cfg = uvicorn.Config(server.make_app(eng), host="127.0.0.1", port=18765, log_level="warning")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
while not srv.started:
    time.sleep(0.1)

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:18765{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, r.read()

print("health", urllib.request.urlopen("http://127.0.0.1:18765/health").status)
c = G["cases"][8]
st, b = post("/score", {"text": c["text"], "speaker": "Ryan", "language": "English", "codes": c["codes"]})
r = json.loads(b); print("score", st, len(r["logprobs"]), len(r["logprobs"][0]), r["eos_logprob"])
st, b = post("/generate", {"text": c["text"], "speaker": "Ryan", "language": "English", "stream": True, "codes": c["codes"]})
print("codes", st, len(b) // 2, 1920 * c["frames"])
# The async delivery path (default) and the sync control give the same bytes.
assert server.ASYNC_OUT
server.ASYNC_OUT = False
st0, b0 = post("/generate", {"text": c["text"], "speaker": "Ryan", "language": "English", "stream": True, "codes": c["codes"]})
server.ASYNC_OUT = True
assert st0 == 200 and b0 == b and len(b) == 2 * 1920 * c["frames"], (st0, len(b0), len(b))
print("codes sync == async", len(b0) == len(b))
# A client that goes away mid-stream cancels its job (async path).
import http.client
hc = http.client.HTTPConnection("127.0.0.1", 18765, timeout=600)
hc.request("POST", "/generate", body=json.dumps({"text": c["text"], "speaker": "Ryan", "language": "English"}), headers={"Content-Type": "application/json"})
r = hc.getresponse()
first = r.read1(65536)
assert r.status == 200 and len(first) > 0, (r.status, len(first))
hc.sock.close()
hc.close()
t = time.time()
while eng.jobs and time.time() - t < 120:
    time.sleep(0.1)
assert not eng.jobs, f"job not cancelled after the client left: {list(eng.jobs)}"
print("abort mid-stream: job cancelled in", round(time.time() - t, 2), "s")
st, b = post("/generate", {"text": "Yes, of course.", "speaker": "Ryan", "language": "English", "stream": True})
print("generate", st, len(b) / 48000, "s")
srv.should_exit = True
