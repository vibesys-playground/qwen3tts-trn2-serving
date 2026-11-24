"""The check: the candidate server at VIBESYS_SERVICE_URL against the frozen CPU reference.

    python check.py          exit 0 only on a pass; every failure is printed

1. Weights: the operator's read-only "models" mount holds the pinned checkpoint
   (reference/weights.json: every file's size, the small files' sha256).
2. Scoring gate (`POST /score`): teacher-forced log-probabilities of the reference's own
   sampled codes (golden/golden.json, ten utterances, all sent at once so they share batches)
   against the FP32 CPU reference: all 16 codebooks of every frame and EOS after the last frame.
3. Codec gate (`POST /generate` with `codes`): the server streams its decode of the reference's
   codes through its serving path; the PCM must have exactly 1920 samples per frame and match the
   reference decode (golden/golden_pcm.npz) by SNR, over the whole utterance and every 10-frame
   window that is not silent.
4. Sampled gate (`POST /generate`, the default sampling): SAMPLED_TEXTS LibriTTS utterances, all
   at once, each must complete with a duration in a plausible range for its text, and the whole
   set must transcribe (openai/whisper-small.en, CPU, from the "hf" mount) with a word error rate
   within the reference's own (golden/sampled_reference.json). One text is sent twice: sampling
   must give two different waveforms (no greedy decoding, no caching of answers).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import client  # noqa: E402

HERE = Path(__file__).parent
GOLDEN = HERE / "golden" / "golden.json"
GOLDEN_PCM = HERE / "golden" / "golden_pcm.npz"
SAMPLED_REF = HERE / "golden" / "sampled_reference.json"
WEIGHTS_MANIFEST = HERE / "reference" / "weights.json"
MOUNT = Path("/vibesys/mounts/models")
ASR_MODEL = "openai/whisper-small.en"
ASR_REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"

# Scoring gate: from the BF16 calibration in golden.json (see OBJECTIVE.md).
LOGPROB_MEAN_MAX = 0.08
LOGPROB_P99_MAX = 0.40
LOGPROB_MAX = 2.0
EOS_ABS_MAX = 0.5
# Codec gate.
CODEC_SNR_MIN_DB = 15.0
CODEC_WINDOW_SNR_MIN_DB = 10.0
# Sampled gate.
DURATION_RANGE = (0.4, 2.5)  # x the reference's seconds per character, per utterance


def problems_weights() -> list[str]:
    man = json.loads(WEIGHTS_MANIFEST.read_text())
    root = MOUNT / man["mount_dir"]
    out = []
    for name, want in man["files"].items():
        p = root / name
        if not p.is_file():
            out.append(f"{p}: missing")
        elif p.stat().st_size != want["size"]:
            out.append(f"{p}: {p.stat().st_size} bytes, the pinned checkpoint has {want['size']}")
        elif want["hashed_in_check"] and hashlib.sha256(p.read_bytes()).hexdigest() != want["sha256"]:
            out.append(f"{p}: sha256 differs from the pinned checkpoint")
    return out


async def post_json(sess, url: str, path: str, body: dict) -> tuple[int, object]:
    async with sess.post(f"{url}{path}", json=body) as resp:
        text = await resp.text()
        try:
            return resp.status, json.loads(text)
        except json.JSONDecodeError:
            return resp.status, text[:300]


async def score_gate(url: str, cases: list[dict]) -> list[str]:
    out = []
    async with client.session() as sess:
        replies = await asyncio.gather(
            *(
                post_json(sess, url, "/score", {"text": c["text"], "speaker": client.SPEAKER, "language": client.LANGUAGE, "codes": c["codes"]})
                for c in cases
            )
        )
    deltas, eos_bad = [], []
    for c, (status, reply) in zip(cases, replies):
        if status != 200 or not isinstance(reply, dict):
            out.append(f"/score {c['name']}: HTTP {status}: {reply}")
            continue
        try:
            got = np.asarray(reply["logprobs"], dtype=np.float64)
            eos = float(reply["eos_logprob"])
        except (KeyError, TypeError, ValueError) as e:
            out.append(f"/score {c['name']}: malformed reply ({e})")
            continue
        want = np.asarray(c["logprobs"], dtype=np.float64)
        if got.shape != want.shape:
            out.append(f"/score {c['name']}: logprobs shape {got.shape}, expected {want.shape}")
            continue
        if not np.all(np.isfinite(got)) or np.any(got > 1e-6):
            out.append(f"/score {c['name']}: logprobs must be finite and <= 0")
            continue
        deltas.append(np.abs(got - want).ravel())
        if abs(eos - c["eos_logprob"]) > EOS_ABS_MAX:
            eos_bad.append(f"{c['name']}: EOS logprob {eos:.3f}, reference {c['eos_logprob']:.3f}")
    if deltas:
        d = np.concatenate(deltas)
        mean, p99, mx = float(d.mean()), float(np.quantile(d, 0.99)), float(d.max())
        print(f"score gate: {len(d)} logprobs, mean |d| {mean:.4f} (max {LOGPROB_MEAN_MAX}), p99 {p99:.4f} (max {LOGPROB_P99_MAX}), max {mx:.4f} (max {LOGPROB_MAX})")
        if mean > LOGPROB_MEAN_MAX or p99 > LOGPROB_P99_MAX or mx > LOGPROB_MAX:
            out.append(f"/score: teacher-forced logprobs differ from the FP32 reference: mean |d| {mean:.4f}, p99 {p99:.4f}, max {mx:.4f}")
    out += [f"/score {e}" for e in eos_bad]
    return out


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    noise = np.sum((ref - test) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.sum(ref**2) / max(noise, 1e-12)))


def worst_window(ref: np.ndarray, test: np.ndarray, spf: int) -> float:
    vals = []
    for s in range(0, len(ref), 10 * spf):
        r = ref[s : s + 10 * spf].astype(np.float64) / 32767
        if np.sqrt(np.mean(r**2)) < 1e-3:
            continue
        vals.append(snr_db(ref[s : s + 10 * spf], test[s : s + 10 * spf]))
    return min(vals) if vals else float("inf")


async def codec_gate(url: str, cases: list[dict], spf: int) -> list[str]:
    pcm = np.load(GOLDEN_PCM)
    out = []
    async with client.session() as sess:
        results = [client.Result(text=c["text"], sent=0.0) for c in cases]
        await asyncio.gather(*(client.request(sess, url, r, keep_pcm=True, codes=c["codes"]) for r, c in zip(results, cases)))
    worst = []
    for c, r in zip(cases, results):
        if not r.complete:
            out.append(f"/generate codes {c['name']}: {r.error or 'incomplete'}")
            continue
        got = np.frombuffer(bytes(r.pcm), dtype="<i2")
        ref = pcm[c["name"]]
        if len(got) != len(ref):
            out.append(f"/generate codes {c['name']}: {len(got)} samples, {c['frames']} frames decode to {len(ref)}")
            continue
        s, w = snr_db(ref, got), worst_window(ref, got, spf)
        worst.append((s, w, c["name"]))
        if s < CODEC_SNR_MIN_DB or w < CODEC_WINDOW_SNR_MIN_DB:
            out.append(f"/generate codes {c['name']}: SNR {s:.1f} dB (min {CODEC_SNR_MIN_DB}), worst 10-frame window {w:.1f} dB (min {CODEC_WINDOW_SNR_MIN_DB})")
    if worst:
        print(f"codec gate: worst SNR {min(x[0] for x in worst):.1f} dB, worst window {min(x[1] for x in worst):.1f} dB")
    return out


def normalizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained(ASR_MODEL, revision=ASR_REVISION).normalize


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def transcribe(pcms: list[np.ndarray]) -> list[str]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    proc = WhisperProcessor.from_pretrained(ASR_MODEL, revision=ASR_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, revision=ASR_REVISION).eval()
    import scipy.signal

    out = []
    for pcm in pcms:
        wav = pcm.astype(np.float32) / 32768.0
        # Whisper takes 16 kHz: polyphase resampling 24 kHz -> 16 kHz (2:3).
        wav16 = scipy.signal.resample_poly(wav, 2, 3).astype(np.float32)
        feats = proc(wav16, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    return out


async def sampled_gate(url: str) -> list[str]:
    ref = json.loads(SAMPLED_REF.read_text())
    items = ref["texts"]
    texts = [t["text"] for t in items] + [items[0]["text"]]  # the first text twice
    out = []
    async with client.session() as sess:
        results = [client.Result(text=t, sent=0.0) for t in texts]
        await asyncio.gather(*(client.request(sess, url, r, keep_pcm=True) for r in results))
    rate = ref["audio_seconds_per_char"]
    pcms = []
    for r in results:
        if not r.complete or r.audio_bytes == 0:
            out.append(f"/generate {r.text[:40]!r}: {r.error or 'no audio'}")
            pcms.append(None)
            continue
        expected = rate * len(r.text)
        lo, hi = DURATION_RANGE
        if not (lo * expected <= r.audio_seconds <= hi * expected + 1.0):
            out.append(f"/generate {r.text[:40]!r}: {r.audio_seconds:.2f} s of audio, expected about {expected:.2f} s")
        pcms.append(np.frombuffer(bytes(r.pcm), dtype="<i2"))
    if pcms[0] is not None and pcms[-1] is not None:
        a, b = pcms[0], pcms[-1]
        if len(a) == len(b) and np.array_equal(a, b):
            out.append("/generate: the same text twice gave identical audio; sampling must sample (no greedy decoding, no cached answers)")
    if out:
        return out
    hyps = transcribe(pcms[:-1])
    norm = normalizer()
    errs = words = 0
    for t, h in zip(items, hyps):
        e, n = word_errors(norm(t["text"]), norm(h))
        errs, words = errs + e, words + n
    wer = errs / max(words, 1)
    limit = max(2 * ref["wer"], ref["wer"] + 0.05)
    print(f"sampled gate: WER {wer:.3f} over {words} words (reference {ref['wer']:.3f}, limit {limit:.3f})")
    if wer > limit:
        out.append(f"/generate: word error rate {wer:.3f} of the sampled speech, limit {limit:.3f} (reference {ref['wer']:.3f})")
    return out


async def health(url: str) -> list[str]:
    async with client.session() as sess:
        async with sess.get(f"{url}/health") as resp:
            return [] if resp.status // 100 == 2 else [f"/health: HTTP {resp.status}"]


def main() -> int:
    url = os.environ.get("VIBESYS_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-weights":
        problems = []
    else:
        problems = problems_weights()
        if problems:
            print("the models mount is not Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice @ 0c0e305:")
            print("\n".join(problems))
            return 1
    golden = json.loads(GOLDEN.read_text())
    cases, spf = golden["cases"], golden["meta"]["samples_per_frame"]
    for name, gate in (
        ("health", lambda: health(url)),
        ("score", lambda: score_gate(url, cases)),
        ("codec", lambda: codec_gate(url, cases, spf)),
        ("sampled", lambda: sampled_gate(url)),
    ):
        found = asyncio.run(gate())
        print(f"{name}: {'pass' if not found else 'FAIL'}", flush=True)
        problems += found
        if name == "health" and found:
            break
    if problems:
        print("\n".join(problems))
        return 1
    print("check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
