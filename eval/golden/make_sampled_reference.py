"""Build golden/sampled_reference.json: what the reference's sampled speech looks like.

Run on CPU with the official qwen-tts package (0.1.1) and the checkpoint at revision 0c0e305:

    python make_sampled_reference.py --weights <checkpoint dir>

It picks N_TEXTS LibriTTS test-clean utterances (seeded, 5 to 40 words, none of the golden.json
ones), samples each REPEATS times in FP32 with the checkpoint's sampling defaults (speaker Ryan,
English), and records:

- `texts`: the utterances the check's sampled gate sends;
- `audio_seconds_per_char`: total audio seconds over total characters, the scale of the
  check's per-utterance duration range and of the benchmark's integrity guard;
- `wer`: the word error rate of openai/whisper-small.en (revision e872752) on the reference's
  own speech, after Whisper's English normalizer, the bar the check's WER limit is set from.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.signal
import torch
from qwen_tts import Qwen3TTSModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

N_TEXTS = 16
REPEATS = 2
ASR_MODEL = "openai/whisper-small.en"
ASR_REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    here = Path(__file__).parent
    golden = {c["text"] for c in json.loads((here / "golden.json").read_text())["cases"]}
    rows = [json.loads(line) for line in (here.parent / "data" / "libritts_test_clean.jsonl").read_text().splitlines()]
    pool = [r for r in rows if 5 <= len(r["text"].split()) <= 40 and r["text"] not in golden]
    picks = random.Random(20261121).sample(pool, N_TEXTS)

    m = Qwen3TTSModel.from_pretrained(args.weights, device_map="cpu", dtype=torch.float32)
    proc = WhisperProcessor.from_pretrained(ASR_MODEL, revision=ASR_REVISION)
    asr = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, revision=ASR_REVISION).eval()
    norm = proc.tokenizer.normalize

    total_s = total_chars = errs = words = 0
    per = []
    for i, r in enumerate(picks):
        durs = []
        for k in range(REPEATS):
            torch.manual_seed(5000 + REPEATS * i + k)
            wavs, sr = m.generate_custom_voice(text=r["text"], language="English", speaker="Ryan")
            wav = np.asarray(wavs[0], dtype=np.float32)
            pcm = np.clip(np.round(wav * 32767), -32768, 32767).astype(np.int16)
            dur = len(pcm) / sr
            durs.append(dur)
            total_s += dur
            total_chars += len(r["text"])
            wav16 = scipy.signal.resample_poly(pcm.astype(np.float32) / 32768.0, 2, 3).astype(np.float32)
            feats = proc(wav16, sampling_rate=16000, return_tensors="pt").input_features
            with torch.inference_mode():
                hyp = proc.batch_decode(asr.generate(feats, max_new_tokens=440), skip_special_tokens=True)[0]
            e, n = word_errors(norm(r["text"]), norm(hyp))
            errs, words = errs + e, words + n
            print(f"{r['id']} {dur:.2f}s errs {e}/{n}: {hyp[:80]!r}", flush=True)
        per.append({"id": r["id"], "text": r["text"], "reference_seconds": durs})
    out = {
        "texts": per,
        "audio_seconds_per_char": total_s / total_chars,
        "wer": errs / words,
        "repeats": REPEATS,
        "asr": f"{ASR_MODEL}@{ASR_REVISION}",
        "reference": "qwen-tts 0.1.1, transformers 4.57.3, torch 2.15.0+cpu, FP32 on CPU",
    }
    (here / "sampled_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("audio_seconds_per_char", "wer")}))


if __name__ == "__main__":
    main()
