"""Build golden/golden.json and golden/golden_pcm.npz: the frozen CPU reference for the check.

Run on CPU with the official qwen-tts package (0.1.1; transformers 4.57.3, torch 2.15.0+cpu)
and the checkpoint Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice at revision 0c0e305 (no accelerator):

    python make_golden.py --weights <checkpoint dir>

For each case (speaker Ryan, language English, no instruct):

1. The reference samples one utterance in FP32 with the checkpoint's own sampling defaults and a
   fixed torch seed, at most MAX_FRAMES frames: `codes`, 16 per frame (codebook 0 from the
   talker, 1-15 from the code predictor), and whether it emitted EOS (`eos`).
2. Teacher forcing in FP32 over exactly those codes gives `logprobs`: per frame, for each of the
   16 codebooks, the log-softmax of the model's raw logits (no temperature, top-k, repetition
   penalty or token suppression) at the given code; codebook 0 over the talker's codec head
   (3072 entries), codebooks 1-15 over the code predictor's heads (2048 entries). Plus
   `eos_logprob`, codebook 0's log-probability of EOS (2150) right after the last frame.
3. The same teacher forcing with the model in BF16 calibrates the gate's tolerance
   (`calibration.bf16_*`): the largest |delta logprob| a correct BF16 implementation shows.
4. The codec: the reference decode of `codes` (qwen-tts speech_tokenizer.decode, FP32, exact for
   these lengths) as 16-bit PCM in golden_pcm.npz, and the SNR of a BF16 decode and of a
   streaming decode in 10-frame chunks with 25 frames of left context against it (whole
   utterance and the worst 10-frame window), which calibrates the codec gate.

The texts are LibriTTS test-clean utterances at spread word counts plus two extra sentences.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from transformers import LogitsProcessor, LogitsProcessorList

MAX_FRAMES = 64
SPEAKER = "Ryan"
LANGUAGE = "English"
SAMPLES_PER_FRAME = 1920
N_LIBRITTS = 8
EXTRA_TEXTS = [
    "Yes.",
    "On March 3rd, 1999, Dr. Smith paid $4,250 for 17 antique clocks at 221B Baker Street.",
]


def pick_texts(data: Path) -> list[dict]:
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    rows.sort(key=lambda r: (len(r["text"].split()), r["id"]))
    picks = [rows[int((i + 0.5) * len(rows) / N_LIBRITTS)] for i in range(N_LIBRITTS)]
    cases = [{"name": f"libritts-{r['id']}", "text": r["text"]} for r in picks]
    cases += [{"name": f"extra-{i}", "text": t} for i, t in enumerate(EXTRA_TEXTS)]
    return cases


def prompt_ids(m: Qwen3TTSModel, text: str):
    return m._tokenize_texts([m._build_assistant_text(text)])


def sample(m: Qwen3TTSModel, text: str, seed: int):
    torch.manual_seed(seed)
    with torch.inference_mode():
        codes, _ = m.model.generate(
            input_ids=prompt_ids(m, text),
            languages=[LANGUAGE],
            speakers=[SPEAKER],
            non_streaming_mode=False,
            max_new_tokens=MAX_FRAMES,
            **{k: v for k, v in m.generate_defaults.items() if k != "max_new_tokens"},
        )
    return codes[0]


class Force(LogitsProcessor):
    def __init__(self, targets: list[int]):
        self.targets = targets
        self.step = 0

    def __call__(self, input_ids, scores):
        t = self.targets[min(self.step, len(self.targets) - 1)]
        self.step += 1
        forced = torch.full_like(scores, float("-inf"))
        forced[:, t] = 0.0
        return forced


def teacher_forced(m: Qwen3TTSModel, text: str, codes: torch.Tensor) -> tuple[np.ndarray, float]:
    """Raw-logit log-probabilities of `codes` [T, 16] and of EOS after the last frame."""
    talker = m.model.talker
    eos_id = m.model.config.talker_config.codec_eos_token_id
    T = codes.shape[0]
    targets = codes[:, 0].tolist() + [eos_id]
    orig_gen, orig_pred = talker.generate, talker.code_predictor.generate
    captured = {}
    calls = {"n": 0}

    def gen(*args, **kwargs):
        kwargs.update(
            logits_processor=LogitsProcessorList([Force(targets)]),
            output_logits=True,
            max_new_tokens=T + 1,
            min_new_tokens=0,
        )
        captured["r"] = orig_gen(*args, **kwargs)
        return captured["r"]

    def pred(*args, **kwargs):
        k = calls["n"]
        calls["n"] += 1
        return SimpleNamespace(sequences=codes[k : k + 1, 1:].to(talker.device))

    talker.generate, talker.code_predictor.generate = gen, pred
    try:
        with torch.inference_mode():
            got, hidden = m.model.generate(
                input_ids=prompt_ids(m, text),
                languages=[LANGUAGE],
                speakers=[SPEAKER],
                non_streaming_mode=False,
                do_sample=False,
                subtalker_dosample=False,
                repetition_penalty=1.0,
                max_new_tokens=T + 1,
            )
            got, hidden = got[0], hidden[0]
            assert torch.equal(got.cpu(), codes.cpu()), "teacher forcing did not reproduce the codes"
            raw = torch.stack([captured["r"].logits[t][0].float() for t in range(T + 1)])
            lp0 = torch.log_softmax(raw, dim=-1)
            first = lp0[torch.arange(T), codes[:, 0]]
            eos = float(lp0[T, eos_id])
            sub, _ = talker.forward_sub_talker_finetune(codes.to(talker.device), hidden)
            lps = torch.log_softmax(sub.float(), dim=-1)  # [T, 15, 2048]
            rest = lps.gather(-1, codes[:, 1:].unsqueeze(-1).to(lps.device)).squeeze(-1)
    finally:
        talker.generate, talker.code_predictor.generate = orig_gen, orig_pred
    return torch.cat([first.unsqueeze(1), rest], dim=1).cpu().numpy(), eos


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = ref.astype(np.float64), test[: len(ref)].astype(np.float64)
    noise = np.sum((ref - test) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.sum(ref**2) / noise))


def worst_window_snr(ref: np.ndarray, test: np.ndarray, frames: int = 10) -> float:
    w = frames * SAMPLES_PER_FRAME
    vals = []
    for s in range(0, len(ref), w):
        r = ref[s : s + w]
        if np.sqrt(np.mean(r.astype(np.float64) ** 2)) < 1e-3:  # near-silent window
            continue
        vals.append(snr_db(r, test[s : s + w]))
    return min(vals) if vals else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    cases = pick_texts(Path(__file__).parent.parent / "data" / "libritts_test_clean.jsonl")

    m32 = Qwen3TTSModel.from_pretrained(args.weights, device_map="cpu", dtype=torch.float32)
    m16 = Qwen3TTSModel.from_pretrained(args.weights, device_map="cpu", dtype=torch.bfloat16)
    eos_id = m32.model.config.talker_config.codec_eos_token_id
    dec32 = m32.model.speech_tokenizer.model.decoder
    dec16 = copy.deepcopy(dec32).to(torch.bfloat16)

    pcm = {}
    for i, case in enumerate(cases):
        codes = sample(m32, case["text"], seed=1000 + i)
        T = codes.shape[0]
        lp32, eos32 = teacher_forced(m32, case["text"], codes)
        lp16, eos16 = teacher_forced(m16, case["text"], codes)
        d = np.abs(lp32 - lp16)

        wavs, sr = m32.model.speech_tokenizer.decode([{"audio_codes": codes}])
        assert sr == 24000
        wav = np.asarray(wavs[0], dtype=np.float32)
        assert len(wav) == T * SAMPLES_PER_FRAME
        pcm[case["name"]] = np.clip(np.round(wav * 32767), -32768, 32767).astype(np.int16)
        with torch.inference_mode():
            c = codes.T.unsqueeze(0)
            w16 = dec16(c.clone()).squeeze().float().numpy()
            ws = dec32.chunked_decode(c, chunk_size=10, left_context_size=25).squeeze().numpy()
        case.update(
            seed=1000 + i,
            frames=T,
            eos=bool(T < MAX_FRAMES - 1),  # at the cap the last sampled code 0 has no frame
            codes=codes.tolist(),
            logprobs=np.round(lp32, 5).tolist(),
            eos_logprob=round(eos32, 5),
            calibration={
                "bf16_mean_abs_dlogprob": float(d.mean()),
                "bf16_p99_abs_dlogprob": float(np.quantile(d, 0.99)),
                "bf16_max_abs_dlogprob": float(d.max()),
                "bf16_eos_abs_dlogprob": abs(eos32 - eos16),
                "bf16_decode_snr_db": snr_db(wav, w16),
                "bf16_decode_worst_window_snr_db": worst_window_snr(wav, w16),
                "stream10_ctx25_snr_db": snr_db(wav, ws),
                "stream10_ctx25_worst_window_snr_db": worst_window_snr(wav, ws),
            },
        )
        print(case["name"], f"T={T} eos={case['eos']}", json.dumps(case["calibration"]), flush=True)

    meta = {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "speaker": SPEAKER,
        "language": LANGUAGE,
        "max_frames": MAX_FRAMES,
        "eos_id": eos_id,
        "sample_rate": 24000,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "reference": "qwen-tts 0.1.1, transformers 4.57.3, torch 2.15.0+cpu, FP32 on CPU",
    }
    (out / "golden.json").write_text(json.dumps({"meta": meta, "cases": cases}) + "\n")
    np.savez_compressed(out / "golden_pcm.npz", **pcm)


if __name__ == "__main__":
    main()
