# Evaluation: correctness check and goodput benchmark

This directory holds the harness that produced the published numbers. Its logic is unchanged.
`OBJECTIVE.md` is the full specification the server was built against: the model contract, the
service contract, the correctness gates, the workload, and the rules.

Both scripts are HTTP clients. They run on the CPU, can run on the same host as the server or on
another machine, and need no Neuron device.

```
eval/
  OBJECTIVE.md        the specification (model, API, gates, workload, rules)
  check.py            correctness check: weights, score gate, codec gate, sampled gate
  bench.py            goodput benchmark (VoxServe v2 §4 sweep)
  client.py           the streaming client and player model shared by both
  Dockerfile          the image the harness ran in for the published numbers
  data/libritts_test_clean.jsonl   the 4,837 LibriTTS test-clean texts (CC BY 4.0)
  golden/             frozen CPU references produced by the Qwen3-TTS model through qwen-tts 0.1.1
    golden.json         10 utterances: sampled codes, teacher-forced FP32 logprobs, calibration
    golden_pcm.npz      the reference decode of those codes (16-bit PCM)
    sampled_reference.json  16 texts, reference audio seconds per character, reference WER
    make_golden.py, make_sampled_reference.py   how the references were made (CPU, qwen-tts)
  reference/weights.json   file sizes and sha256 of the pinned checkpoint
```

## Dependencies

- Python 3.12, `aiohttp` (3.13.5), `numpy` (2.5.4) and `scipy` (1.18.1).
- For the check's word error rate: CPU `torch` (2.15.0) and `transformers` (5.22.1).
- The check transcribes with **`openai/whisper-small.en`** at revision
  `e8727524f962ee844a7319d92be39ac1bd25655a`, loaded through `transformers`. Either let it
  download from the Hugging Face Hub, or pre-populate a cache and set `HF_HOME` and
  `HF_HUB_OFFLINE=1`.
- `bench.py` needs only `aiohttp` and `numpy`.

`Dockerfile` is the judge image the numbers were measured with. Its base,
`vibesys-evaluator-llm-serving`, was built locally and is not published. Any Python 3.12
environment with the packages above is equivalent. The Dockerfile also sets `HF_HOME` to a
read-only mount that held the Whisper cache; adjust that path to your setup.

```bash
python -m venv .venv && . .venv/bin/activate
pip install aiohttp numpy scipy transformers
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Pointing the scripts at a server

Both scripts read **`VIBESYS_SERVICE_URL`**, which defaults to `http://127.0.0.1:8000`.
`bench.py` also accepts `--url`. The variable names come from the harness the numbers were
measured with.

```bash
export VIBESYS_SERVICE_URL=http://127.0.0.1:8000
```

## The check: `python eval/check.py [--skip-weights]`

The check exits 0 only if every gate passes. It prints every failure.

### 1. Weights

The check verifies the checkpoint against `reference/weights.json`: every file's size, and the
sha256 of the small files.

It looks for the checkpoint at the fixed path
`/vibesys/mounts/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`, the mount point of the original
harness. Either make the checkpoint visible there:

```bash
sudo mkdir -p /vibesys/mounts/models
sudo ln -s /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice /vibesys/mounts/models/
```

or skip this step with `--skip-weights`, which runs only the server gates below.

### 2. Health

`GET /health` must return 2xx.

### 3. Score gate

The check sends `POST /score` for the ten golden utterances all at once, so they share LM batches.
Each utterance is 9 to 63 frames of codes that the reference itself sampled in FP32. The server's
teacher-forced log-probabilities, over all 16 codebooks plus EOS after the last frame, are
compared with the FP32 CPU reference.

| Statistic | Limit |
|---|---|
| mean \|Δ\| | ≤ 0.08 |
| p99 | ≤ 0.40 |
| max | ≤ 2.0 |
| \|Δ EOS\| | ≤ 0.5 |

A BF16 CPU run of the reference shows mean 0.04, p99 0.15 and max 0.22 (`calibration` in
`golden.json`). **This server: mean 0.034.**

### 4. Codec gate

The check sends `POST /generate` with each golden utterance's `codes`, all at once. The server
streams its decode through the serving codec path. The output must be exactly 1920 samples per
frame, and the SNR against `golden_pcm.npz` must be:

- ≥ 15 dB over each utterance;
- ≥ 10 dB in every 10-frame window that is not silent.

**This server: worst SNR 41.5 dB.**

### 5. Sampled gate

The check sends `POST /generate` with real sampling on the 16 texts of `sampled_reference.json`,
all at once. One of the texts is sent twice. The gate requires:

- every request completes;
- each duration is 0.4 to 2.5 times the reference's seconds per character for its text;
- the two copies of the duplicated text give different audio (the server samples, and does not
  decode greedily or cache);
- the Whisper WER of the set, after Whisper's English normalizer, is ≤ max(2 × reference WER,
  reference WER + 0.05). The reference WER is 0.026.

Audio is resampled from 24 kHz to 16 kHz with `scipy.signal.resample_poly` before Whisper.
**This server: WER 0.014.**

## The benchmark: `python eval/bench.py --output-json goodput.json`

`bench.py` replays VoxServe v2's open-loop client (arXiv:2602.00269v2, §4 and Appendix B). Its
docstring has the exact rules.

**Sweep.** For each arrival seed in 0, 1 and 2, back to back against the same server:

1. A discarded burst of 64 concurrent requests, then 4 sequential warm-up requests.
2. Rates of 10, 12, 14, ... req/s. Each rate is a 60 s window of Poisson arrivals on an absolute
   schedule, seeded by (seed, rate). After each window, the client drains every request, with a
   900 s timeout.
3. The seed stops after the first rate that meets either condition:
   - fewer than half of the requests got audio within 1 s (the paper's rule);
   - it is the third consecutive rate with no new best SLO-met count (our shortening; see
     docs/RESULTS.md).

**Player model** (`client.py`). Every network chunk is timestamped. TTFA is the time from the
scheduled send to the first audio byte. The player starts at the first byte and accumulates a
pause whenever the next audio has not yet arrived. A request **meets the SLO** if all of these
hold:

- it completes with HTTP 200 and an even, non-zero number of PCM bytes;
- TTFA ≤ 200 ms;
- total pause < 10 ms.

**Score.**

- `goodput` = max over rates of (SLO-met requests at that rate, summed over seeds) / (3 × 60 s).
- Reported alongside it: `peak_rate`, the SLO capacity at 90% and 99% attainment, TTFA
  percentiles, and per-rate and per-seed tables.

**Integrity guard.** The run fails if the audio seconds per character, across counted requests,
fall below 0.75 × the reference's.

**Options and outputs.**

- `--seeds 0` runs a shorter trial on a single seed. It is not a valid score.
- If `VIBESYS_ARTIFACTS_DIR` is set, the benchmark also writes per-request logs
  (`requests-seed<N>.jsonl`) and `goodput.json` there.

A full run takes 1-2 hours against this server.

## Regenerating the golden references

`golden/make_golden.py --weights <checkpoint dir>` and
`golden/make_sampled_reference.py --weights <checkpoint dir>` rebuild the references on the CPU.
They need the official `qwen-tts` 0.1.1 package (transformers 4.57.3, torch 2.15.0+cpu) and the
checkpoint at revision 0c0e305. They run the Qwen3-TTS model itself in FP32, so the golden files
are outputs of that model.
