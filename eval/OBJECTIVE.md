# Qwen3-TTS goodput on one AWS Trainium2 chip (VoxServe v2 protocol)

Build and optimize a streaming text-to-speech serving system for **Qwen3-TTS 12Hz 1.7B
CustomVoice** on **one AWS Trainium2 chip** (a trn2.3xlarge: one device, four logical
NeuronCores at LNC=2, 96 GB HBM; the host has 12 vCPUs and 124 GB RAM), and maximize its
**goodput** as defined by the main evaluation of VoxServe (arXiv 2602.00269v2, section 4 and
Appendix B), with the paper's workload. No server is provided: porting the model to Trainium and
serving it correctly is part of the task. The candidate lives in the project root; `.vibesys/`
holds trusted task inputs that no candidate changes.

## Hardware budget

The budget is the whole chip: the four NeuronCores vibesys-cli gives the server
(`NEURON_RT_VISIBLE_CORES`, `/dev/neuron0`) and their 96 GB of HBM. Use nothing else: no other
accelerator, machine, or hosted model. Host-side work (tokenization, scheduling, HTTP, PCM
packing) is allowed and its cost stays inside the measurement; the host is shared with the judge
container and the load generator, so do not pin every vCPU. The codec decoder runs on the chip
or the host, as the candidate chooses, inside this budget.

## Model contract

Serve the official checkpoint **`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at revision `0c0e305`**
(`reference/weights.json`), read from the operator's read-only mount `models` at
`/vibesys/mounts/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` (the check refuses a mount whose files
differ). It holds the talker (a 28-layer Qwen3 decoder, hidden 2048, that predicts codebook 0 of
each 80 ms frame), the code predictor (a 5-layer depth transformer that predicts codebooks 1-15
of the frame), and in `speech_tokenizer/` the 12.5 Hz codec (Qwen3-TTS-Tokenizer-12Hz, 16
codebooks of 2048) whose decoder turns codes into 24 kHz audio, 1920 samples per frame. The
decoder is causal (causal convolutions, a transformer with a 72-frame sliding window), so a
streaming decoder that keeps the needed left state reproduces the full decode exactly.
The official Python package `qwen-tts` 0.1.1 (`Qwen3TTSModel`) is the reference implementation:
port from it; never import it from `.vibesys/`.

Every request uses speaker **Ryan**, language **English**, no instruct text, the prompt the
reference builds (`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` with the
streaming text interleaving of `non_streaming_mode=False`), and the checkpoint's sampling
defaults (`generation_config.json`): temperature 0.9, top-k 50, top-p 1.0, repetition penalty
1.05 on codebook 0, the same temperature/top-k/top-p for the code predictor, the reference's
suppression of the 1024 control tokens except EOS on codebook 0, and at least two frames.
Generation ends at EOS or when prompt plus generated frames reach 2048 positions.

Compute the same function: keep weights and activations in BF16 or wider (FP32 where the
reference needs it), no quantization (FP8, INT8 or otherwise, of weights, activations, KV cache
or codec), no pruning, no layer or head skipping, no smaller or distilled model, no change to the
sampling distribution, the codebook count, the frame rate or the sample rate. Speculative or
parallel decoding schemes must leave the sampling distribution exactly the model's.

`Dockerfile.neuron` in the project root builds the server's image; its first `FROM` must stay
the Neuron SDK base image (`BASE_IMAGE` in `Dockerfile.neuron`; the manifest's `allowed_bases`): the Neuron SDK (torch 2.14 +
torch-neuronx 2.14.3, neuronx-cc 2.28, NKI 0.7, Neuron runtime 2.35, transformers 5.22.1, Python
3.12 in `/opt/venv`) and no weights. Add serving dependencies after it (the reference package
pins transformers 4.57.3; vendor or adapt its model code rather than fight the pin). Read the
weights from the mount; never copy them into the image, re-encode them on disk, or fetch others.

## Service contract

vibesys-cli builds the image from the candidate's commit and runs, from the checkout's root,
`python -m engine.server --host 127.0.0.1 --port <port>`. It must answer `GET /health` with 2xx
within 3600 s (weight load plus any Neuron compilation). Keep compiler caches in
`$VIBESYS_CACHE_DIR` (kept across commits of the same server image), so a later start loads
graphs instead of compiling them.

- `POST /generate`, JSON `{"text": str, "speaker": "Ryan", "language": "English", "stream": true}`:
  HTTP 200 and a body of raw 16-bit little-endian mono PCM at 24 kHz (no WAV header), streamed
  as the audio is generated, in chunks of the server's choosing, ending when the utterance ends.
  The server may reject a request (any non-200) when overloaded; a rejection is a miss.
  With `"codes": [[16 ints] x T]` (check only) it skips the language model and streams the decode
  of exactly those frames through the same streaming decode path: exactly `1920 x T` samples.
- `POST /score` (check only), JSON `{"text", "speaker", "language", "codes": [[16 ints] x T]}`:
  `{"logprobs": [[16 floats] x T], "eos_logprob": float}`, the teacher-forced log-probability of
  each given code: codebook 0 from the talker's codec head over all 3072 entries, codebooks 1-15
  from the code predictor over 2048 entries, each the log-softmax of the raw logits (no
  temperature, top-k, repetition penalty or suppression), and `eos_logprob`, codebook 0's
  log-probability of EOS (id 2150) after the last frame. It must run the same model, weights and
  numerics as `/generate`.

## Correctness

The check (`python /task/check.py`, against the server) confirms the weights, then:

1. **Scoring gate.** `/score` on the ten golden utterances of `golden/golden.json` (all sent at
   once, so they share batches): the reference's own FP32 sampled codes, 9 to 63 frames, scored
   against the FP32 CPU reference over all 16 codebooks. Mean |delta logprob| at most 0.08, p99
   at most 0.40, max at most 2.0, and |delta| of EOS at most 0.5. A BF16 CPU run of the reference
   shows mean 0.04, p99 0.15, max 0.22 and EOS 0.06 (`calibration` in golden.json).
2. **Codec gate.** `/generate` with each golden utterance's codes (all at once): exactly 1920
   samples per frame, SNR against the reference decode at least 15 dB over the utterance and at
   least 10 dB in every non-silent 10-frame window. A BF16 decode gives 32-37 dB (worst window
   25 dB); the reference's own 25-frame left-context chunking gives 21-27 dB (worst window 14 dB).
3. **Sampled gate.** `/generate` on 16 fixed LibriTTS utterances (`golden/sampled_reference.json`),
   all at once, with the real sampling: each completes with a duration of 0.4 to 2.5 times the
   reference's seconds per character for its text; one text sent twice must give different audio
   (sampling, not greedy decoding or a cache); and the Whisper (`openai/whisper-small.en`)
   word error rate of the set, after Whisper's English normalizer, stays within
   max(2 x, + 0.05) of the reference's own WER on the same texts.

`golden/make_golden.py` and `golden/make_sampled_reference.py` are how the reference was made
(CPU, qwen-tts 0.1.1, no accelerator). A server that fails the check is not measured.

## Workload and objective (VoxServe v2, section 4.1 and Appendix B)

`python /task/bench.py` replays the paper's open-loop client against the server:

- Texts: the normalized text of the **4,837 LibriTTS test-clean utterances**
  (`data/libritts_test_clean.jsonl`, 1-71 words, 18.7 on average), in a random order drawn per
  arrival seed.
- **Three arrival seeds** (0, 1, 2), each one sweep, back to back against the same server: a
  discarded burst of 64 concurrent requests and four uncounted warm-up requests, then offered
  rates **10, 12, 14, ... req/s**, each a **60 s window of Poisson arrivals** with a fixed seed
  on an absolute schedule (a slow server does not lower the offered load), after which the client
  waits until every request finishes or times out after 900 s. A sweep stops after a rate at
  which fewer than half the requests receive audio within 1 s (the paper's rule), or after 3
  consecutive rates none of which sets a new best count of SLO-met requests for that seed.
  This deviates from the paper's protocol (which starts at 2 req/s and has only the 1 s rule) to
  cut measurement time, as approved by the user on 2026-11-24: rates below 10 req/s are far
  under this system's capacity (goodput ~22 req/s), so they never set the max. Each rate keeps
  its own arrival seed, so its arrivals are unchanged; the SLO capacity diagnostics consider
  rates from 10 req/s up.
- A request **meets the SLO** if it completes, its **TTFA** (send to first audio byte, HTTP and
  queueing included) is **at most 200 ms**, and it **does not stall**: a player that starts at
  the first byte and pauses whenever the next audio has not arrived pauses less than 10 ms in
  total. Rejected and failed requests are misses.
- **Score: `goodput` (req/s, maximize)** = the maximum over offered rates of the requests
  meeting the SLO at that rate, summed over the three seeds, divided by 3 x 60 s (a seed whose
  sweep stopped earlier contributes nothing at the higher rates). One measure is one full run of
  the protocol (`samples = 1`), with a provisional 5% noise floor (`objectives.toml`).
  SLO capacity at 90% and 99% attainment, TTFA percentiles and per-rate tables are reported
  beside it as diagnostics (the artifacts' `goodput.json` and per-request logs).
- Integrity guard: over the counted requests, audio seconds per text character must be at least
  0.75 of the reference's; a server that says less than the text fails the sample.

For scale: on one H100, VoxServe's goodput peaks at 47 req/s for this model and workload
(SGLang-Omni 30, Nari Labs' engine 21, vLLM-Omni 6).

## Rules

Never special-case the golden or benchmark texts, detect the check or the benchmark, cache or
reuse audio or codes across requests, start speaking before the request arrives, drop or shorten
any part of the text, truncate generation before EOS, or change the speaker, language or sampling
parameters. `/score` and the `codes` path of `/generate` exist for the check and must share the
serving path's model and numerics. Use no hardware beyond the budget. Speed must come from the
serving system (batching, scheduling, kernels, graphs, overlap of the language model and the
codec, streaming), never from doing less work per request than the reference does.
