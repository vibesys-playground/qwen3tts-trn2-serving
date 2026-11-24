# qwen3tts-trn2-serving

A streaming text-to-speech server for
[Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
(revision `0c0e305`) on **one AWS Trainium2 chip**. The target is a trn2.3xlarge: one device
exposed as four logical NeuronCores at LNC=2, with 96 GB of HBM.

The server runs the official checkpoint and keeps the reference model's sampling distribution.
There is no quantization and no pruning, and sampling is unchanged: every speculated code is
verified exactly on the host. It reaches **26.54 req/s goodput** under the VoxServe v2 protocol
(TTFA ≤ 200 ms and no audible stall, LibriTTS test-clean texts, Poisson arrivals).

```
POST /generate  {"text": "...", "speaker": "Ryan", "language": "English", "stream": true}
                -> 200, raw 16-bit little-endian mono PCM at 24 kHz, streamed as it is generated
POST /score     {"text", "speaker", "language", "codes": [[16 ints] x T]}
                -> {"logprobs": [[16 floats] x T], "eos_logprob": float}   (teacher-forced)
GET  /health    -> {"ok": true} once every graph is compiled and every self-test has passed
```

## Results

| Metric | Value |
|---|---|
| **Goodput** (VoxServe v2, §4) | **26.54 req/s** (mean of 2 full runs: 26.51 and 26.57) |
| Offered rate at peak goodput | ~34 req/s |
| SLO | TTFA ≤ 200 ms and total player stall < 10 ms |
| Workload | LibriTTS test-clean (4,837 utterances), Poisson arrivals, 3 seeds, 60 s per rate |
| Score gate (teacher-forced logprobs vs the FP32 reference) | mean \|Δlogprob\| 0.034 (limit 0.08) |
| Codec gate (streamed decode vs the reference decode) | worst SNR 41.5 dB (limit 15 dB) |
| Sampled gate (Whisper small.en WER) | 0.014 |
| Hardware | 1 × Trainium2 chip (trn2.3xlarge), BF16, no quantization |

For scale, the VoxServe paper reports 47 req/s for this model and workload on one H100.

Our benchmark differs from the paper's protocol in one respect. The sweep starts at 10 req/s
instead of 2, and a seed stops after 3 rates with no new best; the paper's TTFA stop rule is
kept. [docs/RESULTS.md](docs/RESULTS.md) has the methodology, the noise, and the step-by-step
history from 0.022 to 26.54 req/s.

## Architecture

```
                         HTTP (FastAPI/uvicorn, asyncio event loop)
                                        |
                     admission gate (deadline feasibility, engine/admit.py)
                                        |  route to the replica with the fewest requests
          +-----------------------------+-----------------------------+
          v                             v                             v
  LM replica 0 (core 0)         LM replica 1 (core 1)         LM replica 2 (core 2)
  process, continuous batch     (same)                        (same)
  of up to 64 streams:
   host: code 0 (exact Gumbel-max)
   dev : PredChain proposes codes 1-15
   host: exact verify + fix-up   <-- overlaps -->  dev: talker step on the proposals
          |  one 16-code frame per stream per step (80 ms of audio)
          +-----------------------------+-----------------------------+
                                        v
                       codec service (core 3, server process)
            deadline-ordered pick of up to 64 streams per call
            ring-window transformer (NKI attention) + DAC head + fused NKI DAC tail
                                        |  1920 PCM samples per stream per call
                                        v
            delivery thread -> asyncio queues -> first audio held to submit + 150 ms
                                        |
                                        v
                               streamed s16le PCM
```

- **Language model.** The model has two parts. The talker is a 28-layer Qwen3 decoder that
  predicts codebook 0 of each 80 ms frame. The code predictor is a 5-layer depth transformer that
  predicts codebooks 1-15.
- **Replicas and batching.** Three LM replicas run, one per NeuronCore. Each replica
  continuously batches up to 64 streams in buckets of 16, 32 and 64, with a 512-slot KV cache per
  row. A stream that grows past 512 positions borrows free rows of its batch as paged KV.
- **Speculate and verify.** One device graph proposes all 15 code-predictor codes. The host
  checks each proposal against an exact Gumbel-max draw on the model's own logits with the same
  noise, and re-runs the chain from the first mismatch. The served codes are therefore exactly
  the model's samples.
- **Overlap.** The next talker step is launched on the proposals while the host verifies them.
- **Codec.** The streaming 12.5 Hz decoder runs on the fourth NeuronCore. It keeps the exact
  causal state of every stream (conv contexts and a 72-frame key/value ring), so the streamed
  audio equals the full decode up to BF16 rounding.
- **Scheduling.** The codec serves streams in earliest-deadline order. First audio is released
  at submit + 150 ms, which gives every stream the same playout buffer.
- **Admission.** A length-neutral gate answers 503 at once when the codec could not keep its
  deadlines.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers each part in detail.

## Requirements

- **Hardware.** One trn2.3xlarge, or any Trainium2 host that exposes one chip as four logical
  NeuronCores at LNC=2: three cores for the LM, one for the codec. The host has 12 vCPUs. The
  server uses 4 torch threads in its main process and 2 per replica.
- **Base image.** A container image with the AWS Neuron SDK in `/opt/venv`, Python 3.12 and
  `uv`:

  | Component | Version |
  |---|---|
  | torch | 2.14 |
  | torch-neuronx | 2.14.3 |
  | neuronx-cc | 2.28 |
  | NKI | 0.7 |
  | Neuron runtime | 2.35 |
  | transformers | 5.22.1 |

  The results were measured with an internal image of that content, whose tag is the default of
  `ARG BASE_IMAGE` in `Dockerfile.neuron`. That image is not published, so pass your own with
  `--build-arg BASE_IMAGE=...`.
- **Serving dependencies.** `Dockerfile.neuron` adds `fastapi==0.165.0` and `uvicorn==0.54.0`.
- **Weights.** About 4.5 GB on local disk, read-only.
- **Evaluation (optional).** The check and benchmark run on a separate CPU machine or container;
  see [eval/README.md](eval/README.md).

## Quickstart

### 1. Download the weights at the pinned revision

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe \
    --local-dir /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The server reads the checkpoint directory named by **`QWEN3TTS_WEIGHTS`**. That directory holds
`config.json`, `model.safetensors`, the tokenizer files and `speech_tokenizer/`. The weights are
never copied into the image.

### 2. Build the image

```bash
docker build -f Dockerfile.neuron -t qwen3tts-trn2 \
    --build-arg BASE_IMAGE=<your Neuron SDK image> .
```

### 3. Run the server

```bash
docker run --rm --device /dev/neuron0 -p 8000:8000 \
    -e NEURON_RT_VISIBLE_CORES=0-3 \
    -v /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice:/models/Qwen3-TTS-12Hz-1.7B-CustomVoice:ro \
    -e QWEN3TTS_WEIGHTS=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    -v /data/neuron-cache:/cache -e VIBESYS_CACHE_DIR=/cache \
    -v "$PWD":/app -w /app \
    qwen3tts-trn2 python -m engine.server --host 0.0.0.0 --port 8000
```

On startup the server compiles every graph and then runs self-tests against FP32 CPU references.
If any self-test fails, the server exits non-zero. The self-tests cover:

- the LM scores;
- the exactness of the speculative chain and of the overlap;
- the NKI kernels;
- the codec decode;
- the codec scheduler.

A cold start compiles many graphs and can take tens of minutes. If a startup stage makes no
progress for `QWEN3TTS_STALL_S` (2400 s), the server aborts.

`VIBESYS_CACHE_DIR` keeps the Neuron compile caches across restarts, so a warm start loads graphs
instead of compiling them. The variable's name is historical: the measurement harness supplied
that directory.

The server is ready when `GET /health` returns 200.

### 4. Synthesize speech

```bash
curl -sN http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"text": "Hello from Trainium.", "speaker": "Ryan", "language": "English", "stream": true}' \
    -o hello.pcm

# play the raw PCM (s16le, 24 kHz, mono)
ffplay -f s16le -ar 24000 -ch_layout mono -autoexit hello.pcm   # older ffmpeg: -ac 1
play -t raw -r 24000 -e signed -b 16 -c 1 hello.pcm              # sox
# or convert it to WAV
sox -t raw -r 24000 -e signed -b 16 -c 1 hello.pcm hello.wav
```

`speaker` and `language` default to `Ryan` and `English`, the only configuration that was
evaluated. When the admission gate rejects a request, the server answers HTTP 503
`{"error": "overloaded"}`.

## Evaluation

`eval/` holds the check and the benchmark that produced the numbers above. Only their location
has changed. [eval/README.md](eval/README.md) explains how they work.

```bash
pip install aiohttp numpy scipy torch transformers   # CPU is enough
export VIBESYS_SERVICE_URL=http://127.0.0.1:8000

python eval/check.py --skip-weights               # score, codec and sampled gates; exit 0 on pass
python eval/bench.py --output-json goodput.json   # the full 3-seed sweep (1-2 h)
```

Without `--skip-weights`, `check.py` also verifies the checkpoint files against
`eval/reference/weights.json`. It looks for them under the fixed mount path
`/vibesys/mounts/models/`; eval/README.md explains how to provide that path.

## Repository layout

```
engine/            the server
  server.py          HTTP front end, replica routing, codec startup, gc.freeze
  admit.py           admission gates (deadline feasibility, the default; token bucket)
  replica.py         one LM replica: load, compile, self-tests, serving loop
  batch.py           continuous batching, speculative chain + host verify, paged KV rows
  model.py           device graphs: talker/predictor steps, PredChain, TalkerIn(P), PrefillInto, ...
  runtime.py         weights, the B=1 2048-slot path, the host reference path
  sampling.py        exact Gumbel-max sampling, the proposal sampler, the verifier
  prompt.py          prompt construction and embeddings (host)
  codec.py           host FP32 streaming codec decoder (the reference for self-tests)
  codec_dev.py       batched device codec and the codec service (EDF scheduler, delivery)
  nki_xf.py          NKI kernel: ring-window attention of the codec transformer
  nki_dac.py         NKI kernels: the fused DAC residual stacks and tail
tests/             CPU unit tests (test_*.py); no Neuron device needed
benchmarks/        device benchmarks with exactness checks (need a NeuronCore)
tools/device_load.py  starts the server, then drives /score, bursts and Poisson load against it
eval/              the correctness check, the goodput benchmark, data and golden references
docs/              ARCHITECTURE.md, CONFIGURATION.md, RESULTS.md
Dockerfile.neuron  the server image
```

The CPU tests need torch (CPU), numpy, scipy, safetensors, transformers, fastapi and uvicorn. Run
each file directly, with `NEURON_RT_VISIBLE_CORES` unset:

```bash
QWEN3TTS_DEVICE=cpu QWEN3TTS_WEIGHTS=/data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice python tests/test_prefill_cpu.py
```

`test_admit_cpu.py`, `test_async_out_cpu.py`, `test_hold_cpu.py` and `test_pages_cpu.py` need no
weights. The other tests load the checkpoint and run the model eagerly on the CPU.

### Known test failures

These CPU tests fail on the released code exactly as they did on the measured commit. None of them is a failure of the measured serving path, whose startup self-tests and the full check pass on Trainium2.

- `test_batch_cpu.py`, `test_spec_cpu.py`, `test_engine_cpu.py` and `test_http_cpu.py`: with the tests' small CPU buckets (`2,4`), the replica's startup self-test reports "the batch never changed bucket with pages live". Every code it checks still matches (for example, 17,280 codes with 0 mismatches in `test_engine_cpu.py`). The device self-test with the default buckets passes.
- `test_codec_dev.py`: the golden streams pass at 36.7 dB or more. The final random-code assertion needs 25 dB on the CPU and gets 23.8 dB.
- `test_codec_sched.py`: its stand-in `OldTickService` predates the scheduler's `backlog` attribute, so the worker thread dies and the test hangs. Do not run it until the stand-in is updated.

Passing on the CPU: `test_admit_cpu`, `test_async_out_cpu`, `test_hold_cpu`, `test_pages_cpu`, `test_prefill_cpu`, `test_frame_cpu`, `test_codec` and `test_score`.

## Limitations

- **One configuration.** The server is built and tuned for one model and one chip size (4 logical
  cores at LNC=2). Only speaker Ryan in English was evaluated. There is no instruct text, voice
  cloning or non-streaming mode.
- **Long cold start.** Startup compiles every graph unless the compile cache is warm, and then
  runs the self-tests.
- **The codec is the bottleneck.** On its own, the codec core decodes ~3400 frames/s. Under full
  serving load it decodes ~2300 frames/s, because of host CPU and GIL contention with the HTTP
  path and the replicas' host work. Above ~34 req/s offered, the admission gate rejects the
  excess so that admitted streams stay on time.
- **Overload answers 503.** The goodput metric counts each rejection as a miss.
- **Unpublished base image.** The exact image the results were measured with is not public. A
  different Neuron SDK build may compile or perform differently.
- **Not production-hardened.** There is no authentication, no TLS and no request size limit.

## How it was built

An autonomous multi-agent optimization campaign wrote and optimized this server against the
check and benchmark in `eval/`. The campaign used vibesys, driven by Anthropic's Claude, and ran
44 plans and 164 hardware jobs over ~78 hours on one trn2.3xlarge. Every promoted change was
measured on the device and reviewed independently before it was accepted.
[docs/RESULTS.md](docs/RESULTS.md) lists each step.

## Acknowledgements and citations

- **Qwen3-TTS** and the `qwen-tts` reference package (Apache-2.0) are by the Qwen team, Alibaba
  Group. The engine's model, prompt and codec code is ported from `qwen-tts` 0.1.1.

  ```bibtex
  @article{Qwen3-TTS,
    title={Qwen3-TTS Technical Report},
    author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
    journal={arXiv preprint arXiv:2601.15621},
    year={2026}
  }
  ```

- **VoxServe** (arXiv:2602.00269v2) defines the goodput metric, the SLO and the workload. Please
  cite the paper as listed on its arXiv page.
- **LibriTTS** (Zen et al., Interspeech 2019, CC BY 4.0) supplies the test-clean texts in
  `eval/data/`.

  ```bibtex
  @inproceedings{zen2019libritts,
    title={LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
    author={Zen, Heiga and Dang, Viet and Clark, Rob and Zhang, Yu and Weiss, Ron J. and Jia, Ye and Chen, Zhifeng and Wu, Yonghui},
    booktitle={Interspeech},
    year={2019}
  }
  ```

- **Whisper** (`openai/whisper-small.en`) measures the word error rate in the check.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Qwen3-TTS weights are not
part of this repository; their authors license them separately under Apache-2.0.
