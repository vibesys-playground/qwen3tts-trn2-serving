# Qwen3-TTS on Trainium2

This is a streaming text-to-speech server for
[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) (the 1.7B CustomVoice
model) that runs on **a single AWS Trainium2 chip**.

You send it text over HTTP, and it streams back audio while the audio is still being generated.
With realistic traffic, it serves about **26.5 requests per second**. Each of those requests
starts playing within 200 ms and never pauses mid-sentence.

It runs the official checkpoint as is: no quantization, no pruning, and no change to how the model
samples. The speed comes from the serving system, not from doing less work.

## Results

We measured the server with the benchmark from the
[VoxServe paper](https://arxiv.org/abs/2602.00269). That benchmark asks: how many requests per
second can the server handle while each one still gets a good experience? A request counts as
good when:

- its first audio arrives within **200 ms**;
- playback never stalls for **10 ms or more** in total.

The benchmark replays LibriTTS sentences with random (Poisson) arrival times, at increasing rates.

| | |
|---|---|
| **Goodput** | **26.54 req/s** (average of two full runs: 26.51 and 26.57) |
| Busiest load tested at peak goodput | ~34 req/s offered |
| Hardware | one Trainium2 chip (trn2.3xlarge), BF16 |
| Accuracy vs. the FP32 reference | mean log-probability difference 0.034 (limit 0.08) |
| Audio fidelity of the streamed decode | worst case 41.5 dB SNR (limit 15 dB) |
| Intelligibility (Whisper word error rate) | 0.014 |

For comparison, the VoxServe paper reports 47 req/s for the same model on one NVIDIA H100.

Our benchmark differs from the paper in one respect, to save time. Its load sweep starts at
10 req/s instead of 2, and it stops once three rates in a row bring no improvement.
[docs/RESULTS.md](docs/RESULTS.md) has the full method and how the number grew from 0.02 to
26.54 req/s.

## How it works

Turning text into speech with Qwen3-TTS takes two models in sequence:

1. **The language model** reads the text and writes out audio *codes*: 16 numbers for every
   80 ms of sound.
2. **The codec** turns those codes into the actual waveform.

The chip has four cores, and the server splits the work across them:

```
                      HTTP requests
                            │
                   admission check ── reject (503) if the server can't keep up
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   language model     language model     language model      cores 0, 1, 2
   (up to 64 voices   (same)             (same)
    at a time)
         └──────────────────┼──────────────────┘
                            ▼
                    codec  (core 3)  ── decodes up to 64 streams per call,
                            │           most urgent first
                            ▼
                    streamed PCM audio
```

The main ideas:

- **Batching.** Each language-model copy works on up to 64 requests at once. New requests join a
  running batch as soon as there's room.
- **Speculate, then verify.** For each audio frame, the chip first guesses 15 of the 16 codes in a
  single pass. The CPU then checks each guess against an exact sample from the model. A wrong
  guess is corrected, so the output is exactly what the model would have produced. While the CPU
  checks, the chip has already started on the next step.
- **Streaming codec.** The codec keeps a running state for every stream. Each new 80 ms chunk of
  audio is therefore identical (up to rounding) to decoding the whole utterance at once.
  Custom NKI kernels make it fast.
- **Deadlines.** The codec always serves the streams closest to running out of audio first. Every
  stream's first audio is released 150 ms after the request arrives, which gives each one the
  same small buffer.
- **Saying no early.** When admitting another request would make the server miss deadlines it
  has already promised, it answers 503 at once rather than serving everyone late. The check never
  looks at how long the text is.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains each part in depth.

## Getting started

### What you need

- A **trn2.3xlarge**, or any Trainium2 host that exposes one chip as four logical NeuronCores
  (LNC=2).
- A container image with the **AWS Neuron SDK**: torch 2.14, torch-neuronx 2.14.3,
  neuronx-cc 2.28, NKI 0.7, Neuron runtime 2.35 and transformers 5.22.1, plus Python 3.12 and
  `uv` in `/opt/venv`. We measured with an internal image that isn't published. Pass your own
  with `--build-arg BASE_IMAGE=...`.
- About **4.5 GB** of disk for the model weights.

### 1. Download the model

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe \
    --local-dir /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 2. Build the image

```bash
docker build -f Dockerfile.neuron -t qwen3tts-trn2 \
    --build-arg BASE_IMAGE=<your Neuron SDK image> .
```

### 3. Start the server

```bash
docker run --rm --device /dev/neuron0 -p 8000:8000 \
    -e NEURON_RT_VISIBLE_CORES=0-3 \
    -v /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice:/models/Qwen3-TTS-12Hz-1.7B-CustomVoice:ro \
    -e QWEN3TTS_WEIGHTS=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    -v /data/neuron-cache:/cache -e VIBESYS_CACHE_DIR=/cache \
    -v "$PWD":/app -w /app \
    qwen3tts-trn2 python -m engine.server --host 0.0.0.0 --port 8000
```

The first start compiles every graph, which can take tens of minutes. The compiled graphs are
kept in the cache directory, so later starts are much faster. Then the server checks itself
against a CPU reference before it accepts traffic: it exits with an error if anything doesn't
match. It's ready when `GET /health` returns 200.

`VIBESYS_CACHE_DIR` has its name for historical reasons. It's just the compile cache location.

### 4. Make it talk

```bash
curl -sN http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"text": "Hello from Trainium.", "speaker": "Ryan", "language": "English", "stream": true}' \
    -o hello.pcm

# The audio is raw 16-bit mono PCM at 24 kHz. Play it, or turn it into a WAV file:
ffplay -f s16le -ar 24000 -ch_layout mono -autoexit hello.pcm   # older ffmpeg: use -ac 1
sox -t raw -r 24000 -e signed -b 16 -c 1 hello.pcm hello.wav
```

If the server is overloaded, it answers `503 {"error": "overloaded"}`.

### API

| Endpoint | What it does |
|---|---|
| `POST /generate` | `{"text", "speaker": "Ryan", "language": "English", "stream": true}` streams raw PCM (s16le, 24 kHz, mono) as it's generated. |
| `POST /score` | `{"text", "speaker", "language", "codes": [[16 ints], ...]}` returns the model's log-probability of each given code. It's used to check accuracy. |
| `GET /health` | Returns 200 once the server is compiled, self-tested and ready. |

Every setting is listed in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Reproducing the numbers

`eval/` contains the correctness check and the benchmark that produced the results above. They
run on any CPU machine that can reach the server:

```bash
pip install aiohttp numpy scipy torch transformers
export VIBESYS_SERVICE_URL=http://127.0.0.1:8000

python eval/check.py --skip-weights               # accuracy, audio fidelity, intelligibility
python eval/bench.py --output-json goodput.json   # the full load sweep, 1–2 hours
```

[eval/README.md](eval/README.md) explains what each test does.
[docs/TESTING.md](docs/TESTING.md) covers the unit tests, including a few known failures in the
CPU-only tests.

## What's in this repository

```
engine/        the server
  server.py      HTTP front end; starts the replicas and the codec
  replica.py     one language-model copy: load, compile, self-test, serve
  batch.py       batching, speculate-and-verify, paging for long requests
  model.py       the device graphs
  sampling.py    exact sampling and verification
  codec_dev.py   the streaming codec on the chip, and its scheduler
  codec.py       a plain CPU codec, used as the reference
  nki_*.py       custom Trainium kernels for the codec
  admit.py       the admission check
  prompt.py, runtime.py   prompt building, weights, the CPU reference path
eval/          correctness check, benchmark, test data
tests/         unit tests that run on a CPU
benchmarks/    microbenchmarks for the chip
tools/         a load generator
docs/          architecture, configuration, results, testing
```

## Limitations

- **Tuned for one setup.** One model, one chip size, speaker Ryan, English. There's no voice
  cloning, instruct text or non-streaming mode.
- **Slow first start.** Compiling every graph takes a while; the compile cache fixes later starts.
- **The codec is the bottleneck.** On its own, the codec core decodes about 3,400 audio frames
  per second, but only about 2,300 under full load. The difference comes from contention for the
  host CPU, not from the chip. Beyond ~34 req/s of offered load, the server turns away the
  excess.
- **Not hardened for production.** There's no authentication, no TLS and no request size limit.

## How it was built

An autonomous team of AI agents wrote and optimized this server. It used vibesys, driven by
Anthropic's Claude. Over about 78 hours, the agents tried 44 ideas and ran 164 jobs on the chip.
Every accepted change was measured on the hardware and reviewed independently first.
[docs/RESULTS.md](docs/RESULTS.md) lists each step.

## Acknowledgements

- **Qwen3-TTS** and its reference package `qwen-tts` (Apache-2.0) are by the Qwen team at Alibaba.
  This server's model, prompt and codec code is ported from `qwen-tts` 0.1.1.

  ```bibtex
  @article{Qwen3-TTS,
    title={Qwen3-TTS Technical Report},
    author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
    journal={arXiv preprint arXiv:2601.15621},
    year={2026}
  }
  ```

- **VoxServe** ([arXiv:2602.00269](https://arxiv.org/abs/2602.00269)) defined the goodput metric
  and the workload we used. Please cite it as listed on its arXiv page.
- **LibriTTS** (Zen et al., Interspeech 2019, CC BY 4.0) provides the test sentences in
  `eval/data/`.

  ```bibtex
  @inproceedings{zen2019libritts,
    title={LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
    author={Zen, Heiga and Dang, Viet and Clark, Rob and Zhang, Yu and Weiss, Ron J. and Jia, Ye and Chen, Zhifeng and Wu, Yonghui},
    booktitle={Interspeech},
    year={2019}
  }
  ```

- **Whisper** (`openai/whisper-small.en`) measures intelligibility in the check.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights aren't part of
this repository. Their authors license them separately, under Apache-2.0.
