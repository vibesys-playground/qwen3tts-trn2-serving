# Configuration

The server is configured only through environment variables. The table below lists every
variable the engine reads, found with `grep -n environ engine/*.py`.

**The defaults are the measured configuration.** Change them only to experiment, and re-run the
check (`eval/check.py`) and the benchmark after any change.

The **Status** column means:

- **config**: you are expected to set it for your deployment.
- **tuning**: a performance knob; the default is the tuned value.
- **experimental**: a non-default code path that is kept in the code but was not used for the
  results. Some were controls in A/B measurements; some are alternatives that measured slower.
  These paths are not covered by the published numbers.
- **diagnostic**: logging or testing only.
- **safety**: a startup self-test threshold.

## Deployment

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_WEIGHTS` | `/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Checkpoint directory: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice at revision 0c0e305, holding `config.json`, `model.safetensors`, the tokenizer files and `speech_tokenizer/`. Set this to wherever you mount the checkpoint. | config |
| `NEURON_RT_VISIBLE_CORES` | (all) | NeuronCores the server may use, e.g. `0-3` or `0,1,2,3`. With ≥ 2 cores, the last one runs the codec and each of the others runs one LM replica process. With one core, the LM and the codec share it. | config |
| `VIBESYS_CACHE_DIR` | unset | Persistent directory for the compile caches. When set, the server sets the defaults `TORCH_NEURONX_NEFF_CACHE_DIR=<dir>/neff`, `TORCHINDUCTOR_CACHE_DIR=<dir>/inductor` and `TORCH_NEURONX_HLO_COMPILE_CACHE=1`, so a restart loads compiled graphs instead of compiling them. The name comes from the measurement harness. | config |
| `TORCH_NEURONX_DEBUG_DIR` | a fresh temp dir | Where the Neuron backend writes compiler logs. The server sets it if unset. | config |
| `QWEN3TTS_DEVICE` | unset (first NeuronCore) | `cpu` runs everything eagerly on the CPU, for the unit tests only. | diagnostic |
| `QWEN3TTS_STALL_S` | `2400` | Seconds a startup stage may make no progress before the server exits (code 3). | config |
| `QWEN3TTS_HTTP_THREADS` | `1024` | anyio worker-thread limit. Only the sync streaming path uses these threads. | tuning |

## Process layout and host threads

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_LM_PROCS` | `1` | `1`: with ≥ 2 visible cores, LM replicas are separate processes. `0`: one in-process replica on the first core. | experimental (`0`) |
| `QWEN3TTS_LM_THREADS` | `2` | torch CPU threads per replica process. | tuning |
| `QWEN3TTS_CPU_THREADS` | `4` | torch CPU threads in the server process (host codec quantizer, self-tests). | tuning |
| `QWEN3TTS_REPLICA_DEVICE` | `neuron:0` | Device string inside a replica process. `cpu` is for tests. | diagnostic |
| `QWEN3TTS_GC_THRESHOLD` | unset | Optional `gc.set_threshold` values, e.g. `50000,50,100`. The server already calls `gc.freeze()` after startup; this is a fallback knob. | experimental |

## Language model: batching and speculation (`engine/batch.py`, `engine/sampling.py`)

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_SPEC` | `1` | Speculative code-predictor chain (PredChain) with exact host verification. `0` uses PredFirst/PredNext with a host draw per code, i.e. 15 round trips per frame. | `0` experimental |
| `QWEN3TTS_BUCKETS` | `16,32,64` (`16,32` if SPEC=0) | Batch-size buckets per replica. Each step uses the smallest bucket that holds the live rows. | tuning |
| `QWEN3TTS_MAX_ADMIT` | `8` | Arrivals admitted into the batch per step boundary. | tuning |
| `QWEN3TTS_SHRINK_AFTER` | `12` | Steps at lower load before the batch moves to a smaller bucket. | tuning |
| `QWEN3TTS_CHAIN_PAD` | `32:64` (empty if SPEC=0) | `X:Y,...` runs bucket X's chain on the compiled size-Y chain with pad rows. Measured: the B=64 chain is faster than the B=32 one. | tuning |
| `QWEN3TTS_OVERLAP` | `1` | Launch the next talker step on the proposed codes while the host verifies, and relaunch if a code changes. `0` runs the plain order. | `0` experimental |
| `QWEN3TTS_SPEC_TALKER` | `0` | Older variant, used only when OVERLAP=0: a speculative talker launch kept when no fix-up was needed. | experimental |
| `QWEN3TTS_PAGES` | `1` | Paged KV rows: streams past position 511 borrow free batch rows. `0` moves long streams to the B=1 2048-slot path at position 512. | `0` experimental |
| `QWEN3TTS_PAGE_P` | `3` | Maximum pages per stream (3 reaches position 2043). | tuning |
| `QWEN3TTS_PAGE_BUCKETS` | `""` (every bucket) | Buckets that compile the paged talker (TalkerInP). | experimental |
| `QWEN3TTS_FREE_BISECT` | `12` | Bisection steps of the in-graph proposal sampler. Affects only how often a fix-up is needed, never the output. | tuning |
| `QWEN3TTS_FREE_ROUNDS` | `2` | Candidate rounds of the proposal sampler after bisection. Affects speed only. | tuning |
| `QWEN3TTS_TIMING` | `1` | Time a few batch steps per bucket at startup and log them. | diagnostic |

## Codec (`engine/codec_dev.py`, `engine/nki_xf.py`, `engine/server.py`)

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_CODEC_DEVICE` | `neuron:0` (the last visible core); with one core, the LM's device | Device of the batched codec decoder. | config |
| `QWEN3TTS_CODEC_GROUPS` | `2,4,8` | Call buckets in groups of 8 streams (16, 32, 64 streams per call). `2,4` caps calls at 32 streams. | tuning |
| `QWEN3TTS_CODEC_SLOTS` | `max(128, 64 × replicas)` | Streams that can hold codec state at once. | tuning |
| `QWEN3TTS_CODEC_TTFA_S` | `0.2` | First-frame deadline after arrival, in seconds. | tuning |
| `QWEN3TTS_CODEC_WAIT_S` | `0.015` | Longest wait for a call to fill, in seconds. | tuning |
| `QWEN3TTS_CODEC_MARGIN_S` | `0.04` | Wait only if nothing is due within one call plus this margin, in seconds. | tuning |
| `QWEN3TTS_CODEC_DENSE` | `24` | Active slots that make a call worth taking without waiting. | tuning |
| `QWEN3TTS_CODEC_FPS0` | `1500` | Initial codec frames/s estimate, replaced on the device by a startup measurement. | tuning |
| `QWEN3TTS_CODEC_HEAD_MIN` | `4` | Minimum groups the DAC head graph runs at; smaller calls are padded. | tuning |
| `QWEN3TTS_CODEC_HEAD_PAD` | `1` | Pad small head calls up to HEAD_MIN groups. `0` turns padding off. | `0` experimental |
| `QWEN3TTS_CODEC_HEAD_MAX` | `0` | Split the DAC head into launches of this many groups. `0` means one launch per call. | experimental |
| `QWEN3TTS_CODEC_DAC_SPLIT` | `1` | Run the DAC as head + tail graphs. `0` runs a single DAC graph. | `0` experimental |
| `QWEN3TTS_CODEC_NKI` | `1` | Fused NKI DAC kernels. `0` uses XLA code for the DAC tail. | `0` experimental |
| `QWEN3TTS_CODEC_XF_ATTN` | `nki` | Codec transformer ring attention: `nki` (NKI kernel); `mm`, `vec` and `vecf` (XLA formulations); `none` (wrong output, timing only, refused unless the next variable is set). | non-`nki` values experimental |
| `QWEN3TTS_CODEC_XF_ATTN_TIMING_ONLY` | unset | `1` allows `XF_ATTN=none`, for benchmarks. Never serve with it. | diagnostic |
| `QWEN3TTS_XF_NKI_ENGINE` | `vector` | Engine for the ring kernel's multiplies: `vector`, `gpsimd` or `split`. | non-default experimental |
| `QWEN3TTS_CODEC_PIPE` | `1` | Delivery thread between the codec thread and the streams. `0` delivers inline from the codec thread. | `0` experimental |
| `QWEN3TTS_ASYNC_DIRECT` | `0` | `1`: the codec thread hands PCM straight to the event loops, with no delivery thread. Measured slower. | experimental |
| `QWEN3TTS_CODEC_SWITCH_S` | `0.0005` | Process-wide `sys.setswitchinterval`, in seconds. | tuning |
| `QWEN3TTS_CODEC_STATS_S` | `10` | Interval of the `[codec]` stats log line, in seconds. | diagnostic |

## Streaming and first audio (`engine/server.py`)

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_HOLD_MS` | `150` | First-audio hold: a text `/generate` releases its first PCM at max(first PCM ready, submit + HOLD). Codec deadlines are anchored at that release. `0` disables the hold. | tuning (`0` experimental) |
| `QWEN3TTS_ASYNC_OUT` | `1` | Stream PCM from async generators on the event loop, fed by one callback per codec call. `0` uses sync generators on worker threads. | `0` experimental |

## Admission (`engine/admit.py`)

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_ADMIT` | `1` | Admission gate on. `0` admits everything, and past saturation every stream goes late. | `0` experimental |
| `QWEN3TTS_ADMIT_MODE` | `deadline` | `deadline`: the deadline-feasibility gate (see ARCHITECTURE.md §10). `bucket`: the older codec-frame token bucket. | `bucket` experimental |
| `QWEN3TTS_ADMIT_H` | `0.5` | Deadline gate horizon H, in seconds. | tuning |
| `QWEN3TTS_ADMIT_U` | `0.8` | Deadline gate utilization target U. | tuning |
| `QWEN3TTS_ADMIT_NMIN` | `64` | Both gates always admit while fewer than this many codec streams are open. | tuning |
| `QWEN3TTS_ADMIT_RHO` | `0.9` | Token bucket only: fraction of the measured codec frames/s that refills the bucket. | experimental |
| `QWEN3TTS_ADMIT_LOG_S` | `10` | Interval of the `[admit]` stats log line, in seconds. | diagnostic |

### The token-bucket gate (`QWEN3TTS_ADMIT_MODE=bucket`)

This was the first gate. It holds a bucket of codec frames that refills at `ρ × fps_cap` frames/s,
where `fps_cap` is the codec's measured frames per busy second. The bucket's depth is
`1.5 s × ρ × fps_cap`.

- An admitted request is charged `ceil(chars × f)` frames, where `f` is a decayed ratio of frames
  per character over finished requests.
- When the request finishes, the estimate is replaced by the actual frame count.
- A request is rejected only when all three hold: the bucket is empty, the codec backlog exceeds
  0.25 s of capacity, and at least `NMIN` streams are open.

The decision itself never reads the request's length. The deadline gate replaced it after
measuring higher: 22.08 vs 18.79 req/s goodput in an A/B at 30 req/s offered.

## Self-tests

| Variable | Default | Meaning | Status |
|---|---|---|---|
| `QWEN3TTS_CODEC_SELFTEST_FRAMES` | `80` on the device, `6` on the CPU | Frames per stream in the startup codec self-test. More than 72 wraps the key/value ring. | safety |
| `QWEN3TTS_CODEC_SCHED_SELFTEST` | `1` | Run the codec scheduler self-test at startup. | safety |
| `QWEN3TTS_NKI_SELFTEST_DB` | `35` | SNR bar in dB for the NKI DAC tail kernel against FP32 CPU. | safety |
| `QWEN3TTS_XF_NKI_SELFTEST_DB` | `50` | SNR bar in dB for the NKI ring-attention kernel against FP32 CPU. | safety |

Two self-test bars are constants rather than variables: the per-stream codec self-test requires
20 dB against the host FP32 decoder (`server.CODEC_SELFTEST_DB`), and the scheduler self-test
requires 30 dB.

## Variables read by the tools, not the server

- `tools/device_load.py` and the `benchmarks/` read their own knobs, such as `STEPS`, `ITERS`,
  `MODES` and `RATE`, which each script's docstring describes. `VIBESYS_ARTIFACTS_DIR` (optional)
  is where some of them write JSON results.
- The CPU tests set small buckets themselves (`QWEN3TTS_BUCKETS=2,4`,
  `QWEN3TTS_CODEC_GROUPS=1`).
