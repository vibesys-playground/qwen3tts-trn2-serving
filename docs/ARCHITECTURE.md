# Architecture

This document explains how the server turns a `POST /generate` into streamed PCM on one Trainium2
chip, and why each design choice keeps the output identical in distribution to the reference
implementation (`qwen-tts` 0.1.1).

## Contents

1. [Model recap](#1-model-recap)
2. [Request lifecycle](#2-request-lifecycle)
3. [LM replicas and continuous batching](#3-lm-replicas-and-continuous-batching)
4. [Paged KV rows for long streams](#4-paged-kv-rows-for-long-streams)
5. [The speculative code-predictor chain and why it is exact](#5-the-speculative-code-predictor-chain-and-why-it-is-exact)
6. [Talker launch overlapped with host verification](#6-talker-launch-overlapped-with-host-verification)
7. [The streaming codec on a NeuronCore](#7-the-streaming-codec-on-a-neuroncore)
8. [Codec scheduling](#8-codec-scheduling)
9. [First-audio hold and stall avoidance](#9-first-audio-hold-and-stall-avoidance)
10. [The admission gate](#10-the-admission-gate)
11. [Threads, processes and the GIL](#11-threads-processes-and-the-gil)
12. [Numerics](#12-numerics)
13. [Startup self-tests](#13-startup-self-tests)

## 1. Model recap

Qwen3-TTS-12Hz-1.7B-CustomVoice generates speech as frames of 16 codes at 12.5 Hz (80 ms of
audio per frame).

- **Talker.** A 28-layer Qwen3 decoder (hidden size 2048). For each frame it predicts
  **codebook 0** over 3072 entries, which include the codec control tokens and EOS (id 2150).
- **Code predictor.** A 5-layer depth transformer that runs 16 positions per frame. Head `p-1`
  predicts **codebook p** (p = 1..15) over 2048 entries.
- **Frame embedding.** The sum of the 16 codes' embeddings plus the trailing text token's
  embedding. It is the talker's next input.
- **Codec** (`speech_tokenizer/`, Qwen3-TTS-Tokenizer-12Hz). It turns each frame into 1920
  samples at 24 kHz. Its stages are a quantizer, pre-conv, an 8-layer transformer with a 72-frame
  sliding window, two upsamplers, and a DAC-style conv decoder. Everything in it is causal.

Sampling follows the checkpoint's `generation_config.json`:

- temperature 0.9;
- top-k 50, with ties at the k-th value kept;
- top-p 1.0, which is the identity;
- repetition penalty 1.05 on codebook 0;
- the 1024 control tokens suppressed on codebook 0, except EOS;
- EOS masked before 2 frames.

Generation ends at EOS or when the prompt plus the generated frames reach 2048 positions.

## 2. Request lifecycle

```
client ──POST /generate──> event loop (server.py)
  1. admit(): DeadlineGate.decide(codec state)        -> 503 at once if infeasible
  2. submit(): open a codec stream (sink), bind it to the loop, set hold_until = now + 150 ms,
               pick the replica with the fewest outstanding requests, put ("gen", jid, text...)
               on its inbox
  3. StreamingResponse(agenerate_stream(job)) starts awaiting the sink's asyncio queue

replica process (batch.py)
  4. at the next step boundary: build the prompt on the host, prefill it into a free row
  5. every step: one frame for every live row -> ("ev", [(jid, "frame", codes[16]), ...])

server process
  6. _recv thread: job.sink.push(codes)          (frames queue on the codec stream)
  7. codec thread: earliest-deadline pick, device decode, int16 PCM
  8. delivery thread: one loop.call_soon_threadsafe per call -> per-stream asyncio queues
  9. event loop: the first chunk is released at max(first PCM, hold_until), later chunks as
                 they arrive
 10. EOS ("done") -> sink.finish() -> the stream drains, then ends; a client disconnect cancels
                 the job in its replica
```

Two other routes use the same engine:

- **`/score`** submits a teacher-forced job to the same replicas, so its rows share batches and
  graphs with generation. The job returns log-softmaxes of the raw logits.
- **`/generate` with `codes`** bypasses the LM. It pushes the given frames into a codec stream,
  so they take the same decode path as generated speech.

## 3. LM replicas and continuous batching

**Placement** (`server.py`, `replica.py`):

- With two or more visible NeuronCores, the last core belongs to the codec in the server process.
- Every other core runs one LM replica in its own spawned process, with
  `NEURON_RT_VISIBLE_CORES` set to that core. On trn2.3xlarge this gives replicas on cores 0-2
  and the codec on core 3.
- Replica 0 compiles every graph, or loads it from the cache. The others start afterwards, in
  parallel, and load the compiled graphs.
- With one visible core, the replica is a thread of the server process. It and the codec take
  turns on the core under a FIFO lock.

**Graphs** (`model.py`) are fixed-shape and compiled with `torch.compile(backend="neuron",
fullgraph=True, dynamic=False)`. They share one set of Parameters.

| Graph | Role |
|---|---|
| `PrefillInto` | Prefills up to 4 arriving prompts (10 positions each) straight into free rows of a bucket's KV cache, in place, and resets their per-row state |
| `PredChain` | All 16 code-predictor positions in one graph. It proposes codes 1-15 and returns each position's logits given the codes before it |
| `TalkerIn` / `TalkerInP` | The frame embedding plus one talker step at each row's own position, written into the cache in place (`TalkerInP`: with paged rows, §4) |
| `MoveRows` | Copies live rows between bucket caches when the batch changes size |
| `FrameStep` | The B=1 path: a whole frame (code 0 draw, 16 predictor positions, embedding, next talker step) in one graph with a 2048-slot cache |
| `PredFirst` / `PredNext` | The non-speculative path (`QWEN3TTS_SPEC=0`): one host round trip per code |

**Continuous batching** (`batch.py`, `BatchLM`):

- **Slot table.** `rows[i]` holds a job or nothing. The host keeps per-row presence bits (for
  the repetition penalty) and frame counts, because codebook 0 is drawn on the host.
- **Step boundaries.** At each boundary the batch admits up to `QWEN3TTS_MAX_ADMIT` (8) pending
  jobs, bounded by the free rows. Their prompts are embedded on the host and prefilled.
- **One step.** A step advances every live row by one frame. It then emits all events of the step
  (`frame`, `done`, `score`, `error`) to the server in one message.
- **Buckets.** Batch sizes are **16, 32 and 64**, each with its own 512-slot cache. Every step
  uses the smallest bucket that holds the live rows. The batch grows at once and shrinks after
  `QWEN3TTS_SHRINK_AFTER` (12) steps at lower load.
- **Chain padding.** The B=32 bucket runs its chain on the compiled B=64 chain with pad rows
  (`QWEN3TTS_CHAIN_PAD=32:64`). Measured on trn2, the B=64 chain graph is faster than the B=32
  one (20.7 vs 25.4 ms).
- **Leaving the batch.** A row leaves at EOS, at the position cap, or when its job is cancelled.

## 4. Paged KV rows for long streams

The batched caches have 512 slots per row, because a B=64 cache of 2048 slots would need DMA
offsets above 4 GiB. Long streams are handled like this (`QWEN3TTS_PAGES=1`, the default):

- **Borrowing a page.** At position 511 a stream keeps its row as the *leader*. It borrows a free
  row of the same batch as a 511-slot *page*. Pages are taken from the top of the table and
  admissions fill from the bottom, so the two rarely collide.
- **Attention across pages.** A page row attends with the leader's query. The leader combines its
  own unnormalized attention with its pages' by an FP32 log-sum-exp at the joint maximum. This is
  the same function as one softmax over the concatenated keys, up to rounding, and no KV is copied.
- **Reach.** With `QWEN3TTS_PAGE_P=3` pages a stream reaches position 511 + 3 × 511 − 1 = 2043,
  which covers the model's 2048-position cap.
- **Cost.** With paging on, every bucket's talker is `TalkerInP`. Measured on trn2, the B=64
  TalkerInP step takes 19.4 ms against 22.7 ms for TalkerIn, so paging costs nothing at the step
  level.
- **Fallback.** A stream moves to the B=1 `FrameStep` path (2048-slot cache) in three cases: it
  needs more pages than allowed, no row is free in the largest bucket, or the bucket compiles no
  paged talker. Its rows are copied there. Nothing is ever truncated.

## 5. The speculative code-predictor chain and why it is exact

A plain implementation needs 15 host↔device round trips per frame: run the predictor position,
copy the logits to the host, sample, send the code back. The speculative path
(`QWEN3TTS_SPEC=1`) does one launch:

1. **Code 0 on the host.** It is drawn by exact Gumbel-max: `argmax(score + g)` over the top-k
   masked, temperature-scaled scores, with `g ~ Gumbel(0, 1)`. The noise comes from a background
   thread with its own generator, so it is i.i.d. and independent of every logit.
2. **Proposals on the device.** `PredChain` proposes codes 1-15 inside the graph with a cheap
   sampler, `gumbel_pick_free`. It has no `topk` or gather. Instead it bisects a threshold
   (`QWEN3TTS_FREE_BISECT`, 12 steps) and then runs a few candidate rounds
   (`QWEN3TTS_FREE_ROUNDS`, 2). The sampler uses the host's noise for those positions.
3. **Logits back.** The chain also returns, for every position, the logits computed from the codes
   it fed in before that position.
4. **Host verification** (`gumbel_check`). A proposal at position p is accepted only if it equals
   the exact draw `gumbel_pick` (top-k 50, temperature 0.9, ties included) on position p's logits
   with the same noise.
5. **Fix-up.** At a row's first rejected position g, the host's exact draw replaces the proposal.
   The row then re-runs the **same compiled chain graph** with codes 1..g forced. A smaller
   bucket's graph is never used, so the logits round identically. Fix-ups repeat until every
   code is the host's exact draw on logits computed from its final prefix.

**Why the distribution is unchanged.** Each served code is `gumbel_pick` applied to the model's
logits for its true prefix, with fresh independent Gumbel noise. That is exactly a sample from
the softmax of those logits restricted to the top-k set, which is the reference's sampler. The
proposal sampler only decides how often a fix-up pass is needed, which affects speed and never
the output. The device's `topk` is avoided because it was observed to pair indices with the
wrong values at ties.

Teacher-forced `/score` rows run through the same chain with their given codes forced.

## 6. Talker launch overlapped with host verification

With `QWEN3TTS_OVERLAP=1` (the default), the next talker step is launched on the **proposed**
codes as soon as the chain's outputs reach the host. The device computes the talker step while
the host verifies and runs any fix-ups.

The launch cannot happen earlier: the device queue is in order, so the chain's device-to-host
copy would wait behind the talker.

If verification changes any code, the talker is launched again on the final codes. The relaunch
uses the same graph and the same cache slot, because the slot at `pos` is masked on read and
overwritten. The next step's host inputs (trailing text, positions, page tables) are built before
waiting on the talker's codebook-0 logits.

A startup self-test runs the overlapped order and the plain order on the same state and noise,
with every proposal deliberately corrupted. Codes, hidden states and logits must match
**bitwise**.

## 7. The streaming codec on a NeuronCore

`engine/codec.py` is the reference: a host FP32 port of the `qwen-tts` decoder that keeps every
causal state between calls, so any chunking equals one full decode. `engine/codec_dev.py` is the
serving version, batched on the codec's NeuronCore.

**Slots and calls**

- Each live stream owns a slot. Slots come in groups of 8.
- A call decodes one frame for every active row of G whole groups. The compiled buckets are
  G = 2, 4 and 8, i.e. 16, 32 and 64 streams per call.
- The slot table holds `max(128, 64 × replicas)` slots, so every LM row can have codec state at
  once.
- Per-row `active` and `fresh` masks leave idle rows bit-for-bit untouched and start new streams
  from zero state.

**The transformer graph (xf)**

- The quantizer (codebook lookup plus projection) runs on the host in FP32.
- The device graph runs pre-conv, the 8-layer transformer and the two upsamplers.
- Each slot's key/value window is a **72-column ring** in a single K table and a single V table.
  Frame p's post-RoPE key/value goes to column `p mod 72` with one in-place `scatter_` per
  tensor.
- A host-built bias masks every column that does not hold one of the previous 71 frames. This
  covers a stream's first frames and the column being overwritten.
- RoPE uses absolute frame positions.
- Conv contexts live in per-slot tables, read with `index_select` and written with `index_copy_`.

**NKI ring attention** (`engine/nki_xf.py`, `QWEN3TTS_CODEC_XF_ATTN=nki`)

- (row, head) pairs sit on the 128 partitions.
- Ring rows are gathered straight from the K/V tables by indirect DMA.
- q·K and p·V run as VectorE multiplies with free-axis reductions in FP32; the softmax
  exponential runs on ScalarE.
- The row's own new key/value is a 73rd column.
- The kernel is launched on a grid of 2, one program per physical core of the LNC=2 logical core.
- Measured on trn2, it cuts the B=64 xf graph from 6.73 to 2.52 ms. It is FP32-accurate (124 dB
  against FP32 CPU), whereas the XLA matmul form rounds q and p to BF16.

**The DAC graphs**

- The head graph covers decoder blocks 0-2.
- The tail is one fused NKI kernel (`engine/nki_dac.py`) covering:
  - the two transposed convs;
  - two residual stacks of three units (snake, dilated k=7 conv, snake, k=1 conv);
  - the final snake and conv_out.
- All tail intermediates stay in SBUF. Matmuls take BF16 inputs and accumulate in FP32 PSUM.
- The snake's argument is range-reduced before `sin`, because ScalarE's `sin` is only valid on
  [−π, π]. The reduction leaves sin² unchanged.
- Head calls smaller than `QWEN3TTS_CODEC_HEAD_MIN` (4 groups) are padded with scratch groups. At
  B=16 the head graph spills to HBM: 6.38 ms against 3.82 ms at B=32.

**Exactness.** The decoder is causal. Causal convs keep their left context, transposed convs keep
the previous input frame, and the transformer keeps its 72-frame window with absolute positions.
The streamed decode therefore equals the full decode up to BF16 rounding. The check's codec gate
measured a worst SNR of 41.5 dB against the reference decode; the limit is 15 dB.

## 8. Codec scheduling

One codec thread (`CodecService`, `codec_dev.py`) runs a loop.

**Each iteration**

1. **Housekeeping.** Reap finished or aborted streams, free their slots, and give free slots to
   streams with pending frames. Slots are handed out lowest-first, so live streams pack into few
   groups.
2. **Deadline-ordered pick.** Group pending streams by slot group. Order the groups by their
   earliest deadline, quantized to 20 ms, with ties going to the denser group. Take up to 8
   groups, i.e. up to **64 streams per call**, and decode the oldest pending frame of each.
3. **Wait to fill, if safe.** Wait up to `QWEN3TTS_CODEC_WAIT_S` (15 ms) for more frames when all
   three conditions hold:
   - no picked stream is new;
   - fewer than `QWEN3TTS_CODEC_DENSE` (24) slots would be active;
   - nothing is due within the last call's duration plus `QWEN3TTS_CODEC_MARGIN_S` (40 ms).
4. **Decode and convert.** Run the device call, then convert all rows' PCM to int16 in one array
   operation.

**Deadlines** (`CodecStream.deadline()`):

- Before the first frame: arrival + `QWEN3TTS_CODEC_TTFA_S` (0.2 s).
- After it: `anchor + 80 ms × frames delivered`, where the anchor is when the client's player
  starts, i.e. the release of the first PCM (§9). This is the moment a player that started at
  the first byte would run out of audio.

**Capacity tracking** for the admission gate:

- `backlog`: frames pushed but not yet decoded.
- `calls`: a ring of the last 64 calls, each recorded as (active slots, groups, cycle time).
- `call_full_s()`: an online least-squares fit of cycle = a + b × active over full-bucket calls,
  which predicts the cycle time of a full 64-slot call. A host slowdown therefore shows up within
  about 2 s.

## 9. First-audio hold and stall avoidance

The SLO requires TTFA ≤ 200 ms and less than 10 ms of total player pause over the whole
utterance.

**Hold** (`QWEN3TTS_HOLD_MS=150`). For a text `/generate`, the first PCM is released at
`max(first PCM ready, submit + 150 ms)`. Any PCM decoded during the hold is merged into that
first chunk.

- **Uniform buffer.** Every stream starts playing with the same buffer: whatever was decoded by
  submit + 150 ms. Without the hold, a stream whose first frame was decoded quickly would start
  playing with almost no audio queued, and a single late codec call could stall it.
- **Anchored deadlines.** Codec deadlines are measured from the release, not from the first
  decode. A stream's later frames therefore get the slack the hold created.
- **TTFA budget.** 150 ms is inside the 200 ms TTFA budget and leaves 50 ms for HTTP and the
  client.

The hold does not apply to the `codes` path or to `/score`.

**Stall avoidance.**

- The codec's earliest-deadline-first order always serves the stream whose player would run dry
  first.
- The admission gate (§10) keeps the total load within what the codec can deliver on time.
- The LM produces a frame for every live stream each step. While a step takes less than the
  80 ms a frame plays, streams run ahead of real time and their buffers grow.

## 10. The admission gate

When offered load exceeds codec capacity, an earliest-deadline-first scheduler makes *every*
stream late, and goodput collapses. The gate (`engine/admit.py`, `DeadlineGate`, the default)
rejects new text requests with 503 *before any LM or codec work*. Admitted requests therefore
stay on time, and goodput stays near its peak past saturation.

**Decision.** For an arriving request:

```
reject  iff  open_streams ≥ NMIN  and  urgent(H) + ALLOW > U × cap_H
```

where:

- `H` = 0.5 s (`QWEN3TTS_ADMIT_H`) is the horizon.
- `urgent(H)` (`CodecService.urgent`) sums, over every open stream, the frames it needs within
  H. Those are its frames whose deadline falls before `now + H`, counting every late frame, and at
  least `H / 80 ms` frames for a stream that is ahead of its player, since it needs them once its
  buffer runs down. A stream whose LM has finished counts at most its pending frames.
- `cap_H = H × 64 / per_call(64)` is how many frames the codec can decode in H, using the
  measured cycle time of a full 64-slot call (§8).
- `ALLOW = max(H / 80 ms, 1 + ⌊(H − 0.2 s) / 80 ms⌋)` = 6.25 frames is the new stream's own
  demand within H. It is a constant.
- `U` = 0.8 (`QWEN3TTS_ADMIT_U`) is the utilization target. A trn2 A/B against U = 0.9 gave the
  same goodput at 30 req/s offered, with higher attainment among admitted requests and half the
  codec backlog.
- `NMIN` = 64 (`QWEN3TTS_ADMIT_NMIN`). With fewer than 64 open streams the gate always admits.

**Length-neutral.** The decision reads only server state: streams, deadlines and call timings.
It never reads the request's text, its length, or any estimate derived from them. The gate cannot
raise goodput by turning away long utterances. `tests/test_admit_cpu.py` checks that the decision
does not change with the text length, and in simulation it reports the mean length of admitted
and rejected requests, which come out the same.

An older token-bucket gate (`QWEN3TTS_ADMIT_MODE=bucket`) is kept as an alternative. It is
documented in [CONFIGURATION.md](CONFIGURATION.md).

## 11. Threads, processes and the GIL

**Server process**

| Thread | Work |
|---|---|
| asyncio event loop (uvicorn) | HTTP parsing, admission, streaming responses |
| `_recv` | Replica events → codec streams |
| `codec` | The scheduler loop and device calls |
| `codec-out` | Delivery to the event loop |
| others | The watchdog during startup, a replica monitor, the admission stats logger |

The sync streaming path (`QWEN3TTS_ASYNC_OUT=0`) iterates its generators on anyio worker
threads, whose limit is raised to `QWEN3TTS_HTTP_THREADS` (1024). `/score` waits for its result
in `asyncio.to_thread`.

**Replica processes.** Each LM replica is a separate process, so its host work (code-0 sampling,
verification, input building) never contends for the server's GIL. A replica's main loop runs the
batch steps. A `_NoisePump` thread pre-draws the next step's Gumbel noise, because torch kernels
release the GIL. Each replica uses `QWEN3TTS_LM_THREADS` (2) torch threads, and the server uses
`QWEN3TTS_CPU_THREADS` (4).

**GIL measures**

- **Switch interval.** `sys.setswitchinterval(0.5 ms)` (`QWEN3TTS_CODEC_SWITCH_S`). The codec
  thread re-takes the GIL quickly instead of waiting up to the default 5 ms behind HTTP threads.
- **Async delivery** (`QWEN3TTS_ASYNC_OUT=1`). PCM goes to async generators on the event loop
  through **one** `call_soon_threadsafe` per loop per codec call, instead of one thread hop per
  chunk per stream. Measured on trn2, async delivery took the codec from ~2240-2460 to
  ~2740-2760 frames/s under synthetic HTTP load.
- **Delivery thread** (`QWEN3TTS_CODEC_PIPE=1`). PCM handoff leaves the codec thread's critical
  path. FIFO order guarantees that a stream's end marker never overtakes its PCM.
- **`gc.freeze()` after startup.** About 750k long-lived objects (module graphs, compile-time
  objects) move to the permanent generation. A full collection would otherwise walk them with the
  GIL held, which takes 0.3-0.35 s and freezes every thread in the server process.

Host contention remains the main limit. The codec core reaches ~3400 frames/s alone but ~2300
under full load.

## 12. Numerics

The server stays in BF16 throughout and uses FP32 where the reference does or where accuracy
needs it. Nothing is quantized.

**LM**

| Precision | What |
|---|---|
| BF16 | Weights, the residual stream, projections, the KV cache |
| FP32 | RMSNorm, attention scores, softmax and the weighted sum (including the paged log-sum-exp merge), logits, all sampling arithmetic, RoPE tables (computed in FP32, then cast) |

Frame and prompt embedding sums are computed in FP32 in the reference's order and rounded to
BF16 once.

**Codec**

| Precision | What |
|---|---|
| BF16 | Weights, matmul inputs, carried state (KV ring, conv contexts) |
| FP32 | Residual streams, norms, softmax, snake, the quantizer (host), PSUM accumulation, the output clamp |

**Results**

- **LM.** Teacher-forced log-probabilities differ from the FP32 reference by a mean of 0.034
  (the check's limit is 0.08). A BF16 CPU run of the reference itself shows 0.04.
- **Codec.** The worst SNR is 41.5 dB against the FP32 reference decode.

## 13. Startup self-tests

`/health` answers only after every self-test has passed. A failure exits the process.

**Per replica** (`replica.py`), through the real scheduler:

- Teacher-forced score rows against host BF16 references (mean < 0.1, max < 1.5, EOS < 0.5).
- Traced sampled rows, each code recomputed on the host.
- Bucket growth and shrink, and migration to the B=1 path.
- The chain recheck on every step.
- Injected proposal corruption that forces fix-ups.
- Paged rows at a small split.
- The overlap-vs-plain bitwise comparison (§6).

**Codec**:

- The NKI ring attention against FP32 torch (≥ 50 dB).
- The NKI DAC tail against FP32 CPU (≥ 35 dB).
- Every slot against the host FP32 decoder over 80 frames (≥ 20 dB per stream). The ring wraps,
  one stream skips calls, and one slot is reused mid-run.
- A scheduler self-test: 40 streams pushed out of order against a plain-order decode (≥ 30 dB).
  A skipped or duplicated frame would give ~0 dB.
