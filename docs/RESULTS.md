# Results

## Headline

| | |
|---|---|
| **Goodput** | **26.54 req/s**, the mean of 2 full runs of the protocol (26.51 and 26.57) |
| Offered rate at peak goodput | ~34 req/s |
| Hardware | 1 × AWS Trainium2 chip (trn2.3xlarge: 4 logical NeuronCores at LNC=2, 96 GB HBM; 12 vCPUs) |
| Model | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice @ 0c0e305, BF16 (FP32 where the reference needs it), no quantization |
| For scale | VoxServe on one H100: 47 req/s (as reported in the VoxServe paper) |

The served build passes the correctness check (`eval/check.py`):

| Gate | Measured | Limit |
|---|---|---|
| Score: teacher-forced logprobs vs the FP32 CPU reference, 10 utterances, all 16 codebooks | mean \|Δlogprob\| 0.034 | mean ≤ 0.08 (a BF16 CPU run of the reference: 0.04) |
| Codec: streamed decode of the reference's codes vs the reference decode | worst SNR 41.5 dB | ≥ 15 dB per utterance, ≥ 10 dB per 10-frame window |
| Sampled: 16 LibriTTS texts, Whisper small.en | WER 0.014 | ≤ max(2 × ref, ref + 0.05); reference 0.026 |

## Methodology

`eval/bench.py` implements the main evaluation of VoxServe v2 (arXiv:2602.00269v2, §4 and
Appendix B).

**Workload**

- The texts are the normalized texts of the 4,837 LibriTTS test-clean utterances (1-71 words,
  18.7 on average), in a random order drawn per seed.
- Every request uses speaker Ryan, English, and the checkpoint's sampling defaults.

**Load**

The benchmark runs three arrival seeds (0, 1, 2) back to back against the same server. Each seed
proceeds as follows:

1. A discarded burst of 64 concurrent requests, then 4 sequential warm-up requests.
2. Offered rates of 10, 12, 14, ... req/s. Each rate is a 60 s window of Poisson arrivals on an
   absolute schedule, so a slow server does not lower the offered load.
3. After each window, the client waits until every request finishes or reaches a 900 s timeout.

**SLO.** A request meets the SLO when all of these hold:

- it completes;
- its TTFA (send to first audio byte, including HTTP and queueing) is ≤ 200 ms;
- a player that starts at the first byte, and pauses whenever the next audio has not arrived,
  pauses < 10 ms in total.

Rejected (503) and failed requests count as misses.

**Score**

- **goodput** = max over offered rates r of (SLO-met requests at r, summed over the 3 seeds) /
  (3 × 60 s).
- An integrity guard fails the run if the audio seconds per text character fall below 0.75 of the
  reference's, so a server cannot score by saying less.

**Execution.** Each measurement built the server image from a clean checkout of the exact commit
and ran it on a dedicated trn2.3xlarge. The load generator ran in a separate container on the same
host.

### Deviation from the paper's protocol

The paper sweeps from 2 req/s and stops a seed only when fewer than half of the requests receive
audio within 1 s. We changed two things to cut measurement time:

- **Start rate.** The sweep starts at **10 req/s**. Rates below that are far under this system's
  capacity and can never set the maximum.
- **Plateau stop.** A seed also stops after **3 consecutive rates without a new best** SLO-met
  count.

The paper's TTFA stop rule is kept. Each rate keeps its own arrival seed, so the arrivals at any
given rate are identical under both protocols. Re-measuring the same code under both protocols
gave 22.54 req/s (old) and 22.48 req/s (shortened), which is within noise.

### Noise

- A single measurement is one full run of the protocol, which takes 1-2 hours.
- Repeated measurements of the same commit differ by about **5% relative**.
- The two samples of the final build differ by **±0.08 req/s** or less around their mean.
- Under this noise, a change has to gain roughly 10-14% before one or two samples per side can
  call it better.

## History

Each line is one promoted step of the optimization campaign. Every step was measured on the device
and reviewed independently before promotion. The first nine rows were measured under the original
protocol (sweep from 2 req/s) and the last two under the shortened one.

| # | Goodput (req/s) | What changed |
|---|---|---|
| 0 | 0.022 | First correct Trainium port. The model is ported from `qwen-tts`. Each frame runs as one fused device graph (talker step + 16 code-predictor positions with in-graph exact Gumbel-max sampling + frame embedding), after a one-call prefill. One stream at a time. |
| 1 | 3.68 | Continuous batching: 3 LM replicas (cores 0-2) × 16 rows, each code drawn on the host between small batched graphs (exact sampling). The streaming codec is moved to a NeuronCore (core 3) as a batched slot decoder. |
| 2 | 7.08 | LM batch buckets of 16 and 32. The codec scheduler becomes earliest-deadline-first instead of round-robin ticks. |
| 3 | 8.48 | The codec call is split into a ring-window transformer graph and a DAC graph. Codec state stays on the device, in place and indexed by slot (72-column K/V ring, per-slot conv contexts). |
| 4 | 11.09 | The codec's DAC residual stacks run as one fused NKI kernel. |
| 5 | 13.02 | Speculate-and-verify code-predictor chain: one device graph proposes codes 1-15, and the host verifies them exactly and fixes them up. LM batches of up to 64. |
| 6 | 17.61 | Integration of several codec and host changes: (a) DAC head graph plus a fused NKI DAC tail; (b) a codec delivery thread and a 0.5 ms GIL switch interval; (c) `gc.freeze()` after startup; (d) a 64-stream codec bucket with the head padded to the next bucket; (e) paged KV rows for streams past 512 positions. |
| 7 | 20.02 | NKI ring attention for the codec transformer. A codec-capacity admission gate (token bucket, length-neutral) keeps goodput from collapsing past saturation. |
| 8 | 22.54 | The talker launch overlaps host verification. First audio is held to submit + 150 ms with release-anchored codec deadlines. PCM is delivered asynchronously, one event-loop callback per codec call. |
| 9 | 22.48 | The same code, re-baselined under the shortened protocol. |
| 10 | **26.54** | A deadline-feasibility admission gate (length-neutral) replaces the token bucket: 26.51 and 26.57 over 2 samples. |

Notes:

- **Step 5** was 13.15 req/s over 2 samples when step 6 was compared against it.
- **A rejected gate variant.** An earlier variant of the step-7 gate measured 25.74 req/s, but its
  decision depended on the request's text length, so it rejected long utterances preferentially.
  It was not promoted. The promoted gates read only server state (streams, deadlines, call
  timings), and the final build's admitted and rejected requests have the same mean length
  (104.1 vs 104.3 characters).
- **Abandoned plans.** The campaign tried 44 plans over ~78 hours and 164 hardware jobs. Many
  were rejected or parked because they did not measure a gain. Among them were co-locating an LM
  replica with the codec, multi-frame codec calls, and a deeper codec pipeline.

## Where the time goes, and future work

- **The codec under host contention.** The codec core decodes ~3400 frames/s on its own but
  ~2300 frames/s under full serving load. The gap is host CPU and GIL contention between the
  codec thread, PCM delivery, HTTP, and the replicas' host work. Moving delivery and HTTP off the
  codec process's GIL is the most promising next step.
- **Slot-free codec calls.** This unmerged variant also keeps per-context DAC head state, and it
  measured 28.33 req/s over 2 samples (28.39 and 28.27). That is within the noise of 26.54, so it
  was not promoted. It is a candidate for future work, pending more samples.
- **LM admission cost.** At the workload's churn, prefill and admission add roughly 11-16 ms per
  batch step.
- **NKI codec head.** The DAC head still runs as XLA code. An NKI kernel for it was prototyped,
  off by default, and not part of this release.
