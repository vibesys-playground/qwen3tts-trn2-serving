# Testing

## CPU unit tests

The tests in `tests/` run on an ordinary CPU and need no Neuron device. They need torch (the CPU
build), numpy, scipy, safetensors, transformers, fastapi and uvicorn. Run each file directly, with
`NEURON_RT_VISIBLE_CORES` unset:

```bash
QWEN3TTS_DEVICE=cpu QWEN3TTS_WEIGHTS=/data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    python tests/test_prefill_cpu.py
```

`test_admit_cpu.py`, `test_async_out_cpu.py`, `test_hold_cpu.py` and `test_pages_cpu.py` need no
weights. The other tests load the checkpoint and run the model eagerly on the CPU, so they are
slow.

These tests pass on the CPU:

- `test_admit_cpu`
- `test_async_out_cpu`
- `test_hold_cpu`
- `test_pages_cpu`
- `test_prefill_cpu`
- `test_frame_cpu`
- `test_codec` (worst SNR 92 dB)
- `test_score` (mean |Δlogprob| 0.040)

## Known test failures

These tests fail on the released code exactly as they did on the commit that was measured. None
of them is a failure of the serving path that was measured. That path's startup self-tests, and
the full check in `eval/`, pass on Trainium2.

- **`test_batch_cpu.py`, `test_spec_cpu.py`, `test_engine_cpu.py`, `test_http_cpu.py`.** The tests
  use small CPU batch buckets (`2,4`). With those, the replica's startup self-test reports "the
  batch never changed bucket with pages live". Every code it checks still matches; for example,
  `test_engine_cpu.py` checks 17,280 codes with 0 mismatches. The device self-test, with the
  default buckets, passes.
- **`test_codec_dev.py`.** The golden streams pass at 36.7 dB or more. The final assertion on
  random codes needs 25 dB on the CPU and gets 23.8 dB.
- **`test_codec_sched.py`.** Its stand-in `OldTickService` predates the scheduler's `backlog`
  attribute. The test's worker thread dies, and the test then hangs. Don't run it until the
  stand-in is updated.

## Device benchmarks

`benchmarks/` holds microbenchmarks that time device graphs and check their exactness.
`tools/device_load.py` starts the server and drives `/score`, bursts and Poisson load against it.
Both need a NeuronCore.

## End-to-end check and benchmark

See [eval/README.md](../eval/README.md).
