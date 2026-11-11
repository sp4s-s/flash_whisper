# Whisper on H100: vLLM vs PyTorch vs CUDA

Benchmarks `openai/whisper-large-v3-turbo` on one Modal H100 with three backends: vLLM 0.33.0, a plain PyTorch path, and a CUDA path (GPU log-mel, CUDA-graphed encoder and decoder, hand written kernels). FP16, batch 1, greedy decoding. It reports latency, time to first token, throughput and WER on 100 LibriSpeech clips.

## Results (run07, 30 s clip, median of 20)

| Backend | E2E | TTFT | E2E speedup | TTFT speedup |
|---|---:|---:|---:|---:|
| vLLM | 222.8 ms | 33.3 ms | 1.00x | 1.00x |
| PyTorch | 204.2 ms | 19.5 ms | 1.09x | 1.71x |
| CUDA | 44.0 ms | 10.8 ms | 5.06x | 3.08x |

WER is 1.31% on all three and token ids match. vLLM timings vary between runs, so compare within a run.

## Run

```sh
pip install modal
modal setup
modal run bench.py                      # or MODAL_PROFILE=<name> modal run bench.py
```

Costs about $0.40 per run. Output goes to `results/runNN/`; start with `report.txt`.

## Layout

`bench.py` (Modal app), `runner.py` (per-backend process), `reporting.py`, `custom/` (CUDA and PyTorch paths), `eval/` (clips, WER), `tests/`.

See docs/GUIDE.md for how it works, profiling, pitfalls and next ideas.
