# Whisper H100 benchmark

Latency starts with an in-memory waveform and ends with the transcript.

| Seconds | Backend | E2E median / p90 (ms) | TTFT median / p90 (ms) | TTLT median / p90 (ms) | Audio s/s | Reps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | vllm | 49.8 / 55.9 | 24.0 / 25.2 | 49.8 / 55.9 | 100.39 | 20 |
| 5 | custom cuda | 25.9 / 26.2 | 12.3 / 12.6 | 25.9 / 26.1 | 192.96 | 20 |
| 15 | vllm | 87.2 / 107.7 | 24.2 / 25.0 | 87.2 / 107.7 | 172.08 | 20 |
| 15 | custom cuda | 39.6 / 44.3 | 12.4 / 12.5 | 39.6 / 44.2 | 378.49 | 20 |
| 30 | vllm | 157.0 / 177.7 | 24.8 / 25.8 | 157.0 / 177.7 | 191.06 | 20 |
| 30 | custom cuda | 63.0 / 72.9 | 12.6 / 12.9 | 62.9 / 72.8 | 476.32 | 20 |

## Speedup (vLLM / custom, median; >1 means custom is faster)

| Seconds | E2E × | TTFT × | TTLT × | Throughput × | E2E p90 × |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.92x | 1.95x | 1.93x | 1.92x | 2.13x |
| 15 | 2.20x | 1.96x | 2.20x | 2.20x | 2.43x |
| 30 | 2.49x | 1.97x | 2.50x | 2.49x | 2.44x |

## Acceptance

- 30-second speedup: 2.49x; pass (target > 1.5x).
- TTFT: pass.
- WER difference: 0.000 percentage points; pass.

## Status

- completed; inspect correctness gates
- Input hashes match: True; token IDs match: True.

Timings are not comparable unless decoding and token correctness checks pass.
