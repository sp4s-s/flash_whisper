# Whisper H100 benchmark

Latency starts with an in-memory waveform and ends with the transcript.

| Seconds | Backend | E2E median / p90 (ms) | TTFT median / p90 (ms) | TTLT median / p90 (ms) | Audio s/s | Reps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | vllm | 253.5 / 292.2 | 46.0 / 48.9 | 253.5 / 292.2 | 19.72 | 20 |
| 5 | custom reference | 52.5 / 59.1 | 18.5 / 20.3 | 52.4 / 59.0 | 95.17 | 20 |
| 15 | vllm | 552.2 / 794.7 | 51.5 / 55.5 | 552.2 / 794.7 | 27.16 | 20 |
| 15 | custom reference | 97.8 / 124.0 | 19.0 / 19.9 | 97.6 / 123.9 | 153.41 | 20 |
| 30 | vllm | 1090.5 / 1231.4 | 47.2 / 52.5 | 1090.5 / 1231.3 | 27.51 | 20 |
| 30 | custom reference | 192.7 / 220.1 | 19.6 / 20.7 | 192.6 / 219.9 | 155.64 | 20 |

## Acceptance

- 30-second speedup: 5.658x; unverified (target > 1.5x).
- TTFT: unverified.
- WER difference: 0.000 percentage points; unverified.

## Status

- incomplete
- Input hashes match: True; token IDs match: True.
- No custom CUDA kernel was run; optimization stage is incomplete.

Timings are not comparable unless decoding and token correctness checks pass.
