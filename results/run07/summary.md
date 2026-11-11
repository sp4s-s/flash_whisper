# Whisper H100 benchmark

Latency starts with an in-memory waveform and ends with the transcript.

| Seconds | Backend | E2E median / p90 (ms) | TTFT median / p90 (ms) | TTLT median / p90 (ms) | Audio s/s | Reps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | vLLM | 73.5 / 79.7 | 34.6 / 35.2 | 73.5 / 79.7 | 68.02 | 20 |
| 5 | PyTorch | 56.0 / 66.2 | 17.6 / 20.4 | 55.9 / 66.1 | 89.34 | 20 |
| 5 | CUDA | 19.1 / 19.2 | 10.5 / 10.5 | 19.1 / 19.2 | 261.46 | 20 |
| 15 | vLLM | 126.7 / 156.5 | 33.7 / 37.5 | 126.6 / 156.5 | 118.43 | 20 |
| 15 | PyTorch | 110.4 / 142.8 | 18.9 / 20.1 | 110.3 / 142.7 | 135.89 | 20 |
| 15 | CUDA | 28.3 / 31.4 | 10.6 / 10.7 | 28.2 / 31.3 | 530.58 | 20 |
| 30 | vLLM | 222.8 / 254.1 | 33.3 / 36.2 | 222.8 / 254.1 | 134.65 | 20 |
| 30 | PyTorch | 204.2 / 242.7 | 19.5 / 20.8 | 204.0 / 242.6 | 146.92 | 20 |
| 30 | CUDA | 44.0 / 50.6 | 10.8 / 10.9 | 43.9 / 50.4 | 681.95 | 20 |

## Speedup (median; >1 means faster)

| Seconds | PyTorch E2E vs vLLM | PyTorch TTFT vs vLLM | CUDA E2E vs vLLM | CUDA TTFT vs vLLM | CUDA E2E vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.31x | 1.97x | 3.84x | 3.30x | 2.93x |
| 15 | 1.15x | 1.78x | 4.48x | 3.17x | 3.90x |
| 30 | 1.09x | 1.71x | 5.06x | 3.08x | 4.64x |

## Cost per audio hour (30 s clips, GPU list price)

| Backend | Audio s/s | USD per audio hour |
| --- | ---: | ---: |
| vLLM | 134.65 | 0.03 |
| PyTorch | 146.92 | 0.03 |
| CUDA | 681.95 | 0.01 |

## Acceptance (against vLLM)

- PyTorch 30-second speedup: 1.09x; fail (target > 1.5x).
- PyTorch TTFT: 19.5 ms vs 33.3 ms; pass.
- PyTorch WER difference: 0.000 percentage points; pass.
- CUDA 30-second speedup: 5.06x; pass (target > 1.5x).
- CUDA TTFT: 10.8 ms vs 33.3 ms; pass.
- CUDA WER difference: 0.000 percentage points; pass.

## Status

- completed; inspect correctness gates
- PyTorch vs vLLM: input hashes match: True; token IDs match: True.
- CUDA vs vLLM: input hashes match: True; token IDs match: True.

Timings are not comparable unless decoding and token correctness checks pass.
