# Guide

Benchmark of `openai/whisper-large-v3-turbo` on one Modal H100: vLLM 0.33.0, a plain PyTorch path and a CUDA path with hand written kernels. FP16, batch 1, greedy decoding, 100 LibriSpeech test-clean clips for WER.

## Results (run07, H100, 30/15/5 s)

Median of 20 runs. Speedup is vLLM / backend.

| Audio | vLLM E2E | PyTorch E2E | CUDA E2E | CUDA E2E speedup | vLLM TTFT | PyTorch TTFT | CUDA TTFT | CUDA TTFT speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 222.8 ms | 204.2 ms (1.09x) | 44.0 ms | 5.06x | 33.3 ms | 19.5 ms (1.71x) | 10.8 ms | 3.08x |
| 15 s | 126.7 ms | 110.4 ms (1.15x) | 28.3 ms | 4.48x | 33.7 ms | 18.9 ms (1.78x) | 10.6 ms | 3.17x |
| 5 s | 73.5 ms | 56.0 ms (1.31x) | 19.1 ms | 3.84x | 34.6 ms | 17.6 ms (1.97x) | 10.5 ms | 3.30x |

- WER is 1.31% on all three backends, token ids and input hashes match vLLM on all checked pairs, kernel selftests passed, ptxas reports 0 register spills.
- CUDA decodes at 0.40 ms/token against 2.19 for vLLM and 2.16 for PyTorch. At 30 s the CUDA stage shares are mel 1.7%, encoder 19.6%, decode 78.3%.
- Context: run04 used vLLM 0.10.2, which could not torch.compile Whisper. It took 1090.5 ms at 30 s (11.6 ms/token), so it was a much slower baseline. The pin moved to 0.33.0.
- Note: run07 vLLM (222.8 ms at 30 s, 8 vCPU) was slower than run05 (157.0 ms, 2 vCPU). Treat vLLM numbers as run-to-run variable and compare backends within one run.

## How each backend works

- **vLLM** (`runner.py`): AsyncLLM engine with `max_num_batched_tokens=2048`, spawn workers, `gpu_memory_utilization` 0.5.
- **PyTorch** (`custom/pytorch_model.py`): Hugging Face transformers in FP16, CPU feature extraction, manual greedy loop with the HF KV cache and one host sync per token. No custom kernels or graphs.
- **CUDA** (`custom/model.py`, `custom/kernels/whisper_kernels.cu`):
  - Log-mel on the GPU (`torch.stft` with the feature extractor filters).
  - The Hugging Face encoder (SDPA) runs in its own CUDA graph on a static mel buffer, with eager fallback. Cross K/V for all decoder layers come from one GEMM per clip.
  - Decode is a CUDA graph over a static KV cache with the position on the device, replayed in 8-step chunks: one host sync for TTFT, then one per chunk.
  - Kernels: fused residual add + LayerNorm; split-K decode attention (self and cross, self also writes the new K/V row) plus its combine kernel; fused LM head GEMV + argmax; token finalize. Projections and MLP stay on cuBLAS.

## Run and read results

```sh
pip install modal && modal setup
modal run bench.py                      # or: MODAL_PROFILE=<name> modal run bench.py
```

- A run decodes clips once on a CPU container (cached in a volume), then on one H100 runs vllm, pytorch and cuda each in its own process: 3 warmups, 20 timed runs at 5, 15 and 30 s. Then WER on 100 clips, ncu on the CUDA kernels, nsys on all three, and a download to the next free `results/runNN/`.
- `report.txt` is the main file (tables, per-stage breakdown, correctness, selftest, nsys, ptxas). `summary.md` has only the tables. Also: `<backend>.json`, `wer.json`, `transcripts/`, `nsys/`, `ncu/`, `ptxas/`, `clocks.csv`.
- Cost is about $0.40 per run (run07: $0.454, 357 s remote). `bench.py` stops at a $7 local estimate (`budget.json`).
- Timing boundary: starts with an in-memory waveform, ends with the transcript string. Model load, warmups and JIT builds are outside it.

## Profiling shortcuts

- Spills: the extension builds with `-Xptxas=-v,-warn-spills -lineinfo -arch sm_90a`; read `results/runNN/ptxas/summary.txt`. Look for nonzero spill stores or loads.
- nsys, 3 passes per backend: `nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --cuda-graph-trace=node python runner.py --backend X --profile-only`. Graph node tracing is needed to see kernels inside the decode graph.
- ncu: `ncu --kernel-name regex:<k> --launch-count 1 --clock-control none --section SpeedOfLight --section Occupancy python -m custom.selftest`. Without `--clock-control none` Modal fails with "Failed to lock GPU clock frequencies".
- Clock throttling check: `nvidia-smi --query-gpu=timestamp,clocks.current.sm,power.draw,temperature.gpu --format=csv -lms 100` (run07 SM clock median 1980 MHz).
- SASS spill check: `cuobjdump -sass x.o | grep -E "STL|LDL"`. Useful ncu metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_active`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`.

## Pitfalls hit and fixes

- Data prep aborted with exit -6: datasets 3.6 needed `librosa`; fixed by moving to datasets 5.2 with torchcodec and a CPU-only prep function cached in the volume.
- vLLM 0.10.2 stalled on a T4 after "FA2 needs compute capability >= 8": the target moved to H100.
- vLLM refused to start with `max_tokens_per_mm_item (1500) > max_num_batched_tokens`: Whisper needs at least 1500, we use 2048.
- Stale vLLM pin made the baseline too slow: now vLLM 0.33.0 with a full `requirements.lock`.
- ncu "Failed to lock GPU clock frequencies": pass `--clock-control none`.
- 2 vCPUs starved vLLM (frontend, EngineCore and detokenizer are separate processes): GPU function now has 8 vCPU and 32 GB.
- Cache env vars set on the image made build steps write into /cache and broke the volume mount: set them at runtime.

## Next ideas

- Decode is 78% of CUDA E2E. `decode_attention_split_kernel` is the largest custom kernel (19.1 ms per pass, 10.5 us avg), then `lm_head_argmax` (10.1 ms, 44 us avg, versus 52.7 us for the cuBLAS gemv). Tune the split count and the LM head tiling with ncu.
- The 0.4 ms/token decode is dominated by cuBLAS projections (nvjet, 26 ms per pass); fusing QKV or MLP epilogues is the next lever.
- Encoder is 8.6 ms at 30 s (19.6%); a faster attention or fused epilogues would help TTFT.

## Maintenance

- Lock: `uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o requirements.lock`.
- Tests: `pip install pytest numpy jiwer openai-whisper && python -m pytest -q tests`.
- Known limits: runs are sequential not interleaved; only 3 timed pairs per duration are token-compared; vLLM nsys NVTX comes from the frontend process; the volume is still named `whisper-t4-bench-data`.
