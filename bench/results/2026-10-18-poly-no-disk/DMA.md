# CUDA registration correction, 2026-10-18

Same Windows/i5-13600KF/RTX 4090/128 GiB PC and DeepSeek V4 Flash Q3_K_M model as [the first report](README.md).
The earlier implementation OS-locked its RAM but CUDA reported `cudaMemoryTypeUnregistered`. It could therefore
stage pageable transfers internally. The correction registers the original expert source pages before DMA;
CUDA reports Host memory at both ends of every registered range. No second host expert buffer is allocated.

## Registration checks

- CUDA refused one 110 GiB registration (`cudaErrorMemoryAllocation`, 0.13 s). OS-locking that same arena and
  registering a 4 GiB window succeeded; CUDA reported Host memory at its beginning and end.
- `poly_resident_dma_probe 112640 3072` locked all 110 GiB, switched a 3 GiB registration window between the
  arena's beginning and end, checked copies on two nonblocking streams and checked cleanup: passed, 22.65 s.
- `poly_resident_dma_probe 16 8 2` additionally checked disjoint source ranges, a deliberately induced partial
  registration failure, rollback without unregistering an externally owned range, window reuse, and destruction
  after a copy whose completion event was not recorded: both repeats passed, 0.01 s each.
- CPU placement/reader/page-range suite: 1563 checks. The conversation and disk-session suites also passed;
  the three CTest targets passed together.

Registration is bounded by staging plus page rounding. Before changing the window, the engine waits for
source-copy completion on every stream that used it. All expert RAM remains OS-locked between windows.
Registration/verification errors refuse the upload rather than selecting pageable copies.

## Real model run

Fresh process, exactly 10000 synthetic prompt tokens from `tools/bench_poly_no_disk.py`, greedy seed 42,
8 generated tokens, 12 CPU threads, no draft, zero prompt reuse, frozen expert placement. Context 16384,
batch 1024, `--stage-mib 3072` (actual 2848), reserve 1024 MiB, `--no-disk --rows-vram`.

- Load: 202.406 s. Prompt: 20626.7 ms, **484.809 tokens/s**.
- Decode: 340.6 ms, **20.552 tokens/s** using the engine's `(generated - 1) / decode time` convention.
  Eight output tokens are too short for a sustained decode-speed measurement.
- All eight generated IDs matched all six earlier successful runs.
- 10551 experts in RAM, 457 permanently in VRAM; total 11008.
- Registered source window: 2848 MiB peak, 2850 MiB limit including page rounding.
- KV: 107.50 MiB VRAM. Full-context read rows: 1024.00 MiB VRAM. `h_res_`: 0.
- Engine VRAM allocations: 20040.64 MiB. Host weights + pinned buffers: 116718.48 MiB.
  The registration window is part of those host weights and is not added again.
- No open model file after READY. Process reads during the request: 52453 bytes, exactly its stdin command;
  no additional process reads. System disk reads were 204469248 bytes and include unrelated activity.
- Sampled RSS never fell below its starting value; system swap-used decreased by 30629888 bytes.
- A live `VRAM 0` request was refused without ending the engine.

The old 587.945/584.099 figures were from the unregistered path. This corrected run also placed a different
number of experts in VRAM because available RAM differed, so the runs do not isolate registration overhead.

## Host read rows and allocation accounting

A separate fresh process with context 16384 and the same batch/staging/reserve, without `--rows-vram`, processed
two different synthetic prompts of 3000 and 7000 tokens. Both reported zero reused tokens. Its first prompt's
two generated IDs matched the same prompt before the DMA correction.

| Point | h_res_, MiB | Pinned buffers, MiB | Host weights + buffers, MiB |
| --- | ---: | ---: | ---: |
| Load | 0 | 101.54 | 116803.42 |
| First prompt | 256 | 357.54 | 117059.42 |
| Second prompt | 512 | 613.54 | 117315.42 |

The freed 256 MiB was removed before counting the replacement 512 MiB. VRAM engine allocations stayed
18931.69 MiB, with KV 107.50 MiB and zero VRAM read rows. All displayed component sums were checked against
their totals within display rounding. The CUDA-registration peak stayed 2848 MiB. Both requests completed,
and the engine exited normally. The actual 32k DeepSeek and 200k GLM launcher configurations were also checked
to print a partial breakdown when placement/allocation was refused; their capacity settings were not changed.

Raw counters are in [dma-results.json](dma-results.json). The actual allocation diagnostics are in
[dma-memory.txt](dma-memory.txt). CUDA/Windows was built and exercised; Linux and HIP were not validated here.
