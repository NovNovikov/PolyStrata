# Poly no-disk, 2026-10-18

Windows, i5-13600KF, 12 CPU worker threads, RTX 4090 (24564 MiB), 128 GiB RAM (130906 MiB visible).
CUDA 12.4, MSVC 19.41. Model: `DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`, 135658482976 bytes.
The synthetic input was exactly 10000 tokens from `tools/bench_poly_no_disk.py`. Every run started a fresh
process, copied the same supplied expert profile, disabled draft and prompt caching, used greedy sampling with
seed 42, and set `STRATA_FREEZE_EXPERTS=1`. No user chats, prompts, session files or private logs were read.

## Measured runs

| Run | Context | Batch | Staging request, MiB | VRAM reserve, MiB | Read rows | Prefill tokens/s | System disk reads, GiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Previous binary, mmap | 16384 | 1024 | 3072 | 1024 | RAM | 56.325 | 132.676 |
| Current binary, mmap | 16384 | 1024 | 3072 | 1024 | RAM | 51.116 | 142.281 |
| No-disk pilot | 16384 | 1024 | 2048 | 2048 | VRAM | 199.542 | 0.313 |
| No-disk, first load | 16384 | 1024 | 3072 | 1024 | VRAM | 587.945 | 0.049 |
| No-disk, second load | 16384 | 1024 | 3072 | 1024 | VRAM | 584.099 | 0.285 |
| No-disk, host rows | 131072 | 1024 | 3072 | automatic (1024) | RAM | 411.413 | 0.877 |

The two identical resident configurations differed by about 0.65% in prefill speed. Loading itself took
200.768 and 219.295 s and still reads the model; those times are outside the measured prefill.
Their RAM/VRAM split responded to available RAM: 10471/537 and 10595/413 experts, respectively. Each sums to
11008. Locked host memory was 115739 and 117096 MiB, including the embedding and CPU token-routing tables.

The process read exactly 52453 bytes during each of those two prefills, in 13 reads. That is exactly the size
of the token command sent to its stdin, including Windows CRLF. There were no other process reads and no open
model file after READY. The reader never maps a model in this mode and closes its file handles after loading,
so these are not hidden faults on a model mapping. The loader recorded the same 135653146972 payload bytes
read at load in all resident runs; GGUF headers and padding are separate.

System counters include all processes and mapped DLL/code faults; their nonzero values above are reported
as observed, not labelled as reads of the model. The mmap process also reads only its command through normal
ReadFile calls: its model I/O arrives through mapped-page faults, which is why a process read counter alone
would be insufficient evidence. The file/mapping checks matter.

The resident pair started at RSS 121904275456 and 123328118784 bytes. Neither sampled RSS fell below its starting value
during generation; their maxima were 122110775296 and 123543216128 bytes. The sampled system swap-used counter
decreased by 2338816 and 35131392 bytes during generation. The allocated pagefile sizes remained 16384 MiB
on C: and 32768 MiB on L: before and after the run sequence. Both live `VRAM 0` requests were refused without
ending the engine.

The final host-row case tested the main `--no-disk` flag without `--rows-vram`, with full KV for a 131072-token
context. It placed 10494 experts in RAM and 514 in permanent VRAM, read only its 52453-byte stdin command during
generation, and had no open model file. Sampled RSS did not fall below its starting value and the swap-used counter
decreased during the request. This validates the host-row path as well as the optional device-row path.

All six successful runs produced the same eight generated token IDs, recorded in [results.json](results.json).
The old and current default paths therefore matched on this input. The mmap timings were not a controlled
comparison of code speed: OS cache state was not flushed or held constant, and their storage counters differed.
The resident figures are for this synthetic text; no throughput claim is made for a private coding conversation
or a different model. The supplied llama.cpp figure used a different prompt length and is not a direct speed ratio.

## Refusals and checks

The 32768-context device-row configurations did not fit at the RAM available during testing:

- Batch 4096, staging 2048 MiB, VRAM reserve 2048 MiB: remaining expert RAM demand 124970598400 bytes,
  RAM budget 119296724992, expert VRAM budget 1482686464. The loader refused before loading expert payloads.
- Batch 1024, staging 3072 MiB, VRAM reserve 2048 MiB: remaining expert RAM demand 121681412096 bytes,
  RAM budget 120639905792, expert VRAM budget 4771020800. It likewise refused.
- `--ram-limit-mib 1` refused the 1059061760-byte embedding table rather than leaving it in a mapped file.
- `--no-disk --layer-split auto` was refused explicitly.

An independent `poly_resident_probe 112640` locked all 110 GiB in 21.53 s, then released it.
The CPU suite passed 1291 no-disk placement/reader checks, 55602 conversation checks and 68 disk-session checks.
The disk-session test driver needed its already obsolete periodic boundaries replaced with explicit semantic
boundaries; engine checkpoint behavior was not changed for this feature.

Build: `build-strata-poly.bat` (vcvars64, CUDA 12.4); standalone tests were built with CUDA disabled and
`STRATA_BUILD_CONVERSATION_TESTS=ON`. CUDA was built and exercised. HIP/SYCL toolchains and AMD hardware were
not available here; those backends and Linux residency were not built or validated in this run.

The old binary used for default comparison was saved before the feature changes; its SHA256 was
`b13780c75d789a8a32abd6a37dbc2931da823d2d86d9e539ad829c1025cf07c6`.
Raw counters and token IDs are in [results.json](results.json); command construction and input generation are
in [the benchmark tool](../../../tools/bench_poly_no_disk.py).
