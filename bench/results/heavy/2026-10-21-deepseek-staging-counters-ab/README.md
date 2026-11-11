# The staging counters cost nothing, and what the disk actually serves

2026-10-21, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. Model
`DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`. Configuration `strata-dsv4-q3_k_m.json` unchanged, the same two
full inputs of 50000 tokens, `resume=0` both times, answers up to 256 tokens.

Two runs, one binary, back to back, the only difference the variable `STRATA_TRACE_STAGING=1`:
`bat-no-disk` without the counters, `bat-counters` with them.

## The counters cost nothing

| | bat-no-disk (счётчики выкл) | bat-counters (счётчики вкл) |
|---|---:|---:|
| загрузка, с | 236.88 | 199.71 |
| префилл, ответ 1, ток/с | 536.3 | 555.9 |
| префилл, ответ 2, ток/с | 508.7 | 585.5 |
| декод, ответ 1, ток/с | 16.18 | 17.51 |
| декод, ответ 2, ток/с | 16.24 | 17.93 |
| hit, ответ 2 | 33.5% | 34.6% |

The run **with** the counters is the faster one. The counters are a few increments per `stage()` call
(86 calls per prompt), so there was no reason to expect a cost, and now there is a control rather than a
correlation.

**What this does not say.** One sequential comparison does not establish that the cost is zero, and it
explains nothing about the spread: the two runs are one after the other, and the machine's own spread
is larger than the difference. The load phase (the one that reads the disk) ran 236.9 s in one run and
199.7 s in the other, and the drive's read speed over the day was between 0.53 and 0.71 GB/s. Repeats
with the order reversed, and with more than one run of each, are needed before the speed of the
diagnostics is stated.

## What the disk serves in no-disk

Measured while a run was in each phase (a watcher on the process and on every device, 30 s windows):

| phase | who reads | how much | from where |
|---|---|---|---|
| load | **the engine itself** | 126.49 GiB in 5464 `pread` calls of ~3.7 MiB, 0.71 GB/s | PhysicalDrive4, the L: volume with the GGUF |
| prefill | **nobody** | 0.01 GiB from the NVMe and 0.08 GiB from the other disks in 30 s; the process's own `read_bytes` 0; no new faults | — |

The model's weights are copied once from the GGUF into the locked RAM arena, and the file is closed
after that. During the prefill the experts are read from the locked RAM through the registered DMA
window; the disk is not touched. The process's own reads over a request grow by 229 117 bytes, which is
the size of the input from the pipe.

**The 126.78 GiB that the samples attribute to the prefill is a misattribution.** Windows' disk
performance counters lag, and the tail of the load's read - which ends exactly at the load/request
boundary - lands in the next phase's window. The engine's own `read_bytes` over the prefill is zero.

**What the rest of the PCIe reads are is not established.** The counters do not measure it, and a
transfer on the card is not a transfer over PCIe. The expert copies are the 223.42 GiB the staging
counters measured; what carries the rest of the RX sample is not known.

## What is still open

- The prefill's speed varies by ±7-10% between runs of the same configuration, and the load phase (the
  one that reads the disk) by 199.7-236.9 s. Which part of the prefill's spread is the machine and which
  is the engine is not established.
- The no-disk path re-registers the DMA window on every layer: `stage()` calls `ResidentDma::prepare()`,
  which unregisters the previous window and registers the new one, about 2.7 GiB of pages per layer, 86
  times per prompt. The exchange path's `retain()` keeps the window and only adds what is missing, but
  the window's limit (2850 MiB) holds barely one layer (2434-2837 MiB), so it re-registers most of
  the next layer as well. The cost of this is measured separately now: `DmaCost` in `src/poly/dma_cost.hpp`
  counts the calls, the bytes and the three times (the wait for the DMA, `cudaHostUnregister`,
  `cudaHostRegister`), and the staging line prints them split by the prefill and the decode.

## Files

- `bat-no-disk/`, `bat-counters/` — the two runs: `engine.stderr` with the counters, `protocol.txt`,
  `samples.jsonl`, `request-1.json`, `request-2.json`, `results.json`.
- The counters are in `src/poly/engine.cpp` (`stage()`) and `src/poly/engine.hpp` (`StageStats`).
