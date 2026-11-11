# What the staging buffer really carries during a prompt batch

2026-10-21, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. Model
`DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`, 43 MoE layers, 256 experts a layer, top-6, 4 residual
streams, embedding 4096. Configuration `strata-dsv4-q3_k_m.json` unchanged: context 131072, batch 4096,
read-chunk 32768, stage 2848 MiB, rows-vram, compact-scratch, compact-graphs, reserve 256 MiB,
no-disk load-margin 256 MiB, swaps 64. The same two full inputs of 50000 tokens as the earlier series,
`resume=0` both times, answers up to 256 tokens.

The counters are new: `STRATA_TRACE_STAGING=1` makes the engine count, in `stage()` itself, the calls
and the bytes, split into the copies made before the routing was known (ahead), the copies made after it
(routed), and the rounds the buffer already held (reused), with the bytes per layer and the number of
read passes. Without the variable nothing is counted and nothing is printed.

## The answer

```
staging[prompt]: passes=2 ahead_calls=86 ahead_bytes=239896100864 (228782.75 MiB)
                 routed_calls=0 routed_bytes=0 (0.00 MiB)
                 reused_rounds=473 reused_experts=114851; total moved=228782.75 MiB
```

| | measured | expected from the geometry |
|---|---:|---:|
| read passes | 2 | 2 (32768 + 17231) |
| copies | 86 = 43 layers × 2 passes | 43 × 2 |
| copies after the routing | **0** | 0 |
| bytes of experts moved | **228782.75 MiB = 223.42 GiB** | 223.42 GiB |
| rounds the buffer already held | 473 | — |
| experts not copied again | 114851 | — |

**Every copy is an ahead copy, one per layer per pass, and there is not one copy after the
routing.** The staging buffer holds a whole layer: 256 experts × 11.125 MiB = 2781.6 MiB, and the
buffer is 2848 MiB, so the whole non-cached set of a layer fits in one round. There is no second round,
so nothing overwrites the ahead set; the next microbatch of the same layer finds `staged_ == L.ahead`,
takes `kept = 0` and copies nothing. The 473 reused rounds and the 114 851 experts that were not copied
again are that.

The bytes agree with the arithmetic on the layer geometry to the byte: 223.42 GiB for two passes, the
same number the rows/staging review computed offline. **There is no extra copying to find.**

## What the earlier numbers were

The 371 GiB of PCIe reads that the sample of the total RX showed is **not** the experts' copies: the
copies are 223.42 GiB. **What the rest of the RX is, the counters do not measure, and a transfer on the
card is not a transfer over PCIe** - it is not established.

The "14 experts that do not fit" claim came from a layer of 288 experts; the model has 256. And the
"threefold amplification" came from comparing the total RX with the experts' minimum: the copies are
exactly the expected ones.

## The prefill's speed and the file cache

The runs of the same configuration differ in speed, and the reason is the page cache of the model
file, not the engine:

| | prefill, first input | disk read in it | file faults |
|---|---:|---:|---:|
| the earlier series | 86.1 s (580.6 tok/s) | 126.78 GiB | 31 620 344 |
| this run | 92.9 s (538.0 tok/s) | 135.39 GiB | 31 686 437 |
| the run after it | 102.0 s (490.0 tok/s) | more again | — |

**The disk figures in this table are the samples' attribution and they are wrong.** The watchers on the
devices show the engine reading the model file itself during the load (126.49 GiB in 5464 `pread` calls of
~3.7 MiB, 0.71 GB/s, PhysicalDrive4) and **nothing** during the prefill (0.01 GiB from the NVMe and
0.08 GiB from the other disks in 30 s, the process's own `read_bytes` 0, no new faults). The process's
reads over a request grow by 229 117 bytes, which is the size of the input from the pipe. Windows' disk
counters lag, and the tail of the load's read lands in the next phase's window.

The PCIe reads of the experts (RAM to VRAM), the GPU utilisation (84.6% against 86.1%), the SM clock
(2416 against 2430 MHz) and the amount of expert work in the decode (permanent 567, staging cache
255/255, uploads 749) are the same in all of them. **What remains of the prefill's spread is not
established**; the drive's read speed over the day was between 0.53 and 0.71 GB/s, and the load, the
one phase that reads the disk, ran 199.7-236.9 s.

## Files

- `bat-no-disk/` — the run: `engine.stderr` with the counters, `protocol.txt`, `samples.jsonl`,
  `request-1.json`, `request-2.json`, `results.json`.
- `public-inputs.json`, `initial-expert-profile.txt` — the same prepared inputs and the same opaque
  copy of the configured statistics as the earlier series.
- The counters are in `src/poly/engine.cpp` (`stage()`) and `src/poly/engine.hpp` (`StageStats`); they
  are off unless `STRATA_TRACE_STAGING=1`.
