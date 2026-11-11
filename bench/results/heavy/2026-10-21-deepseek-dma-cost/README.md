# What the DMA window costs the prompt, and what the staging buffer copies

2026-10-21, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. Model
`DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`, 43 MoE layers, 256 experts a layer, top-6, 4 residual streams.
Configuration `strata-dsv4-q3_k_m.json` unchanged: context 131072, batch 4096, read-chunk 32768, stage 2848 MiB,
rows-vram, compact-scratch, compact-graphs, reserve 256 MiB, no-disk load-margin 256 MiB, swaps 64. The same two
full inputs of 50000 tokens as the earlier series, `resume=0` both times, answers up to 256 tokens.

The counters are new: `STRATA_TRACE_STAGING=1` makes the engine count, in `stage()` itself, the calls and the
bytes, split into the copies made before the routing was known (ahead), the copies made after it (routed), and
the rounds the buffer already held (reused), with the bytes per layer and the number of read passes. The DMA
window's own cost is counted where it happens: the wait for the DMA that used the pages, `cudaHostUnregister`
and `cudaHostRegister`, separately, and separately for the prefill and the decode.

## The staging copies: exactly the expected ones

```
staging[prompt]: passes=2 ahead_calls=86 ahead_bytes=239896100864 (228782.75 MiB)
                 routed_calls=0 routed_bytes=0 (0.00 MiB) reused_rounds=473 reused_experts=114851
```

| | measured | expected from the geometry |
|---|---:|---:|
| read passes | 2 | 2 (32768 + 17231) |
| copies | 86 = 43 layers x 2 passes | 43 x 2 |
| copies after the routing | **0** | 0 |
| bytes of experts moved | **228782.75 MiB = 223.42 GiB** | 223.42 GiB |
| rounds the buffer already held | 473 | - |
| experts not copied again | 114851 | - |

Every copy is an ahead copy, one per layer per pass, and there is no second round: the staging holds a whole
layer (256 experts x 11.125 MiB = 2781.6 MiB, and the buffer is 2848 MiB), so the next microbatch of the same
layer finds `staged_ == L.ahead`, takes `kept = 0` and copies nothing. The 473 reused rounds and the 114 851
experts that were not copied again are that.

## The DMA window: the unregistration is the cost

```
staging[prompt] dma prefill: calls=86 added=228782.75 MiB removed=228971.88 MiB
                           drain_ms=0.242 unregister_ms=11882.706 register_ms=1201.712
```

| per prompt (2 passes, 86 window changes) | |
|---|---:|
| window changes | 86 |
| bytes registered | 228782.75 MiB |
| bytes unregistered | 228971.88 MiB |
| **wait for the DMA that used the pages** | **0.24 ms** |
| **`cudaHostUnregister`** | **11882.7 ms** |
| **`cudaHostRegister`** | **1201.7 ms** |

**Taking the registration away is ten times as expensive as making it.** The window's limit (2850 MiB) holds
barely one layer (2434-2837 MiB), so the next layer's pages have to be registered anew either way; but the
*removal* of the previous window is what the prefill pays for, and `retain()` (which keeps the window and only
adds what is missing) would remove that cost - at the price of a window that can hold one layer at a time.

**This is 13.1 s of a prompt that takes 85-91 s: about 15%.** It is the largest single overhead measured in
the prompt so far, and it is not the experts' copies (those are overlapped with the attention) but the page
pinning around them.

**It is a fixed cost per layer per pass, not per token.** A second run with a 1000-token prompt (one pass,
the same 43 layers) shows it at its clearest:

| 1000-token prompt, 1 pass, 43 layers | |
|---|---:|
| the prompt itself | 13.3 s |
| window changes | 43 |
| bytes unregistered | 27847.19 MiB |
| wait for the DMA | 0.18 ms |
| **`cudaHostUnregister`** | **7213.3 ms** |
| **`cudaHostRegister`** | **922.1 ms** |
| **the window's share of the prompt** | **61%** |

The same 43 layers, the same number of window changes, the same cost per layer - and the prompt is ten times
shorter. The cost is in the page pinning, and it does not scale with the prompt.

## The decode: the same window, but a hundred times less of it

```
staging[request] dma decode: calls=99 added=4732.62 MiB removed=4977.38 MiB
                            drain_ms=0.195 unregister_ms=109.706 register_ms=64.589
```

| the decode of a 256-token answer | |
|---|---:|
| window changes | 99 |
| bytes unregistered | 4977.38 MiB |
| wait for the DMA | 0.20 ms |
| `cudaHostUnregister` | 109.7 ms |
| `cudaHostRegister` | 64.6 ms |
| **the window's share of the answer** | **0.17 s of 13.9 s: 1.2%** |

The decode's cache uploads one expert at a time, so the window it changes is small and the churn is two
orders of magnitude below the prompt's. **The overhead is the prompt's, not the decode's.**

## What this points at

The unregistration is ten times the cost of the registration, and the window's limit (2850 MiB) holds barely
one layer (2434-2837 MiB), so the next layer's pages have to be registered anew either way. What the
unregistration buys is nothing that the limit does not already decide. A window that was not thrown away
between layers would keep the prompt's 13 s (or 8 s on a short prompt) and would leave the registration's
1.2 s where it is.

**This is a measured overhead, not a proposed change: nothing in the engine was altered by it.**

## Files

- `bat-counters/` — the run: `engine.stderr` with the counters per layer and the DMA window's cost,
  `protocol.txt`, `samples.jsonl`, `request-1.json`, `request-2.json`, `results.json`.
- `public-inputs.json`, `initial-expert-profile.txt` — the same prepared inputs and the same opaque copy of the
  configured statistics as the earlier series.
- The counters are in `src/poly/engine.cpp` (`stage()`, `follow_resident()`), `src/poly/exchange.cpp`
  (`exchange_pair()`) and `src/poly/resident_dma.hpp` (`DmaCost`, in `src/poly/dma_cost.hpp`); they are off
  unless `STRATA_TRACE_STAGING=1`.
