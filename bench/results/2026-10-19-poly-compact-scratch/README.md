# DeepSeek V4: reuse scratch between compute phases

Windows, RTX 4090 24 GiB (24563.50 MiB), i5-13600KF, 12 CPU threads, 128 GiB RAM,
CUDA 12.4 / MSVC 19.41, architecture 89. All inputs are synthetic; no private prompts or logs were read.
Model: DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf, 135653146972 bytes, 43 layers, 11008 routed experts.

## Result and activation

`strata-poly --compact-scratch` opts into phase reuse and bounded DeepSeek4 compressor history. Precision,
batch size, attention context, selected experts and the arithmetic order of compression are preserved.
The default layout and other architectures keep their original allocation path. The legacy compressor
kernel specialization remains selected without the flag.

Measured allocations at context 131072, read chunk 32768, one sequence:

| Prompt batch | Original other, MiB | Phase reuse only, MiB | Final compact, MiB | Final saving, MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.50 | 19.35 | 19.35 | 0.15625 (unrounded delta) |
| 1024 | 1415.54 | 1015.54 | 727.81 | 687.734375 |
| 4096 | 5164.24 | 3747.36 | 2589.63 | 2574.61328125 |

The original scratch arena at batch 4096 was 3723047936 bytes; the compact one is 2237342720 bytes.
The 1485705216-byte difference is allocation saved, not a sum of overlapping logical slices. Short compressor
rings save another 1213972480 bytes: 1171.640625 -> 13.90625 MiB. Combined saving is 2699677696 bytes.
No permanent experts were present in this inventory: the device total fell from 18678.61 to 16104.00 MiB.
In normal no-disk VRAM-first placement the loader spends the freed budget on more permanent GPU experts,
so total requested VRAM can stay near its original budget while locked RAM weights decrease.

## What made other grow

The allocation-only sweep changes batch while keeping context, read chunk, staging, KV format and sequence
count fixed. Both versions use full-context KV, zero permanent GPU experts, and QUIT immediately after READY.
This is mmap mode to avoid pinning a model-sized RAM arena; no inference is timed in this sweep. No-disk
generation is tested separately below. At batch 1 the engine does not allocate batched expert staging.

Between batch 1024 and 4096, original other grows by about 3748.70 MiB:

- architecture scratch: 996.298 -> 3550.575 MiB, +2554.277 MiB;
- compressor/raw rings: 350.016 -> 1349.016 MiB, +999.000 MiB;
- other allocations outside the arena (quantized inputs, routing tables, logits): about +195.42 MiB.

Inside the scratch increase, q_/att_ account for +768 MiB, gu_/hx_/dx_ for +720, the double residual
workspace r_ for +384, cur_/xn_/ao_/fo_/g_/u_/h_ for +336, and oa_/ob_ for +192.
These are batch-sized allocations. The optimization shares their physical storage by lifetime rather
than assuming their calculations are unnecessary. Current-batch compressor values are no longer stored
independently for all layers; only the history needed by the next call and speculative rollback remains.

Commands and binary hashes for every final inventory point are in `inventory-final/results.json`; full allocation
records, offsets and union coverage checks are in the adjacent stderr files. Reproduce with:

```text
python tools/bench_poly_scratch_inventory.py --baseline-exe BEFORE.exe --exe AFTER.exe --model MODEL.gguf --output NEW_DIR
```

## Why the overlapping ranges do not conflict

The layout planner is `src/poly/deepseek4_scratch.hpp`, used by DeepSeek4::assign_scratch only with the flag.

1. Queries q_ stay separate from attention output att_. Preparation buffers t1_/qr_/kv_, compressor
   temporaries and indexer iq_/iw_/score_ finish before dsv4_attend writes att_, on the same stream.
   sel_ and persistent attention caches stay outside the reused range.
2. out_proj first permutes att_ into distinct q_, then produces oa_ in distinct storage. Only after every
   grouped q_ read has finished does it permute oa_ into ob_, which can reuse the dead q_ storage.
   Neither permutation operates in place. The single-token fast path retains the same ordering.
3. Batched MoE uses gu_/hx_/dx_ after attention/out_proj, and moe_sum finishes before the next attention.
   These ranges can share the attention workspace. Small-token MoE uses separate g_/u_/h_/d_.
4. mix_/hcw_ remain outside both workspaces: hyper-connection weights survive until layer_tail.
   Without MMQ there is no expert workspace to borrow; the attention-only sharing still works.

## Why the compressor history can be short

The original code writes all current projected rows into each layer's ring before compressing them.
This needs `batch + overlap` rows for every compressor, although only 7 (ratio 4 with overlap) or
127 (ratio 128) rows are serialized as history between calls. The current projected rows already exist
in ckv_/cscore_, shared between layers.

The compact kernel reads positions belonging to the current call from those temporaries, adding the same
APE score term, and earlier positions from the ring. After compression, a tail-only state kernel stores
the final min(call_rows, ring_rows) rows. This gives each history slot one writer even for a 4096-row call.
The ring is `overlap + window`: 16 rows for ratio 4 and 136 for ratio 128 with window 8. The extra window
preserves the history needed when speculative tokens are rolled back. No full-context compressed KV rows
are discarded, and the serialized history format and size remain unchanged.

`--dump-buffers` keeps each logical slice with its offset, but checks physical coverage using the union
of ranges. Its additional phase-reuse line reports logical bytes minus covered bytes. Device totals
still count only the actual arena allocation. CUDA graphs record/replay the same stream ordering.

## Correctness

- CPU layout test: 128 combinations of rows 1/8/31/32/33/256/1024/4096, context 16/16k/32k/128k,
  MMQ present/absent and two output geometries. Independent live-set checks, guarded writes for small
  layouts, and persistent hyper-connection checks across two phase cycles. This is a CPU guard test,
  not a GPU sanitizer run.
- Full-model comparison, batch 4096, CUDA graphs enabled: **51 frames, 6593280 full-vocabulary logits,
  zero bit differences, max_abs=0, mean_abs=0**. Every captured value was finite.
  Includes prefixes 1/8/31/32/33, window/keep, a full batch plus 33 rows, actual indexer scoring,
  state/attention-row restoration, two-sequence step, and cancellation followed by a new prefill.
- For that comparison both layouts have zero permanent experts, swaps disabled, identical CPU share,
  STRATA_FREEZE_EXPERTS=1 and fixed input IDs. The test compares model logits, not forced output IDs.
  It uses mmap to fix placement without two model-sized locked arenas; it is not a speed result.
- Existing poly_no_disk_test, glm_conversations_test and glm_conversations_disk_test also passed.
- The same full-model graph test passed on the FINAL compressor variant as well: 6593280 logits,
  zero bit differences (`parity-final-b4096-graphs.txt`).
- Phase reuse without graphs, batch 33: another 6593280 logits, zero bit differences
  (`parity-stage1-b33-no-graphs.txt`).
- FINAL batch 33, no graphs, full eight-token window: **71 frames, 9178880 logits, zero bit differences**.
  Serialized state bytes matched across layouts. The compact engine imported the ORIGINAL engine's state
  and attention rows, then reproduced the continuation exactly. Two-sequence step and read cancellation
  also passed (`parity-final-b33-no-graphs.txt`).
- FINAL batch 1, graphs on, context 131072: **34 frames, 4395520 logits, zero bit differences** and the
  same cross-layout checkpoint test. MMQ is absent here, so there is no expert scratch to borrow. This
  case uses one sequence and a one-token window; step/cancellation are intentionally covered by batch 33,
  not claimed for this case (`parity-final-b1-graphs-128k.txt`).
- Combined FINAL full-model comparisons: **20167680 logits**, every value finite and every compared float
  bit-identical. Phase-only runs are additional checks and are not included in this total.
- Final compressor GPU test: ratio 4 at dimensions 512 and 128, ratio 128 at dimension 512. Each has
  11 calls, including 4096 rows, partial blocks, ring wrap, explicit at[] and rollback of five of eight
  guesses. Compressed fp16 outputs and retained float history match the original kernels bit for bit;
  GPU prefix/tail canaries passed. Compute Sanitizer memcheck completed the entire target with zero errors
  (`compressor-memcheck.txt` and `compressor-memcheck-target.txt`); target output confirms all cases ran.

Full-model commands (their output files are named above):

```text
poly_scratch_parity_test MODEL.gguf 4096
poly_scratch_parity_test MODEL.gguf 33 no-graphs
poly_scratch_parity_test MODEL.gguf 1 graphs 131072
```

Additional runtime and no-disk measurements are recorded below.

## Binary identity

Original EXE copied before rebuilding: SHA256
`810c3aac261caf771d95e8c8bbd2babbf5767ca16f7e16d8102b2542590147a3` (source 2c1dbaad).
Phase-reuse-only EXE: SHA256 `e74c7216dc8d5b7b9f1366442df971709001850e87f9215b84a8aaa89ea2aea4`.
Final EXE with bounded compressor rings: SHA256
`96c402130e67d20de09217ea7652ed6a8e9adf3d297da68ddaa0bdec83328d82`.
Source commits: `8065d613` (phase reuse) and `c0edb72e` (bounded history and kernel checks). Binaries were
built from the corresponding working trees before committing; hashes identify the exact measured files.
The build explicitly recompiled engine.cpp, deepseek4.cpp, no_disk.cpp and main.cpp before linking.
Only Windows/CUDA was compiled and run; HIP/Linux/SYCL and other GGUF geometries were not GPU-tested.

## Initial control refusal

`baseline-b4096/` refused before READY at context 32768, chunk/batch 4096, staging 2848 MiB:
experts needed 119958863872 RAM bytes, but 119639089152 were available (about 305 MiB short).
It contains no completed generation or speed result. The paired runs use staging 2048 MiB in BOTH
versions, with other settings identical. No reserve or RAM limit was bypassed.

## Paired no-disk run of phase reuse (first implementation)

Context 32768, chunk/batch 4096, staging 2048 MiB, reserve/loading allowance 256 MiB each, no draft,
lookup disabled, prompt/conversation caches zero. Each process generated two natural 512-token answers
after independent 10000-token prefills; the second input changed its first token to prevent prefix reuse.
Adaptive experts and payload verification were enabled in both versions.

| | Original | Phase reuse only |
| --- | ---: | ---: |
| Permanent GPU experts | 640 | 770 |
| Staging decode cache capacity | 184 slots | 184 slots |
| Other, MiB | 4908.35 | 3491.36 |
| Total requested VRAM, MiB | 22182.23 | 22181.75 |
| Prefill tok/s, requests 1 / 2 | 275.41 / 281.62 | 271.29 / 273.35 |
| Decode tok/s, requests 1 / 2 | 22.09 / 22.21 | 22.63 / 22.96 |
| Decode after first 128 tokens | 22.43 / 22.51 | 22.90 / 23.49 |
| GPU expert hit rate | 49.70% / 49.96% | 52.94% / 52.95% |
| Peak process RSS, MiB | 115612.22 | 114196.48 |

All 512 output IDs matched the original for EACH request. Cache payload/zero-tail checks passed after
each answer; the intervening prefill reclaimed and refilled staging. Both runs read exactly the stdin bytes
after READY, with zero extra process reads and model files closed. VRAM resize was refused as required.
The phase-only run ended with 422.20 MiB whole-device free, above the 256 MiB runtime reserve.

These are practical placement comparisons, not isolated kernel speedups: freed scratch permits more GPU
experts. Prefill was 1.5-2.9% slower in this pair; two requests are insufficient to assign a cause or claim
statistical equivalence. Only 384 answer tokens remain after the 128-token warmup cut in this pair; the
final long run uses 1024-token answers. Raw results and equality/rate calculation: the two stage2048
directories and `pair-analysis.json`. These timings belong to the FIRST implementation's binary, not the final one.

## Full-configuration refusal of the first implementation

`compact-128k-50k/` never reached READY. At context 131072/chunk 32768/staging 2848, the loader made four
RAM/DMA placement attempts and refused when the live VRAM budget kept falling. On the fourth, its planned
expert arena was 5064423320 bytes versus a live budget of 5052768256 (about 11.1 MiB over). Reserve and
loading allowance remained 256 MiB each. No expert payload or prompt was tested in this refused run.
This is retained as a loader-stabilization limitation, not a speed result or a change to reserve policy.

## Final no-disk pair: context 128k, two 50k prefills and two 1024-token answers

Same day, model and initial expert profile, context 131072, batch 4096, read chunk 32768, staging 2848 MiB,
runtime reserve 256 MiB, loading allowance 256 MiB, 12 CPU threads, rows in VRAM, no draft, lookup 0,
prompt/conversation caches zero. Adaptive experts and payload verification were enabled. Lookup is disabled
in BOTH measurements; these rates must not be compared to older lookup-3 timings as if the modes matched.
The second request changes its first input token, forces a new prefill, and reclaims/refills staging.

| | Original EXE | Final compact |
| --- | ---: | ---: |
| Load, s | 230.043 | 203.759 |
| Other, MiB | 5164.24 | 2589.63 |
| Permanent GPU experts | 309 | 577 |
| Permanent expert arena, MiB | 3380.74 | 6312.55 |
| Staging cache capacity | 255 slots | 255 slots |
| Total requested VRAM, MiB | 22059.35 | 22416.55 |
| Host resident weights + pinned buffers, MiB | 118623.76 | 115692.63 |
| Peak process RSS, MiB | 119241.29 | 116309.61 |
| Prefill tok/s, request 1 / 2 | 598.78 / 599.90 | 589.08 / 598.05 |
| Decode tok/s, request 1 / 2 | 19.86 / 19.73 | 21.33 / 21.28 |
| Decode after first 256 tokens | 19.95 / 19.73 | 21.47 / 21.40 |
| GPU expert hit rate | 43.38% / 43.24% | 51.26% / 51.57% |
| Cache uploads, request 1 / 2 | 350 / 286 | 364 / 289 |
| Whole-device free after answer 2, MiB | 542.63 | 417.26 |

ALL 1024 naturally generated token IDs matched for EACH request: 2048 IDs in total, without forced output.
Both processes kept model files closed after READY, read exactly the 525923 stdin bytes and had zero extra
process reads. Every one of the 255 published cache payloads and its zero tail passed after each answer.
The next prefill discarded the 255 cached RAM copies and the next answer refilled them. Resize was refused,
QUIT returned zero, and the compact process stayed above the 256 MiB runtime reserve in the recorded samples.
Post-256-token timing uses the remaining 768 answer samples (767 intervals), over 512 samples after warmup.
It is `(len(times) - 1 - 256) / (times[-1] - times[256])`, using the saved token arrival timestamps.

These are practical operating comparisons, NOT a fixed-placement attribution to faster kernels. Whole-device
availability differed: total requested VRAM was 357.20 MiB higher in the compact run, so its 2931.81 MiB larger
permanent arena is not solely the 2574.61 MiB workspace saving. The allocation-only sweep isolates that saving.
The measured RSS reduction also includes this placement/budget difference. The original run needed a placement
retry; available memory and file cache also differ, so load-time differences are not isolated loader speedups.
There is one load per binary and two requests, not a statistical repeat series. Prefill was 0.3-1.6% slower in
this pair; no claim of equal or faster prefill is made. Warm decode was about 7.6-8.5% faster in these samples.

Raw data: `baseline-128k-50k/`, `final-128k-50k/`, and `final-pair-analysis.json`. Exact commands are in each
results.json. In the final run the expert arena was accepted on the first attempt. This later success does
not erase the first implementation's four-attempt refusal or establish a fix to the loader's stabilization limit.

## Remaining other allocations, batch 4096, context 128k

| Group | MiB |
| --- | ---: |
| Shared attention/MoE workspace (peak: q_ + att_ + oa_) | 1152.000 |
| Double residual workspace r_ | 512.000 |
| cur_, xn_, ao_, fo_, g_, u_, h_ | 448.000 |
| Other arena slices: d_, router/logits, wf_, mix_/hcw_ | 21.696 |
| Raw attention rings | 177.375 |
| Compressor history rings | 13.906 |
| Allocations outside the arena: quantized inputs, routing, window logits | about 264.65 |

Rounded total: 2589.63 MiB. The 1152 MiB pool also contains the 960 MiB of logical gu_/hx_/dx_ slices at
another time; adding those again would double-count reused storage. Full-context compressed KV (860 MiB),
read rows (2048 MiB), expert staging (2848 MiB), dense weights and permanent experts are separate categories.

## Activation on the measurement PC

The machine-local `strata-dsv4-q3_k_m.json` now includes `--compact-scratch` in args. Only that flag was
inserted; context/batch/chunk, staging, reserve/loading allowance, draft/lookup defaults and cache/session
settings were preserved. The config is ignored by Git; it is not a new repository-wide default. A backup
is in the ignored build directory. The existing DeepSeek batch launcher therefore opts in on its next start.

The measured EXE above remains in `build-poly-cuda12/strata-poly.exe`; `binaries.json` records the hashes
of it and both retained controls. No test engine was left running. The CPU checks and Python syntax checks
passed after the changes (`cpu-tests.txt`). HIP/Linux/SYCL and real private sessions were not exercised.
