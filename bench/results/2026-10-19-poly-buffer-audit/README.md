# Buffer inventory of strata-poly, 2026-10-19

Raw data for the buffer audit of `docs/POLY_VRAM_BUFFERS_AUDIT_RU.md`. This file holds what is
already measured and checked; it does not yet say which buffers are unnecessary (that is the rest of
the audit).

## Machine, model, build

- RTX 4090 24 GiB (24563.50 MiB), i5-13600KF, 12 threads, 128 GiB RAM (130906 MiB), Windows, CUDA 12.4.
- DeepSeek V4 Flash Q3_K_M: `L:\AI_pictures_generate\text-generation-webui2\user_data\models\DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`,
  135653146972 bytes, 43 layers x 256 = 11008 experts, experts 117.77 GiB.
- Engine `build-poly-cuda12\strata-poly.exe`, SHA256 `810C3AAC261CAF771D95E8C8BBD2BABBF5767CA16F7E16D8102B2542590147A3`
  (built 2026-10-19 06:09), commit `2c1dbaad`.

## `--dump-buffers`

New opt-in flag (commit `2c1dbaad`), off by default: without it nothing is recorded and nothing is
printed. It records every allocation - `dmalloc`, `hmalloc`, `kv_alloc`, `seq_alloc`, VMM - and every
slice of the architecture scratch arena, each with its call site, and prints grouped tables plus sum
checks from `print_memory()`. Because it prints from `print_memory()`, the inventory also appears on
the no-disk refusal path, where the biggest configurations end.

```text
strata-poly: buffers[<when>] card=0: device=... (plain=..., sequence=..., attention caches=...), pinned=..., scratch slices=... in ...
strata-poly: buffers[<when>] <name>   <device|pinned|sequence|attention cache>   n x size MiB = total MiB
strata-poly: buffers[<when>] scratch <name>   offset=... bytes=... (MiB) count=... <file>:<line>
strata-poly: buffers[<when>] check: scratch slices ... of ... MiB (... alignment and tail), unnamed slices ...; allocator records device=..., pinned=...; mapped KV (VMM, not in device_sizes_)=...; arena mapped=...
```

## The two runs here

### `series1-control/` - the spec's own configuration does not fit

`--no-disk --rows-vram --max-context 131072 --read-chunk 32768 --prompt-batch 4096 --stage-mib 2848
--threads 12 --vram-reserve-mib 256 --no-disk-vram-margin-mib 256 --prompt-cache 0
--conversation-cache-mib 0 --no-draft`

```text
no-disk VRAM[before-plan]: free=4020.00 MiB, engine_allocated=18678.61 MiB
no-disk: model does not fit: experts need 122779140096 RAM bytes after placing 3671606840 in VRAM;
          available RAM 117931556864, available expert VRAM 3678404608
```

`engine_allocated=18678.61 MiB` is exactly the spec's table: dense 7758.37 + experts (permanent
3221.68 at this point) + KV 860 + staging 2848 + rows 2048 + other 5164.24 = 18678.61. The refusal
is a RAM budget refusal, not a VRAM one: with the dense weights, the staging window and the read rows
already placed, the 11008 experts do not fit in the RAM that is free at that moment (112467 MiB), and
only ~3.4 GiB can be placed in VRAM.

### `dump2-ctx32768-chunk4096/` - a no-disk run that loads, at chunk 4096

Same engine, same model, `--no-disk --rows-vram --max-context 32768 --read-chunk 4096 --prompt-batch
4096 --stage-mib 2848 --threads 12 --vram-reserve-mib 256 --no-disk-vram-margin-mib 256
--prompt-cache 0 --conversation-cache-mib 0 --no-draft --dump-buffers`.

```text
no-disk plan: attempt=1, 10442 experts in RAM (114401 MiB), 566 permanent in VRAM (6191 MiB)
no-disk ready: 11008 experts total, 10442 in locked RAM, 566 direct in VRAM; locked=115420 MiB,
               read=135653146972 bytes, model files closed; VRAM free=437 MiB, host buffers=392 MiB
memory[load] card=0 context=32768 sequences=1: KV=215.00 MiB VRAM (full-context=215.00 MiB), rows=256.00 MiB VRAM (chunk=4096, batch=4096)
memory[load] card=0 VRAM allocations total=22177.49 MiB: dense=7758.37, experts=6191.77, KV=215.00, staging=2848.00, rows=256.00, other=4908.35
memory[load] card=0 RAM weights+buffers total=115813.38 MiB: resident_weights=115420.56, pinned_buffers=392.82
memory[load] card=0 whole device used=24125.87 / 24563.50 MiB, free=437.63 MiB
```

Prompt 10000 tokens: 43801.7 ms, 228.30 t/s; decode 18.33 t/s; load 216.17 s; expert hit rate 0.3234;
`model_file_open_after_load=false`, `process_read_bytes=52453` (stdin only). Generated ids
`[126889, 77070, 32901, 1229, 79, 38571, 603, 19093]` - the same eight as the reference runs, so the
instrumented build answers identically. Prefill here is read-bound (chunk 4096 reads the same expert
rows three times for 10000 tokens); the chunk was chosen to keep the run inside the free RAM, not as a
speed result, and it is not compared against other chunk sizes here.

## Checks (these are computed and compared, not assumed)

```text
buffers[load] card=0: device=22177.49 MiB (plain=20613.48, sequence=1349.02, attention caches=215.00), pinned=392.82 MiB, scratch slices=3294.68 MiB in 3294.68 MiB
buffers[load] check: scratch slices 3294.68 of 3294.68 MiB (0.00 alignment and tail), unnamed slices 0; allocator records device=22177.49, pinned=392.82 MiB; mapped KV (VMM, not in device_sizes_)=0.00 of the 215.00 requested; arena mapped=0.00 MiB
```

- device records 22177.49 MiB = the `memory[load] ... total=` line, to the byte.
- the 29 scratch slices sum to 3294.68 MiB = the arena, with 0.00 MiB of alignment/tail and no unnamed
  slice, so the scratch arena is fully explained by named slices.
- pinned records 392.82 MiB = the pinned line; mapped KV is 0.00 because no-disk sets `kv_grow=false`,
  so the attention caches are plain allocations and appear in `device_sizes_`.
- `other` is not a residual: 4908.35 = scratch arena 3294.681 + MMQ/decoder buffers 260.50
  (`mq_` 108.018, `mq2_` 54.018, `mqx_` 36.018, `q8_` 36.000, `xq_` 18.000, `sel_` 8.000, `wlogits_`
  3.945, `tab_` 0.287, `d_ids_` 0.094, `d_wts_` 0.094, `iota_` 0.094, `d_slots_` 0.084) + compressor
  rows 1171.641 + raw attention rings 177.375.

## Finding 1 - 1349.02 MiB of attention state is counted as "other", not as KV

`print_memory()` counts a buffer as KV only when it comes from `kv_alloc()` with `per > 0`. In this
run `kv_alloc()` is reached 105 times: 62 with `per > 0` (215.00 MiB, printed as KV) and 43 with
`per = 0` (177.375 MiB, the raw attention rings), and `seq_alloc()` adds 124 more slices (1171.641
MiB, the compressor rows). The last two groups are per-sequence attention state that lives as long as
the sequence does, but they are printed inside `other`. Attention state is therefore 1564.02 MiB, of
which the KV line shows only 215.00 MiB. This is the misclassification the spec predicted, with sizes.

## Inventory of the `dump2` run

Device (`DEVICE`), largest first - MiB, as printed:

| MiB | buffer | allocation site | what the size follows |
| --- | --- | --- | --- |
| 7758.371 | the dense weights | `dmalloc` in `Engine::load` | the model's non-expert tensors, one arena |
| 6191.775 | no-disk permanent experts | `dmalloc` in the placement | experts that fit in VRAM, 566 here |
| 3294.681 | scratch_ (arch scratch arena) | `seq_alloc`/`assign_scratch` | one slice per intermediate, chunk- and layer-sized |
| 2848.000 | the expert staging buffer | `dmalloc` in the no-disk setup | `--stage-mib` |
| 256.000 | the read rows (`--rows-vram`) | `dmalloc` in the no-disk setup | `--rows-vram`, chunk 4096 |
| 215.000 | the attention cache (KV) | `kv_alloc`, `per > 0` | 62 allocations, 0.250 MiB each here |
| 177.375 | the attention cache (sequence) | `kv_alloc`, `per = 0` | 43 x 4.125 MiB, raw ring per layer |
| 108.018 | mq_ (MMQ rows of x) | `dmalloc` at assign_scratch | batch x n_embd |
| 54.018 | mq2_ (MMQ rows of the expert output) | `dmalloc` at assign_scratch | batch x |
| 36.018 | mqx_ (MMQ rows of q8_tile) | `dmalloc` at assign_scratch | batch x |
| 36.000 | q8_ (quantized weights) | `dmalloc` at assign_scratch | tile of the quantizer |
| 18.000 | xq_ (quantized activations) | `dmalloc` at assign_scratch | batch x n_embd/8 |
| 8.000 | sel_ (indexer picks) | `dmalloc` at assign_scratch | |
| 3.945 | wlogits_ (window()/step() logits) | `dmalloc` at assign_scratch | |
| 0.287 | tab_ (MoE tile tables) | `dmalloc` at assign_scratch | |
| 0.094 | d_ids_, d_wts_, iota_ (row -> expert, weight, index) | `dmalloc` at assign_scratch | 3 x batch |
| 0.084 | d_slots_ (expert slots) | `dmalloc` at assign_scratch | |
| ~0 | the recurrent state, snap_, at_ | `dmalloc` at assign_scratch | 0 floats for this architecture; `at_` clamps to 256 B |

Sequence state, in `other`:

| MiB | buffer | allocation site | what the size follows |
| --- | --- | --- | --- |
| 1171.641 | the compressor's rows (124 allocations, 4.008 MiB each) | `seq_alloc` in `deepseek4.cpp` | the attention compressors of 43 layers |
| 177.375 | the raw attention rings (43 x 4.125 MiB) | `kv_alloc` with `per = 0` | window + 4096 rows x 512 values, f16 |

Pinned: 392.82 MiB in 7 allocations (`h_guess_`, `h_tab_`, `h_ids_`, the shell buffers, the queries).
RAM: resident weights 115420.56 MiB + pinned 392.82 MiB; the staging window 2848.00 MiB is CUDA-
registered inside the resident weights (`no-disk DMA verified` in the control run).

## The architecture scratch arena, slice by slice (3294.681 MiB)

| MiB | slice | formula on this model |
| --- | --- | --- |
| 256.000 | r_ (layer input rows) | batch x n_embd (4096 x 16384) |
| 64.000 each | cur_, xn_, ao_, fo_, g_, u_, h_ | batch x n_embd |
| 0.500 | d_ (MoE sums in progress) | batch x 32 |
| 4.000 | rl_ (router logits) | batch x n_expert |
| 0.493 | logits_ | |vocab| x 8 |
| 16.000 | wf_ (BF16 matrix as floats) | |
| 384.000 | gu_ (expert gate/up rows) | rows x 2 x n_ff |
| 192.000 | hx_ (expert activation rows) | rows x n_ff |
| 384.000 | dx_ (expert down rows) | rows x n_ff |
| 0.391 | mix_ (hyper-connection mix) | |
| 0.312 | hcw_ (hyper-connection weights) | |
| 16.000 | t1_ (q lora) | |
| 16.000 | qr_ (q lora normed) | |
| 512.000 | q_ (queries, then attention output) | batch x n_head x head_dim |
| 8.000 | kv_ (the key/value row) | batch x (2 x head_dim) |
| 512.000 | att_ (attention output) | batch x n_head x head_dim |
| 128.000 | oa_ (output lora a) | |
| 128.000 | ob_ (output lora b) | |
| 16.000 | ckv_ (compressor kv rows) | |
| 16.000 | cscore_ (compressor scores) | |
| 128.000 | iq_ (indexer queries) | |
| 1.000 | iw_ (indexer weights) | |
| 127.984 | score_ (indexer scores) | |

## What is not established yet

- Whether each buffer is needed, overstated, or mergeable by lifetime: not measured here. The spec's
  Phase 2 does that one buffer at a time, and each result is checked against the same eight ids above.
- Whether the same numbers hold at context 131072 and chunk 32768: the spec's configuration refuses
  on this machine while other processes hold RAM, so its inventory is only available from the refusal
  (dense 7758.37, KV 860, staging 2848, rows 2048, other 5164.24).
- Anything about ROCm/HIP or SYCL: only CUDA was measured.
