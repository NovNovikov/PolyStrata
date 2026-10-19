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

`engine_allocated=18678.61 MiB` is exactly the spec's table: dense 7758.37 + KV 860 + staging 2848 +
rows 2048 + other 5164.24 = 18678.61, with **no experts in that sum** - the refusal happens before any
expert is placed. When the same parameters do place experts (`ref-ctx131072-chunk32768/`, 297 experts,
3248.01 MiB) the total is 21926.62 MiB, which is what that run reports. The refusal is a RAM budget
refusal, not a VRAM one: with the dense weights, the staging window and the read rows already placed,
the 11008 experts do not fit in the RAM that is free at that moment (112467 MiB), and only ~3.4 GiB
can be placed in VRAM.

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
| 215.000 | the attention cache (KV) | `kv_alloc`, `per > 0` | 62 allocations; the sizes differ per layer (0.250 MiB is the smallest, not every one) |
| 177.375 | the attention cache (sequence) | `seq_alloc`, one raw ring per layer | 43 slices; sizes differ per layer (4.125 MiB is the smallest) |
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
| 1171.641 | the compressor's rows (124 allocations; sizes differ per layer, 4.008 MiB is the smallest) | `seq_alloc` in `deepseek4.cpp` | the attention compressors of 43 layers; `B.ring = rows_ + overlap` rows (deepseek4.cpp:227) |
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

## Reference point - the spec's configuration, loading (`ref-ctx131072-chunk32768/`)

The same configuration as `series1-control`, run when the machine had 121321 MiB of RAM free. It
loads: 297 experts in VRAM (3248.01 MiB), 10711 in RAM (117344 MiB), locked 118363 MiB, RSS 119171.55
MiB, 4021.80 MiB of RAM still free after loading.

```text
no-disk plan: attempt=1, 10688 experts in RAM (117091 MiB), 320 permanent in VRAM (3501 MiB); available budgets RAM=119264 MiB VRAM=3508 MiB
no-disk plan: attempt=2, 10711 experts in RAM (117344 MiB), 297 permanent in VRAM (3248 MiB); available budgets RAM=119734 MiB VRAM=3250 MiB
no-disk ready: 11008 experts total, 10711 in locked RAM, 297 direct in VRAM; locked=118363 MiB, read=135653146972 bytes, model files closed; VRAM free=692 MiB, host buffers=392 MiB
memory[load] card=0 context=131072 sequences=1: KV=860.00 MiB VRAM (full-context=860.00 MiB), rows=2048.00 MiB VRAM (chunk=32768, batch=4096)
memory[load] card=0 VRAM allocations total=21926.63 MiB: dense=7758.37, experts=3248.01, KV=860.00, staging=2848.00, rows=2048.00, other=5164.24
buffers[load] card=0: device=21926.63 MiB (plain=19717.61, sequence=1349.02, attention caches=860.00), pinned=392.82 MiB, scratch slices=3550.58 MiB in 3550.58 MiB
buffers[load] check: scratch slices 3550.58 of 3550.58 MiB (0.00 alignment and tail), unnamed slices 0; allocator records device=21926.63, pinned=392.82 MiB; mapped KV (VMM, not in device_sizes_)=0.00 of the 860.00 requested; arena mapped=0.00 MiB
```

Prompt 10000 tokens: 19644.2 ms, **509.06 t/s**, one read pass (chunk 32768 >= the prompt); decode
16.88 t/s; load 236.69 s; expert hit rate 0.2198; `model_file_open_after_load=false`,
`process_read_bytes=52453` (stdin only); generated ids
`[126889, 77070, 32901, 1229, 79, 38571, 603, 19093]` again, so the reference answers are unchanged.
This is the base all later measurements are compared against.

## Second scale point - chunk 32768 at context 131072 (`dump3-ctx131072-chunk32768/`)

The same dump on `series1-control`'s parameters, run again with `--dump-buffers`; the placement still
refuses, and the refusal prints the whole inventory:

```text
buffers[placement-refused] card=0: device=18678.61 MiB (plain=16469.60, sequence=1349.02, attention caches=860.00), pinned=392.82 MiB, scratch slices=3550.58 MiB in 3550.58 MiB
buffers[placement-refused] check: scratch slices 3550.58 of 3550.58 MiB (0.00 alignment and tail), unnamed slices 0; allocator records device=18678.61, pinned=392.82 MiB; mapped KV (VMM, not in device_sizes_)=0.00 of the 860.00 requested; arena mapped=0.00 MiB
no-disk: model does not fit: experts need 122779140096 RAM bytes after placing 3671606840 in VRAM; available RAM 120875294720, available expert VRAM 3678404608
```

The RAM demand is 122779140096 bytes in both refusals, to the byte - the placement is reproducible,
only the free RAM moved (117931556864 then, 120875294720 now).

Comparing the two points (batch 4096 in both, so anything equal is batch- or model-determined):

| MiB | chunk 4096, context 32768 | chunk 32768, context 131072 | follows |
| --- | --- | --- | --- |
| KV (attention caches, `per > 0`) | 215.000 | 860.000 | context: 6.72 KiB per token at both points |
| the read rows (`--rows-vram`) | 256.000 | 2048.000 | `--rows-vram` (chunk) |
| the dense weights | 7758.371 | 7758.371 | model |
| the expert staging buffer | 2848.000 | 2848.000 | `--stage-mib` |
| the compressor's rows (124 slices) | 1171.641 | 1171.641 | `rows_` + overlap; not context (but not batch-free either) |
| the raw attention rings (43 slices) | 177.375 | 177.375 | `min(max_context, window_ + rows_)`; the context cap is not reached at either point |
| scratch arena | 3294.681 | 3550.575 | batch, except one slice: r_ 256 -> 512 |
| r_ (layer input rows) | 256.000 | 512.000 | the read chunk's rows in flight |
| gu_, dx_ (384), hx_ (192), q_, att_ (512), q8_ (36), mq_ (108) | same | same | batch and model |
| every other slice | same | same | batch and model |

So of 22177.49 MiB of device memory at chunk 4096, the part that grows with the context is the
attention caches (215 -> 860) and the read rows (256 -> 2048); the architecture scratch arena grows
by 256 MiB (one slice) because of the chunk, and the compressor rows and rings do not grow at all.

## Phase 2, buffer 1: the read rows (`--rows-vram`) - correctness-neutral, 2048 MiB of VRAM

`p2-norows-ctx131072-chunk32768/` is the reference point with `--rows-vram` removed, nothing else
changed (`--dump-buffers` on). Both load, both answer with the same eight reference ids, so the read
rows are not needed for a correct answer; they are a prefill-speed buffer, and their 2048.00 MiB take
VRAM that the placement would otherwise give to experts.

| | with `--rows-vram` | without | |
| --- | --- | --- | --- |
| rows (VRAM) | 2048.00 MiB | 0.00 MiB | chunk 32768 |
| experts permanent in VRAM | 297 (3248.01 MiB) | 508 (5555.31 MiB) | +211 experts |
| experts in RAM | 10711 (117344 MiB) | 10500 (116056 MiB) | |
| device total | 21926.63 MiB | 22185.92 MiB | |
| prefill, 10000 tokens | 19644.2 ms, **509.06 t/s** | 30891.1 ms, **323.72 t/s** | two changes at once, see below |
| effective chunk (the diagnostic's own line) | 32768 | **8192** | host rows are capped at kReadBytes |
| decode | 16.88 t/s | 18.97 t/s | +12% with more experts resident |
| expert hit rate | 0.2198 | 0.2973 | |
| load | 236.69 s | 204.13 s | |
| generated ids | reference | reference | unchanged |

This comparison changes two things at once, and the numbers above do **not** isolate the buffer:
without `--rows-vram` the diagnostic itself reports `chunk=8192` (the rows are `kReadBytes / (streams *
n_embd * 4)` = 8192 rows, so 10000 tokens are processed in two read passes instead of one at 32768).
Both figures are real for the two configurations as given, but the 509.06 -> 323.72 difference is
partly the shorter chunk, not only the placement of the rows. To separate them the chunk has to be
held equal (for example `--rows-vram` with `--read-chunk 8192`, against no rows at the same chunk);
that run has not been made. What the pair does show is that neither configuration is needed for a
correct answer: both produce the same eight reference ids.

What is measured and free of this confound: the 2048.00 MiB of VRAM that the read rows hold, and the
fact that the placement gives them to experts instead (297 -> 508 permanent, 3248.01 -> 5555.31 MiB,
hit rate 0.220 -> 0.297, decode +12%).

## Phase 2, buffer 2: the expert staging window, 2848 vs 1488 MiB

`p2-stage1488-ctx131072-chunk32768/` is the reference point with `--stage-mib 1488`, nothing else
changed. The staging window is the DMA window experts are read through; half of it frees VRAM for the
placement, and `other` does not move (5164.24 MiB at both), so staging is not part of `other`.

| | `--stage-mib 2848` | `--stage-mib 1488` | |
| --- | --- | --- | --- |
| staging (VRAM) | 2848.00 MiB | 1488.00 MiB | |
| experts permanent in VRAM | 297 (3248.01 MiB) | 432 (4719.34 MiB) | +135 experts |
| experts in RAM | 10711 (117344 MiB) | 10576 (116892 MiB) | |
| device total | 21926.63 MiB | 22037.96 MiB | |
| prefill, 10000 tokens | 509.06 t/s | **341.95 t/s** | -33% |
| decode (8 tokens, see the caveat below) | 16.88 t/s | 18.60 t/s | |
| expert hit rate | 0.2198 | 0.2641 | |
| load | 236.69 s | 236.28 s | |
| generated ids | reference | reference | unchanged |

So the trade is 1360.00 MiB of VRAM for 33% of prefill, or for about 135 resident experts. As with the
read rows, both configurations answer identically.

## Phase 2: the batch in `other` - what the two dump points do and do not show

Correction to what this section said first: both dump points had **batch 4096**, so they show what
depends on the context and the chunk, and they do **not** measure the batch. Read from the code, the
batch is in these sizes too:

- the arena is sized by `rows_ = min(prompt_batch, max_context)` (`engine.cpp:2187`), and its
  per-token slices are linear in the batch; but not every slice is: `d_`, `logits_`, `wf_` and the
  tables are constants;
- the compressor rows are `B.ring = rows_ + (ratio == 4 ? 2 : 1) * ratio` rows per block
  (`deepseek4.cpp:227`), so they do grow with the batch;
- the raw attention rings are `min(max_context, window_ + rows_)` rows per layer
  (`deepseek4.cpp:219`), so they grow with the batch until the context caps them;
- the MMQ/decoder buffers (`mq_`, `mq2_`, `mqx_`, `q8_`, `xq_`) are sized on the number of rows as
  well.
  So the statement that these are "not batch-dependent" was wrong, and `other` is not "the arena
  alone".
- only `score_` is bounded independently of the batch: `score_stride_ = max_context / 4 + 1` and
  `score_rows_` is clamped so that the buffer fits a constant (`kScoreBytes`,
  `deepseek4.cpp:236-237`), so it is <= ~128 MiB.

What the two dump points do establish is the context/chunk picture: at batch 4096 the compressor rows
(1171.641 MiB) and the rings (177.375 MiB) are the same at ctx 32768/chunk 4096 and at
ctx 131072/chunk 32768 (the rings because `window_ + rows_` is below both contexts, the compressors
because they do not follow the context at all), while the attention caches grow 215.00 -> 860.00 MiB.
A batch series (for example 1024 against 4096, same chunk) has not been run and is needed before
anything is said about the batch.

Cost of the arena per prompt-batch token, from the dumped sizes divided by the batch (measured bytes,
one sequence, batch 4096, chunk 4096, context 32768):

| KiB/token | slices |
| --- | --- |
| 128 | `q_` |
| 128 | `att_` |
| 96 | `dx_` |
| 96 | `gu_` |
| 64 | `r_` |
| 48 | `hx_` |
| 112 | `cur_`, `xn_`, `ao_`, `fo_`, `g_`, `u_`, `h_` (7 x 16) |
| 63 | `mq_`, `mq2_`, `mqx_`, `q8_`, `xq_` |
| 32 + 32 | `oa_`, `ob_` |
| 32 | `iq_` |
| ~75 | `score_` (capped), `sel_`, `iw_`, `ckv_`, `cscore_`, `kv_`, `tab_`, `mix_`, `hcw_`, `rl_`, the row-index buffers |

### `att_`: a candidate with a real obstacle, not a proven saving

Code reading (`deepseek4.cpp:242-258`, `engine.cpp:2160-2178`): `q_` and `att_` are both
`fbuf(rows_ * n_head_ * head_dim_)` - the same size, 512.00 MiB each at batch 4096 - and the comment on
`q_` (line 248) already says it holds "queries, then attention output". The attention flow is

```text
368:  lin(B.q_b, qr_, q_, n);                  // q_ = the queries
402/405: dsv4_attend(q_, ..., att_, ...);      // reads q_, writes att_
408:  dsv4_rope(att_, att_, ...);              // in place
281:  native_quantize_q8_1(att_, q8_, ...);
294:  permute_groups(att_, q_, in, ...);       // att_ -> q_
```

`att_` is a second buffer of the same size as `q_`. The obvious idea - let the attention write over
the queries it has already read and drop `att_` - does not survive the next step: `permute_groups()`
reaches `permute_kernel` (`kernels_dsv4.cu:495-503`), which is `y[b] = x[a]` over two arrays declared
`__restrict__`, so it is undefined behaviour if `x` and `y` are the same buffer, and an in-place
permutation can overwrite elements that have not been read yet. Removing `att_` therefore needs the
permutation reworked (or the attention output written straight into the grouped layout that
`permute_groups` produces), not just two pointers merged. The saving it would give - 128 KiB per
prompt-batch token, 512.00 MiB at batch 4096 - is a candidate, not a measured result.

A second, safer path for the same 512 MiB: overlap `att_` with a slice that is dead while the
attention runs and is used later in the same layer step - the expert scratch `gu_` (96 KiB/token),
`dx_` (96) and `hx_` (48) belong to the MoE phase, which comes after the attention. That does not
touch the permutation at all, and it is the variant to test first. Neither path has been implemented
or measured; both need a canary run and the eight reference ids.

## Caveat on the test harness: the adaptive expert cache was frozen

`tools/bench_poly_no_disk.py` sets `STRATA_FREEZE_EXPERTS=1` unless `--adaptive-experts` is passed
(line 45), and every run here was without it. The dumps show what that means:
`expert placement: permanent=297, staging_cache=0/255 slots (0.00 MiB used inside existing staging);
uploads=0` - the staging window holds no decode cache at all in these runs. Therefore the staging
result above (half the window, -1360 MiB, prefill -33%, more experts resident) is a measurement of the
frozen-cache mode only. In the adaptive mode the staging window *is* the decode expert cache, so
halving it would halve that capacity (`0/255` -> `0/133` slots), and nothing here measures that cost.
A series with `--adaptive-experts`, warmup and >= 512 generated tokens is required before any
statement about the staging trade in the shipped path.

## Caveat on the test harness: the decode numbers above are not speed results

Every run in this directory used `tools/bench_poly_no_disk.py` with `--generate 8`, that is eight
decode steps with no warmup. Decode t/s figures here are indicative and are not used as a speed
result anywhere; prefill figures are one 10000-token call and are not affected. The spec asks for
>= 512 generated tokens after warmup (`docs/POLY_VRAM_BUFFERS_AUDIT_RU.md`), and that change to the
harness had not been made when these runs were taken.

## What is not here

Two exploratory directories were removed before this directory's first commit, because they were not
documented and one of them ran the model in mmap mode rather than `--no-disk` (`series1b-mapped`,
and `series1-control-fit`). Only reproducible runs with their exact command in `results.json` are
kept here.

## What is not established yet

- Whether each buffer is needed, overstated, or mergeable by lifetime: one is measured (the read rows,
  correctness-neutral, with the chunk confound noted) plus the staging window in the frozen-cache mode
  only; `att_` is a candidate with a real obstacle. The rest of the arena, the compressor rows and the
  MMQ buffers have no verdict yet; the spec's Phase 2 wants one at a time, each checked against the
  same eight ids above.
- A batch series (1024 against 4096 at the same chunk, `other` sliced by the dump): not run, so the
  batch dependence of the compressor rows, the rings and the MMQ buffers is read from the formulas
  (`deepseek4.cpp:219`, `:227`, `engine.cpp:2187`) and not measured.
- A series with `--adaptive-experts`, warmup and >= 512 generated tokens: not run, so the staging
  result is not a statement about the shipped decode path.
- The per-token table above is arithmetic on dumped sizes, not a proof of each slice's formula; only
  `q_ == att_` and the arena's `n = rows_` are taken from the code.
- Decode speed after warmup over >= 512 tokens: the harness still generates 8 tokens.
- Whether the same numbers hold at context 131072 and chunk 32768 *while loading*: the spec's
  configuration refuses on this machine while other processes hold RAM (its refusal inventory is
  above, and it is the same 18678.61 MiB allocated before placement as the spec's table).
- Anything about ROCm/HIP or SYCL: only CUDA was measured.
