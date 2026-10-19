# DeepSeek4 decode VRAM audit: interim result, 2026-10-19

Testing was stopped at the user's request. The cause is demonstrated; a fix meeting both memory and speed requirements is NOT yet accepted. The current row-layout candidate is built but unvalidated. The user's launch config has not enabled `--compact-graphs`.

## Cause

`Engine::record()` retained CUDA executable graphs for each layer, attention bucket and speculative window/half. Reusing a key reuses its handle: this is excessive retained graph storage, not a new lost handle on every token. CUDA's execution storage is outside `dev_sizes_`; the engine's explicit allocations stayed at 22423.61 MiB while process GPU memory grew. Instantiation/upload after permanent expert placement also consumed memory the expert planner had not charged.

Windows PDH Non Local Usage includes registered host DMA allocations. The approximately 3.3 GiB Non Local figure at READY is not proof of that much displaced VRAM. Do not add Non Local and Shared counters together.

## Measured separating pair

Hardware: Windows, RTX 4090, CUDA total 24563.50 MiB, driver 617.14, CUDA 12.4, i5-13600KF, 12 CPU threads, 128 GiB RAM. Model: DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf.

Both `control/` and `verified/` used executable SHA256 `844e503724cfd79a4bc6b45a14293c06581f85ac0eeb84df9f4ecdffb37a1238`, config SHA256 `eb0cde6a44416d6d87ac9ba1843558fcfe565478b37ed69b38122ef75a3a50f0`, public synthetic expert seed SHA256 `b06d1fcd08c83787b79f3528d2da54577e3d557151b6aa1608865199070ec6ed`. Only the opt-in graph mode differs. No graph tracing in this pair.

Context 131072, prompt batch 4096, read chunk 32768, compact scratch, no-disk, rows in VRAM, staging 2848 MiB, expert swaps 64, reserve 256 MiB, loading margin 256 MiB. Six text checkpoints and disk sessions enabled. Freeze-experts removed; GPU cache payload verification enabled. Only synthetic prompts and fresh synthetic sessions were used.

Scenario: 10000 input + 512 output; interior edit resuming at 4096 + 512 output; 32 repeated requests resuming at 7047 + 64 output; B/A switching; SAVE/RESTORE; fresh-process automatic and explicit restore. Exact commands, hashes, token IDs, timings and memory samples are in each run's JSON files.

| Phase | Control Local MiB | Control executions | Compact Local MiB | Compact executions |
|---|---:|---:|---:|---:|
| READY | 22870.67 | 0 | 22866.67 | 280 |
| Initial 10k prompt + output | 23171.12 | 1162 | 22913.12 | 281 |
| Interior edit | 23335.12 | 1666 | 22913.12 | 281 |
| Repeat 32 | 23385.14 | 1924 | 22913.12 | 281 |
| B switch | 23467.17 | 2264 | 22914.58 | 281 |
| A explicit restore | 23425.36 | 2522 | 22913.14 | 281 |

Peak sampled process reduction: 552.59 MiB. This includes moving 118.34 MiB of permanent expert weights back to RAM to charge graph storage before expert placement: 578 permanent GPU experts became 567 (6319.61 to 6201.27 MiB). It is not all a reduction in graph storage. Staging remains 2848 MiB with 255 slots.

Minimum sampled CUDA free memory: control 0 MiB (restart 195.89); compact 351.61 MiB (restart 307.98). Last ten warm repeat samples in compact mode have 281 executions and zero Local Usage spread. Both runs filled and verified all 255 cache slots; after compact restart 253 slots were used and verified. SAVE/RESTORE session bytes matched within each run (603114484 bytes). Functional checks passed in both; control fails the memory reserve acceptance criterion.

**Speed blocks acceptance.** Initial generation: control 29.84, compact 29.43 tok/s. Interior edit: 34.17 versus 35.70. Median of 32 warm repeats: **41.01 versus 35.36 tok/s**, a **13.8% regression**. Thus `verified/summary.json` success means the scripted functional/memory checks passed, not overall task acceptance.

## Numerical and event verification, with version limits

- `parity-4096-graphs.txt`: earlier six-lane compact implementation, 9178880 logits across 71 frames, zero bit differences, batch 4096. Includes full/partial prefill, boundaries 1/8/31/32/33, speculative keep, state/row restore, two sequences and cancelled prefill.
- `parity-final-33-graphs.txt`: six-lane eager-upload implementation plus setup timing tag, 23141120 logits across 179 frames, zero bit differences. Includes windows 2 through 8. This is NOT the newest row-layout implementation.
- `event-overlap.txt`: graph upload ran no inference work; 1000 replays consumed fresh data; host observed router completion while a trailing GPU kernel was still pending.
- No final no-graphs smoke was run. HIP and other GPU backends were not built or tested here. Default flag-off path remains the existing implementation.

## Experiments and current source state

`c493e79c` commits the six-lane design: combine adjacent GPU work around the CPU expert boundary, retain source templates, reuse executable instances via cudaGraphExecUpdate, instantiate AND upload them before expert placement. This fixes memory in the measured pair but has the speed regression above.

The current candidate instead keeps one execution per physical row count/offset/event layout, sharing identical whole/first-half forms and updating only context-dependent forms. At window 8 it has at most 15 head layouts per layer/sequence. A single-sequence count of 661 is a theoretical bound, NOT a measured acceptance result. It also bounds compact-mode speculative work by remaining requested output tokens. Default flag-off behavior is unchanged.

This candidate was built (`build-row-layout.txt`, `build-row-layout-tests.txt` in the ignored build directory). Its parity run was stopped before comparison completed, at the user's request. No row-layout benchmark result is claimed. The built main executable contains this unvalidated opt-in candidate; `strata-dsv4-q3_k_m.json` was not changed and does not activate it.

Other preserved trials:

- `baseline-trace/`: tracing baseline; trace timings and per-record whole-device deltas are not clean throughput or exclusive graph-byte measurements.
- `packed-trace-full/`: combining work while retaining per-form executions was insufficient; free VRAM fell to about 27 MiB.
- `bounded/`: bounded executions without eager charging left about 233-247 MiB free, below reserve.
- `prepared/`: stopped before READY to add cudaGraphUpload.
- `prepared-upload/`: partial run ended on artifact-write EINVAL; not an engine inference crash. Added bounded artifact retries and reran as `verified/`.
- `timing-verified/`: marking template updates as setup did not recover speed (warm median 35.85 tok/s); intentionally stopped.
- `final/`: despite its directory name, a rejected experimental draft-policy cost refresh (warm median 36.30 tok/s). That policy modification was reverted. Its summary is not acceptance of the current source.
- `packed-trace/`: an initial stale-object startup crash was resolved by rebuilding engine objects; this is not evidence of a model failure.

## Handoff

No further tests are running. Next: finish numerical parity for the current row-layout candidate, then a same-binary control/compact pair including warm throughput and the complete cache/checkpoint/restart scenario. Accept only if memory stays bounded without the measured speed regression. If this candidate fails, revise it before enabling the flag in the user launch config. Do not shrink staging, disable adaptive cache, disable checkpoints or increase reserves to mask the issue.

The original already-staged upstream merge was committed separately as `f21a56f2`. Other pre-existing working-tree changes were preserved and excluded from this work's commits.
