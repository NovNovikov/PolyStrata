# DeepSeek4 decode VRAM audit: measured pairs and the current candidate, 2026-10-19

The cause is demonstrated and a fix exists behind the opt-in `--compact-graphs` flag: it lifts the
sampled minimum of free device memory from 0.00 MiB to 320-354 MiB (reserve 256 MiB) while the
functional checks and the adaptive cache stay intact. **The user launch config still does not enable
the flag.** The speed question is now measured rather than assumed: the run-to-run spread of this
scenario is larger than the difference between the arms, so no speed effect of the flag is
distinguishable. The `-13.8 %` figure in the earlier version of this file did not reproduce.

## Cause

`Engine::record()` retained CUDA executable graphs for each layer, attention bucket and speculative
window/half. Reusing a key reuses its handle: this is excessive retained graph storage, not a new lost
handle on every token. CUDA's execution storage is outside `dev_sizes_`; the engine's explicit
allocations stayed at 22423.61 MiB while process GPU memory grew. Instantiation/upload after permanent
expert placement also consumed memory the expert planner had not charged.

Windows PDH Non Local Usage includes registered host DMA allocations. The approximately 3.3 GiB Non
Local figure at READY is not proof of that much displaced VRAM. Do not add Non Local and Shared
counters together.

## Hardware, model, configuration

Windows, RTX 4090, CUDA total 24563.50 MiB, driver 617.14, CUDA 12.4, i5-13600KF, 12 CPU threads,
128 GiB RAM. Model: `DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`.

Config `strata-dsv4-q3_k_m.json`, SHA256 `eb0cde6a44416d6d87ac9ba1843558fcfe565478b37ed69b38122ef75a3a50f0`:
`--compact-scratch --no-disk --rows-vram --read-chunk 32768 --prompt-batch 4096 --stage-mib 2848
--max-context 131072 --conversation-cache-mib 0 --prompt-cache 6 --conversation-cache-min-free-mib 512
--session-min-free-mib 4096 --session-max-mib 16384 --vram-reserve-mib 256 --no-disk-vram-margin-mib
256 --expert-swaps 64`. Public synthetic expert seed profile SHA256
`b06d1fcd08c83787b79f3528d2da54577e3d557151b6aa1608865199070ec6ed` (`seed-expert-profile.txt`).
`STRATA_FREEZE_EXPERTS` absent, GPU cache payload verification enabled. Only synthetic prompts and
fresh synthetic sessions were used.

## Measured runs

Scenario per run (driver `tools/bench_poly_graph_memory.py`): 10000 input + 512 output; interior edit
resuming at 4096 + 512 output; 32 repeated requests resuming at 7047 + 64 output; B/A switching;
SAVE/RESTORE; fresh-process automatic and explicit restore. 32 cycles, disk sessions on.

- `warm med` is the median of the 32 warm repeats' `decode_tps`, read from `engine/results.json`.
- `long-10k` and `interior-4k` are the two decodes of 512 tokens each (the acceptance rule asks for
  decode after warmup of at least 512 tokens).
- `execs` is the plateau of live graph executions from the `graphs[...]` line of the engine's dump.
- `min free` and `restart` are the sampled minimum of whole-device free memory in the first process
  and after the restart, against the 256 MiB reserve.

| run | binary | warm med | long-10k | interior-4k | execs | min free | restart | reserve |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `rowlayout-control/` | `449bc830` | 33.60 | 26.47 | 32.24 | 2182 | 0.00 | 0.00 | not met |
| `rowlayout-control3/` | `449bc830` | 39.12 | 30.60 | 35.17 | 1924 | 0.00 | 118.69 | not met |
| `rowlayout-control4/` | `449bc830` | 34.97 | 26.43 | 33.14 | 1625 | 0.00 | 124.26 | not met |
| `rowlayout-compact/` | `449bc830` | 38.52 | 27.29 | 36.12 | 661 | 188.50 | 421.68 | not met |
| `rowlayout-compact2/` | `449bc830` | 35.46 | 27.96 | 33.52 | 661 | 354.01 | 389.14 | met |
| `sixlane-control/` | `0f11ff30` | 35.44 | 28.00 | 33.48 | 1625 | 0.00 | 130.39 | not met |
| `sixlane-compact/` | `0f11ff30` | 34.92 | 27.86 | 33.09 | 281 | 320.88 | 347.14 | met |
| `verified/` (earlier) | `844e5037` | 35.36 | 29.43 | 35.70 | 281 | 351.61 | 307.98 | met |

`control/` in the earlier pair (binary `844e5037`, 21:04) measured 41.01 warm median, 1924 executions,
0.00 MiB minimum; `control/` is kept in this directory as evidence.

### What the numbers say

- **Memory.** The flag-off path reaches 0.00 MiB of free device memory in every run (four runs, two
  binaries) and stays there (last sample 0.00). With `--compact-graphs` the minimum is 320.88,
  351.61 and 354.01 MiB, all above the 256 MiB reserve. The fourth compact run
  (`rowlayout-compact/`) reports 188.50 MiB, but that is a single sample, not a lower plateau: its
  neighbours are 401.93 and 293.95 MiB, and the process's own committed GPU memory was constant
  across all three (PDH Local Usage 24021954560 bytes). Whole-device free memory fell while the
  engine allocated nothing, so that dip came from outside the engine and the row-layout candidate's
  steady state is the same as its other run. The free memory available at load also differs between
  runs of the same code (`no-disk VRAM[after-expert-zero]` 360.01 against 435.39 MiB, first sample
  467.68 against 434.82 MiB) although the expert plan is identical (10459 experts in RAM, 549
  permanent in VRAM, staging 2848 MiB, `engine_allocated` 22109.86 MiB). What the reserve margin
  actually depends on is therefore what else is on the device at that moment, plus one unexplained
  transient dip; it is not a property of the candidate's allocations.
- **Speed.** The spread inside one arm is larger than the difference between arms. The flag-off arm
  measured 33.60, 39.12 and 34.97 tok/s on one binary and 35.44 on another; the compact arms measured
  38.52 and 35.46 (row-layout) and 34.92 and 35.36 (six-lane). The pair that produced the earlier
  `-13.8 %` (41.01 against 35.36) sits at the top of the control spread: 41.01 has not reproduced in
  any later run. With this spread, the flag's effect on throughput is not measurable in single runs,
  and no regression is demonstrated either.
- **Retained forms.** Flag-off runs keep 1625, 1924 or 2182 live executions, different from run to
  run, with a flat plateau inside each run. Compact runs keep 281 (six-lane) or 661 (row-layout),
  identical across repeated runs.
- **Functional.** Every run filled and verified all 255 cache slots against the original RAM payloads
  (253-254 after a restart), zero tails OK, and the session file matched byte for byte across
  SAVE/RESTORE (603114484 bytes). `functional_success` is true in all nine runs. A run whose memory
  reserve is not met ends in the driver's own `assert summary['success']`; that is the acceptance
  check, not a crash, and all artifacts of such a run are kept.

## Numerical parity

Both candidates were checked against the reference path on the same fixed inputs, batch 4096, context
16384, `graphs` mode, 179 frames, 23 141 120 logits compared, with the boundaries 1/8/31/32/33, the
window sweep and the checkpoint step covered.

| candidate | parity | result | executions at the end | reference executions |
|---|---|---|---:|---:|
| six-lane (`c493e79c`) | `parity-sixlane.txt` | `bit_differences=0 max_abs=0 mean_abs=0` | 667 | 2923 |
| row-layout (`3847b548`) | `parity-row-layout-final.txt` | `bit_differences=0 max_abs=0 mean_abs=0` | 1434 | 2923 |

Command for both:

```
build-poly-cuda12\poly_scratch_parity_test.exe <MODEL.gguf> 4096 graphs 16384 graph-layout
```

The six-lane test was run with the executable rebuilt from `c493e79c`
(`build-poly-cuda12/poly_scratch_parity_test-sixlane-9e198c01.exe`, SHA256
`9e198c0125b026c698c52e123417a2224861a62f1576cee86e6ce2097a02a0ac`); the row-layout test with
`poly_scratch_parity_test.exe` (SHA256 `82cc6a9728bd9c03b07ba50b213fb1ccc72b99d61799b4df18aadf4794329977`,
kept as `poly_scratch_parity_test-row-layout-82cc6a97.exe`). In the six-lane run the candidate
prepared 538 executions and 516 templates before expert placement and ended with 667 live executions
(1491 templates, 1016 updated, 0 replaced); the row-layout candidate ended with 1434 live executions
(1987 templates, 738 updated, 0 replaced). The reference lane is 2923 executions in both runs.

Earlier files, kept:

- `parity-4096-graphs.txt`: the six-lane implementation at an earlier stage, 9178880 logits across 71
  frames, zero bit differences, batch 4096, with full and partial prefill, boundaries 1/8/31/32/33,
  speculative keep, state/row restore, two sequences and a cancelled prefill.
- `parity-final-33-graphs.txt`: the six-lane eager-upload implementation plus the setup timing tag,
  23141120 logits across 179 frames, zero bit differences, windows 2 through 8. This is NOT the
  row-layout implementation.
- `parity-row-layout-33.txt`: the row-layout candidate at boundary 33, stopped before the comparison
  finished; the completed run above supersedes it.

## Commands

```
.venv\Scripts\python.exe tools\bench_poly_graph_memory.py --config strata-dsv4-q3_k_m.json ^
  --profile bench\results\2026-10-19-poly-graph-packing\seed-expert-profile.txt ^
  --work <WORK> --output <OUTPUT> --cycles 32 --sessions [--engine-arg=--compact-graphs]
```

`<WORK>` is a fresh directory under `L:\Temp\opencode\`; `<OUTPUT>` a fresh directory under this one.
Work directories used: `gl-row-control`, `gl-row-control3`, `gl-row-control4`, `gl-row-compact`,
`gl-row-compact2`, `gl-sixlane-control`, `gl-sixlane-compact`. The full stdout of every run is kept
next to its output directory as `<name>-stdout.log`.

The six-lane runs need the six-lane executable, which the row-layout build had overwritten. It was
rebuilt from commit `c493e79c` and kept as `build-poly-cuda12/strata-poly-sixlane-0f11ff30.exe`
(SHA256 `0f11ff3031b41e384a0101b9cd228af9acc94aeae111ab6846ca532c1d87838b`); the row-layout binary
is kept as `strata-poly-row-layout-449bc830.exe` (SHA256
`449bc83040c5966e8ab8dd64ac6cbb753f2e96988bc5193c098cac7f89f49a43`, the one `strata-poly.exe` carries
now). The rebuild is not byte-identical to the original `844e5037...`, but the sources it recorded
match the earlier `verified/` run exactly:

| file | `verified/` | `sixlane-compact/` |
|---|---|---|
| `src/poly/engine.cpp` | `49032bc7ce53` | `49032bc7ce53` |
| `src/poly/engine.hpp` | `242d72516909` | `242d72516909` |
| `src/poly/deepseek4.hpp` | `ecfbdb0e8272` | `ecfbdb0e8272` |
| `src/poly/main.cpp` | `b0fd7b40e8d4` | `b0fd7b40e8d4` |
| `tools/bench_poly_graph_memory.py` | `08b6b120a5ac` | `4c9466b784cd` |

The driver differs because `3847b548` added the `--no-restart` option to it; the requests it sends are
the same. A rebuild of the same sources does not reproduce the same executable bytes
(`fa1000a3457d...` for the row-layout tree), so binaries are named by their hash.

## State of the two candidates

- `c493e79c` (six-lane): one graph per CPU-expert boundary, templates retained, executions reused
  through `cudaGraphExecUpdate`, instantiated and uploaded before expert placement. 281 executions,
  minimum free memory 320.88 and 351.61 MiB, warm medians 34.92 and 35.36 against a same-binary
  control of 35.44. Parity on this exact code: zero bit differences over 23141120 logits, 667 live
  executions against the reference's 2923.
- `3847b548` (row-layout): one execution per physical row count/offset/event layout, whole and
  first-half forms shared, context-dependent forms updated. At window 8 it keeps at most 15 head
  layouts per layer and sequence. 661 executions, minimum free memory 354.01 MiB with one transient
  188.50 MiB sample, warm medians 38.52 and 35.46. Parity: zero bit differences over 23141120 logits,
  1434 live executions against the reference's 2923.
- Default flag-off behaviour is unchanged: `3847b548` touches the shared path in one line only
  (`recorded_ = true` at the top of `record()`, engine.cpp:3181), and in the flag-off path `record()`
  is called only to capture a new graph (engine.cpp:3427, 3653, 3681, 3578, 3806), where the capture
  branch already sets that flag (engine.cpp:3198). Both flag-off runs produced identical generated
  token ids for all 36 requests.

## Not established

- What made whole-device free memory dip 165 MiB for one sample in `rowlayout-compact/` while the
  engine allocated nothing (Local Usage constant): an outside allocation, a driver effect or a
  sampling artefact. Not reproduced in the other runs.
- The effect of the flag on throughput: not measurable with single runs, because the run-to-run spread
  of the flag-off arm is 33.60-39.12 tok/s on one binary.
- No final no-graphs smoke was run. HIP and other GPU backends were not built or tested here.
- The `--compact-graphs` flag is not enabled in the user's launch config.

## Earlier trials, kept

- `baseline-trace/`: tracing baseline; trace timings and per-record whole-device deltas are not clean
  throughput or exclusive graph-byte measurements.
- `packed-trace-full/`: combining work while retaining per-form executions was insufficient; free
  VRAM fell to about 27 MiB.
- `bounded/`: bounded executions without eager charging left about 233-247 MiB free, below reserve.
- `prepared/`: stopped before READY to add cudaGraphUpload.
- `prepared-upload/`: partial run ended on artifact-write EINVAL; not an engine inference crash. Added
  bounded artifact retries and reran as `verified/`.
- `timing-verified/`: marking template updates as setup did not recover speed (warm median 35.85
  tok/s); intentionally stopped.
- `final/`: despite its directory name, a rejected experimental draft-policy cost refresh (warm
  median 36.30 tok/s). That policy modification was reverted. Its summary is not acceptance of the
  current source.
- `packed-trace/`: an initial stale-object startup crash was resolved by rebuilding engine objects;
  this is not evidence of a model failure.
- `rowlayout-control2/`: an attempt that was stopped by the user before it produced a result; only
  `engine/engine.stderr` exists there. Nothing was deleted to make room for later runs.

## Handoff

The flag-off path is the memory problem and the flag fixes it; the numbers above are the evidence.
Both candidates now have exact parity on their own code. What is left is a decision, not a cause:
which of the two compact designs goes into the launch config. On the measured numbers the six-lane
design is ahead: it keeps fewer executions (281 against 661 in the scenario, 667 against 1434 in
parity), leaves more permanent expert weights in VRAM (567 against 549), keeps its footprint
independent of the window size, and did not drop below the reserve in any run. Row-layout is the
better base for a further iteration, not the better shipping candidate. Accepting either design
needs the speed question answered with repeated alternating runs rather than single runs, because
the spread of this scenario is larger than the effect being measured. Do not shrink staging, disable
the adaptive cache, disable checkpoints or raise reserves to mask the issue.

The original already-staged upstream merge was committed separately as `f21a56f2`. Other pre-existing
working-tree changes were preserved and excluded from this work's commits.
