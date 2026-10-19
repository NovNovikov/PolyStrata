# Where the VRAM goes during generation: CUDA graphs

Raw data for `docs/TASK_POLY_DECODE_VRAM_AUDIT.md`. One separating pair, same day, same machine,
same synthetic scenario, one factor changed: CUDA graphs on against `--no-graphs`.

Windows, RTX 4090 24 GiB (24563.50 MiB), i5-13600KF, 12 threads, 128 GiB RAM, CUDA 12.4.
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf. All prompts and session files are synthetic; the driver
replaces the expert profile and the session store with fresh test paths (`tools/bench_poly_checkpoints.py`).

## Configuration and binaries

`strata-dsv4-q3_k_m.json` with `--dump-buffers` added (`graphs/`), and the same with
`--dump-buffers --no-graphs` (`nographs/`). Everything else is identical:
`--serve --compact-scratch --no-disk --rows-vram --read-chunk 32768 --prompt-batch 4096
--stage-mib 2848 --max-context 131072 --conversation-cache-mib 0 --prompt-cache 6
--conversation-cache-min-free-mib 512 --session-min-free-mib 4096 --session-max-mib 16384
--vram-reserve-mib 256 --no-disk-vram-margin-mib 256 --expert-swaps 64`.
Lookup/draft at their launcher defaults (not disabled), `STRATA_FREEZE_EXPERTS` absent,
`STRATA_VERIFY_STAGE_CACHE=1`. The adaptive staging cache ran; the driver's cache and payload checks
passed. Profile: `build-poly-cuda12/checkpoints-cache-20261019/expert-profile.txt` (the public test
copy of the previous series).

- diagnostic EXE (this change, adds the `graphs[...]` line to `--dump-buffers`):
  SHA256 `efac8074da68c2d8e69502e0288363f525c94fb9ab79027fc912f0d2c7948c48`;
- the previously measured EXE `a91c6783e180947fe3c15ed1e5766d4a1cb2601b1a6de66fcca112cb8bb480bf`
  is retained as `build-poly-cuda12/strata-poly-before-graphstats.exe`;
- engine sources: commit `7a7d4a58` plus the diagnostic in this commit. Only `engine.cpp` was changed
  (a new line inside `Engine::dump_buffers`, which runs only with `--dump-buffers`).

The scenario is the checkpoint series: 10000-token A with 512 generated, SAVE, continuation of A,
unrelated B (A parked), an interior edit resuming at 4096, RESTORE and SAVE, then a second process
with an automatic resume and an isolated explicit restore. Both runs printed
`CHECKPOINT_CACHE_INTEGRATION_PASSED`.

## Graph counters (`graphs/`, printed by `--dump-buffers`)

`execs` counts every `cudaGraphExec_t` ever created; they are destroyed only in `~Engine`
(`engine.cpp:400`). `slots` is the total capacity of the three per-layer replay caches
(`L.heads`, `L.shared`, `L.tail`, indexed by `head_key * recs + rec`, `engine.cpp:3145`);
`live` is how many of those slots hold an execution. `passes` is the whole-call cache
(`Engine::passes_`), `tail` the one-token logits graph.

| phase | pos | room_key | execs | slots | live |
| --- | ---: | ---: | ---: | ---: | ---: |
| load | 0 | 0 | 0 | 0 | 0 |
| A, after 10000+512 | 10511 | 4 | 1162 | 5906 | 1161 |
| A, continuation | 10639 | 4 | 1420 | 6680 | 1419 |
| B | 10063 | 4 | 1420 | 6680 | 1419 |
| A, interior resume at 4096 | 7559 | 3 | 1584 | 6680 | 1583 |
| A, after RESTORE | 10639 | 4 | 1584 | 6680 | 1583 |
| restart, load | 0 | 0 | 0 | 0 | 0 |
| restart, automatic resume | 10063 | 4 | 646 | 5906 | 645 |
| restart, isolated RESTORE | 10639 | 4 | 1162 | 5906 | 1161 |

Reading: the number of executions grows with every new form the replay caches see
(`(room_key, rec)` pairs), is kept after the form is no longer used (last row: +164 executions after
returning from `room_key` 4 to 3), and is bounded by the slot table (6680), not by the position.
A restart does not inherit the previous process's graphs: the cache rebuilds them (`646 -> 1162`).
`batch` in that line is `opt_.batch` (extra sequences; 0 here), not `--prompt-batch`;
the label is misleading and will be renamed when the file is next built for other reasons.

## Whole-device free VRAM by phase, MiB (`cudaMemGetInfo`, same phases)

| phase | graphs | `--no-graphs` | difference |
| --- | ---: | ---: | ---: |
| load | 393.85 | 447.29 | 53.44 |
| A, after 10000+512 | 118.05 | 351.88 | 233.83 |
| A, continuation | 43.67 | 395.29 | 351.62 |
| B | 108.38 | 255.39 | 147.01 |
| A, interior resume | 0.00 | 304.98 | 304.98 |
| A, after RESTORE | 0.00 | 352.93 | 352.93 |
| restart, load | 457.50 | 445.16 | -12.34 |
| restart, automatic resume | 321.49 | 396.96 | 75.47 |
| restart, isolated RESTORE | 157.49 | 358.33 | 200.84 |

At load both processes are within 53 MiB of each other (graphs are not created yet), and the
second process's load differs by 12 MiB the other way: that is run-to-run variation of the desktop,
driver and file cache, not a graph effect. From the first A request on, the two trajectories separate:
with graphs the device ends at 0.00 MiB free (below the 256 MiB reserve), without graphs it never
leaves 255 MiB and ends at 352.93.

## Per-process GPU memory (the driver's PDH counters), MiB

| phase | graphs used | no-graphs used | difference |
| --- | ---: | ---: | ---: |
| READY | 22864.67 | 22864.67 | 0.00 |
| A, after 10000+512 | 23165.05 | 22911.09 | 253.96 |
| A, continuation | 23225.99 | 22911.13 | 314.86 |
| B | 23226.52 | 22912.55 | 313.97 |
| A, interior resume | 23289.02 | 22911.09 | 377.93 |
| A, after RESTORE | 23289.02 | 22911.09 | 377.93 |
| restart READY | 22864.67 | 22864.67 | 0.00 |
| restart, automatic resume | 22995.16 | 22861.08 | 134.08 |
| restart, isolated RESTORE | 23157.03 | 22903.34 | 253.69 |

READY is identical to the byte in both runs, which says the loader's placement is unchanged; the
difference appears with the first recorded flushes and grows with the distinct graph forms. The maximum
seen here is 377.93 MiB at the interior resume. These are per-process counters, not isolated graph
bytes: capture/instantiate may also touch the allocator pool, and a graph's device footprint is not
reported by CUDA. No `cudaMemGetInfo` fence was taken around each `record()`, so the per-form cost is
not isolated.

## What this establishes

- The engine's own explicit allocations (`dmalloc`/`kv_alloc`/`seq_alloc`/`scratch_`) do not grow
  during this series: the `buffers[...]` check lines report the same device totals with and without
  graphs. The growth the checkpoint series recorded (explicit 22423.61 MiB constant, whole-device free
  falling to 0) therefore sits outside those allocators.
- CUDA graph executions are that owner: they are created by `Engine::record()` and freed only at
  process exit, they cost device memory outside `device_sizes_`, and removing them keeps the free
  VRAM above the reserve in the same scenario.
- The growth is bounded by the replay-cache slot count (6680 slots here), not unbounded in time; but a
  series of new `room_key`/`rec` forms permanently adds executions, and returning to an earlier form
  creates more instead of reusing the old ones.

## Not established

- The device bytes per graph execution, and how much of the difference is instantiate-time allocator
  pool rather than live nodes: a per-`record()` fence would be needed (it changes timing).
- Whether evicting executions for an old `room_key`, or re-recording on demand, preserves logits and
  speed: not implemented and not measured.
- Attribution of the residual whole-device drop that remains without graphs (255 MiB at the lowest),
  which includes the driver, other desktop processes and CUDA libraries.
- Longer histories, several concurrent sequences, HIP/SYCL, and contexts other than 131072.

This is the owner map and the graphs/no-graphs separating measurement the task asks for first; no
engine layout was changed for speed or capacity. The next step is a targeted eviction/re-record
design with the full-logits comparison (`graphs` and `--no-graphs`, partial last batch, 31/32/33,
speculative keep/rollback, prefill after a generation) before anything is called fixed.
