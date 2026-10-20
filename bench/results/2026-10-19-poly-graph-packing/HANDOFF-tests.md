# Handoff: tests run on the row-layout graph candidate (2026-10-19)

> SUPERSEDED. The measured results and the conclusions are in `README.md` of this directory, written
> after nine runs instead of two. This file is the working note of the first session and is kept for
> the record; where it disagrees with `README.md`, `README.md` is the later measurement.

Written for the next agent taking over. Branch `experiment/disk-checkpoints`, HEAD `3847b548`. Nothing below
is acceptance: the compact half of the pair was NOT run here, and no same-session pair exists.

## State inherited

- `f21a56f2` merged `origin/main` and preserved the resident no-disk staging cache.
- `c493e79c` (six-lane graphs) fixes memory in the measured pair but costs -13.8% warm throughput
  (median 41.01 -> 35.36 tok/s), so it is not accepted. Details in this directory's `README.md`.
- `3847b548` is the current row-layout candidate. Parity was stopped at handoff; the default
  flag-off path is unchanged by design. `strata-dsv4-q3_k_m.json` does not enable `--compact-graphs`.

## Binaries verified as current

Rebuilt `strata-poly` and `poly_scratch_parity_test` from HEAD with
`cmake --build build-poly-cuda12 --target strata-poly poly_scratch_parity_test` (ninja reported no work,
so the tree was already built from this source):

- `build-poly-cuda12/strata-poly.exe`  SHA256 `449bc83040c5966e8ab8dd64ac6cbb753f2e96988bc5193c098cac7f89f49a43`
- `build-poly-cuda12/poly_scratch_parity_test.exe`  SHA256 `82cc6a9728bd9c03b07ba50b213fb1ccc72b99d61799b4df18aadf4794329977`

## Numerical parity: PASS

Command (model is the Q3_K_M GGUF, standard path, context 16384):

```
build-poly-cuda12\poly_scratch_parity_test.exe <MODEL.gguf> 4096 graphs 16384 graph-layout
```

Result (`parity-row-layout-final.txt` in this directory):

```
PASS: batch=4096 graphs=1 frames=179 logits=23141120 bit_differences=0 max_abs=0 mean_abs=0
```

Both lanes completed boundaries 1/8/31/32/33, the window sweep and the checkpoint step. Executions at the
end: reference layout 2923, row-layout 1434 (the candidate halves the count in this test). No bit
difference, so the row-layout candidate is numerically correct on this run.

## Control run (flag OFF) - invalid as an isolated test

Command:

```
.venv\Scripts\python.exe tools\bench_poly_graph_memory.py --config strata-dsv4-q3_k_m.json ^
  --profile bench\results\2026-10-19-poly-graph-packing\seed-expert-profile.txt ^
  --work L:\Temp\opencode\gl-row-control ^
  --output bench\results\2026-10-19-poly-graph-packing\rowlayout-control --cycles 32 --sessions
```

`rowlayout-control/summary.json`: exe `449bc830...`, config `eb0cde6a...`, seed profile `b06d1fcd...`; 32
cycles; engine `minimum_cuda_free_mib` 0.0 with `reserve_mib` 256 (`reserve_satisfied` false),
`cache_used` 255 / `verified_payloads` 255, `last_ten_graph_counts` all 2182 (`plateau` true),
`last_ten_local_spread_mib` 0.02; restart min free 0.0, 253/253 verified; `functional_success` true,
`success` false (memory only).

Two faults with how this was run, which the next agent must not repeat:
1. It is one half of a pair. Without the compact run on the SAME binary it proves nothing about
   `--compact-graphs`. It only shows the flag-off path of this binary: plateau 2182, cache verified.
2. Its speed was not read out, not missing: `engine/results.json` carries `decode_tps` per request
   (plus `decode_ms`, `prompt_ms`, `token_times`). The 32 warm repeats of this run have a median of
   33.60 tok/s. `summary.json` carries no speed and stdout was filtered, so the numbers have to be
   read from the per-request JSON, not from stdout and not from `summary.json`.

Also: a first control run was made with the work directory on C:, then deleted and rerun on L:. Both were
the same invalid standalone test; the first result is lost. See "Never destroy work in progress" in
`AGENTS.md`.

## The flag-off path of 3847b548 is unchanged; 2182 vs 1924 is not a code difference

Checked: the only shared-code change of `3847b548` is `recorded_ = true` at the top of `record()`
(engine.cpp:3181). In the flag-off path `record()` is called only to capture a new graph
(engine.cpp:3427, 3653, 3681, 3578, 3806), and the capture branch already sets the same flag
(engine.cpp:3198), so the line changes nothing there. Everything else in the commit is behind
`opt_.compact_graphs`. The two flag-off runs also produce identical generated token ids for all 36
requests, which is what an unchanged path would do.

So 2182 vs 1924 executions and 33.60 vs 41.01 tok/s are not a code difference. The runs are from
different sessions 80 minutes apart (21:04 and 22:25) under different machine conditions, and they
are not comparable to each other. The six-lane pair's own 41.01 vs 35.36 is a within-session,
back-to-back comparison of one binary (compact 20:52-21:00, control 21:04-21:12, 4 minutes apart,
same config, same profile, same machine) and stands as measured: the six-lane design costs 13.8 % of
the warm median.

Consequence for the next agent: measure the new pair back to back inside ONE session, and take the
row-layout candidate's control from that same session, not from 20:52 or 22:25.

## What the acceptance needs (not done)

1. Run the pair on the SAME binary `449bc830...`: control (no extra arg) and compact
   (`--engine-arg --compact-graphs`), same config, same profile, work under `L:\Temp\opencode\`, and
   capture the FULL stdout to a log file (do not filter). Run them back to back in ONE session; take
   the candidate's control from that same session.
2. Compare memory: compact must keep CUDA free at or above the 256 MiB reserve (this is what the
   six-lane fix achieved at 351.61 MiB minimum) and must not increase permanent-GPU experts by hiding
   weights in RAM unnoticed (`--compact-graphs` must not shrink staging, disable the adaptive cache,
   disable checkpoints or raise reserves).
3. Compare speed: warm medians of the 32 repeats, control vs compact, same binary, same session.
   Read the numbers from `engine/results.json` (`decode_tps` per request).
4. Accept only if parity stays 0 bit differences (done for row-layout), memory is within the reserve, and
   the warm median does not regress beyond noise.

## Evidence kept (do not delete)

- `parity-row-layout-final.txt` - the PASS above.
- `rowlayout-control/` - engine stderr, protocol, per-request results.json and summary.json of the
  flag-off run.
- `parity-4096-graphs.txt`, `parity-final-33-graphs.txt`, `parity-row-layout-33.txt`, `event-overlap.txt`,
  `event-upload.txt`, `policy-refresh-test.txt` - from the other chat's work.
- Earlier graph-eviction and merge results: `bench/results/2026-10-19-poly-decode-vram-audit/`
  (`README.md`, `EVICTION_AND_MERGE.md`, `graphs/`, `nographs/`, and the untracked `evict2/`,
  `evict2-4096/`, `noevict-4096/`, `cap-4096/`, `mergecheck/`, `mergecheck-1/`).
