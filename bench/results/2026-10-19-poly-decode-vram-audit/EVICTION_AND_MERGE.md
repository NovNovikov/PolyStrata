# Graph eviction, cache bound and the merge of origin/main

Everything here was measured on the RTX 4090 box described in the main README of this directory. The
scenario is the one `tools/bench_poly_checkpoints.py` runs (10000-token prompt, 512 generated tokens,
continuation, B-switch, interior checkpoint, explicit restore, process restart), with the no-disk
configuration: `--compact-scratch --no-disk --rows-vram --read-chunk 32768 --prompt-batch 4096
--stage-mib 2848 --max-context 131072 --expert-swaps 64 --vram-reserve-mib 256`, and
`--dump-buffers`. `STRATA_VERIFY_STAGE_CACHE=1`, no `STRATA_FREEZE_EXPERTS`.

The main README (commit `1fbc7880`) located the growth in CUDA graph executions that live outside
`device_sizes_`. This file records what was tried against it, why it was dropped, and what the merge
of `origin/main` changed.

## 1. Eviction by room bucket

Built on top of `1fbc7880`: `--graph-keep N` (default 2, 0 = keep every graph) evicted a layer's
head-cache recordings whose `head_key` bucket was older than the current one by more than N. The
evicted executions were destroyed after the `cudaStreamSynchronize` at the start of the next `run()`,
so nothing could still be launching them.

Measured, 512-token scenario (`evict2/`, exe `3dbb9941…`):

- all 1344 generated IDs equal to the run without eviction (`graphs/`, exe `efac8074…`);
- execs 0 → 1033 → 1678 → 1678 → 1842 → 1842; free VRAM 0 MiB at the end, as before.

Measured with 4096 generated tokens in the first request:

- `--graph-keep 0` (`noevict-4096/`, same exe): execs 2194 after the request, slots 7196, live 2193;
- `--graph-keep 2` (`evict2-4096/`, same exe): execs 1678 after the request — 516 executions freed —
  and 2194 again after the continuation; free VRAM 0 MiB in both.

The continuation token IDs of the two 4096-token runs were identical (the driver did not write
`results.json`: it stops on a hardcoded `resume == 10511` check, see section 3).

Why it did not help in this test: the 516 freed executions were buckets 0–1; the continuation's keys
(bucket 4, and later bucket 3) were different ones, so the count returned to 2194 by recording new
keys, not by re-recording freed ones (the code reuses an existing key through `if (!made[key])`).
That does not prove a cache limit cannot be useful: the keys of the destroyed and newly created
executions and the bytes they took were not logged. The feature was reverted; `git diff HEAD --
src/poly` was empty before the merge.

## 2. Cap: do not record when rec >= recs

Hypothesis: `rec` grows past `recs = 3 * window + batch + 1` (25 here), so the head cache records one
graph per token instead of 25 per bucket. Built as `046b7ef4…`: when `rec >= recs` the launches run
eagerly and are not recorded (same result, no graph).

Measured, 4096-token first request (`cap-4096/`): execs 2194, slots 7196, live 2193 — exactly the
same as `--graph-keep 0`. The cap never fired, so `rec` never reaches `recs`.

What the numbers say: in this configuration the keys are bounded (room buckets run from 1024 tokens
up to the context, so eight; `rec` never reached `recs` in the runs made), and the 512→4096
difference is 352 executions ≈ 43 × 8. But `slots` is the `key + 1` size of a growing vector, not a
measured ceiling: it was 6680 in the 512-token run and 7196 in the 4096-token run. No repeated
warmed cycle was run, so a plateau is not proven, and the bytes per execution were never measured;
the "40–80 MiB" estimate in the first version of this file was extrapolation and is withdrawn.

The cap was a measured no-op at the first request (2194 executions with and without it). After the
continuation the counts differ (`cap-4096` 2452, `noevict-4096` and `evict2-4096` 2194), and the
likely cause is the speculative step sizes: that continuation guessed 54 rows and kept 54 in the cap
run, 56 and 53 in the other two, and `rec` is derived from the step's row count. The cap branch was
never entered (it would only add work when `rec >= recs`), so this difference is not attributed to
it; a paired rerun with the same guesses is needed before calling the runs equivalent. The cap was dropped as a
dead branch; eviction was dropped because neither its benefit nor its risk was measured.

## 3. Test-driver failures, not engine failures

- The first two 4096-token runs (`evict2-4096`, `noevict-4096`) used a temporary copy of
  `tools/bench_poly_checkpoints.py` with the first generation changed to 4096 but the hardcoded
  `resume == 10511` (10000 prompt + 512 generated − 1) left in place. Both wrote `results.json` with
  two requests (the full answer and the continuation) and then failed on that assert; the engine had
  completed both. The third (`cap-4096`) had those checks recomputed and wrote four requests, then
  stopped at `SAVE` for the disk reason below. These are driver and disk failures, not engine
  checks.
- Two earlier starts failed on Python imports (`strata_tokenizer`, `serve`) because the copy ran from
  `L:\Temp\opencode`; `PYTHONPATH` fixed it.
- The first run of the 512 scenario on the merged build refused `SAVE` with
  `SERR storage 0 the disk does not have the room the session file takes`: L: had 3.9 GiB free and
  the protocol asked for a 575,089,724-byte session (548 MiB); the 772,867,004-byte figure in the
  same log is the 14095-token session of another run. The rerun used a work directory on C:
  (22.8 GiB free) and passed; its saved session file is 603,114,484 bytes.
  The large session work directories (about 12 GiB) were deleted afterwards; the result and log
  directories under this path were kept.

## 4. Merge of origin/main

`origin/main` had six commits not in the branch: `--skip-weight F` (`5e0ab2b0`), `--stage-read-mib N`
(`bc14d967`), a `strata_bench` fix (`4bb74efb`), a community-benchmark line (`65e9309a`) and the two
merge commits (`e8a9342a`, `26e4cb5d`).

One conflict, in `Engine::alloc()` (`src/poly/engine.cpp`): the staging buffer. Resolution: their
`stage_read_mib` path applies only when `!opt_.no_disk`; under `--no-disk` the buffer stays the plain
`dmalloc`, because there `stage_` is itself the resident decode expert cache (one slot per expert,
`resident_stage_owners_`, `s->slot = stage_ + slot * slot_max_`, commit `f5adeec5`). Growing it for a
read would displace that cache rather than borrow a bounce buffer. `h_stage_` stays allocated only
outside no-disk, as before.

Build: `25e7871680bcf518c5eafbd5ae14917201b783ac2882f61187d2548bc562add4`.

Measured on the merged build, our path (512-token scenario, work directory on C:, `mergecheck-1/`):

- `CHECKPOINT_CACHE_INTEGRATION_PASSED`; all 1344 generated IDs equal to the pre-merge run;
- hit rate within 0.014 of the pre-merge run in every request;
- execs 0 → 1033 → 1678 → 1678 → 1883 → 1883, free VRAM 0 MiB at the end;
- process memory: READY 22 865 MiB → 23 135 (full cache) → 23 229 (continuation) → 23 249 (B) →
  23 259 (interior and restore), i.e. about +394 MiB over READY; spilled 3 316 MiB at READY and
  481–1 191 MiB during the requests;
- decode: 25.33 / 21.07 / 22.02 / 35.86 / 27.11 / 23.15 / 22.43 t/s against 28.07 / 23.30 / 28.67 /
  33.46 / 27.74 / 25.04 / 22.00 before the merge. Single runs, and the work directory moved from L:
  to C:, so this is not a speed comparison.

Not tested: the default (non-no-disk) path with `--stage-read-mib` or `--skip-weight`, longer
generation than 512 tokens, full-logit parity, and behaviour when VRAM is short. The merge is staged
but not committed at the time of writing.

## Raw data

Committed: `graphs/`, `nographs/` (main README).

Untracked, kept as evidence: `evict2/`, `evict2-4096/`, `noevict-4096/`, `cap-4096/`,
`mergecheck-1/`, and the aborted `mergecheck/`. All three 4096-token runs have
`initial/results.json` (`cap-4096` four requests, the other two two each); `mergecheck/` has one
request and the `SERR storage 0` refusal.
