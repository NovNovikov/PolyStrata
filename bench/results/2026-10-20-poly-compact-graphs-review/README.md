# Retrospective review of compact CUDA graph methods

2026-10-20. Offline review of preserved public synthetic experiments from
`../2026-10-19-poly-graph-packing/`, plus source review of `c493e79c` and
`3847b548`. No GPU run, build, model restart or config change was made.
Original results and other people's report edits were preserved.

## Conclusions

Both compact designs demonstrably reduce this process's sampled Local Usage
and retained executable count. The memory result does not depend on whole-card
free VRAM. The early rejection of six-lane as a demonstrated 13.8% speed
regression was unjustified. The evidence does not establish that row-layout is
faster, or establish a universal speed cost/benefit for either design.

Six-lane is the stronger candidate for memory efficiency in the recorded ordinary
scenario: it retains 567 permanent experts versus row-layout's 549, about 195 MiB
more weights, at only 4 MiB greater warm process Local Usage. This supports further
consideration of that method; it does not establish a shipping throughput winner.
The currently enabled implementation remains row-layout.

## What each design changes

The original graph caches retain instantiated executions for combinations of
layer, attention-size bucket, window/half and sequence. Executions remain owned
until engine destruction. Repeated calls to an existing key reuse it. Growth as
new forms are encountered is excessive retained cache storage, not evidence of
an unreachable new allocation every token. The original path also plateaus in
repeated identical requests, so a plateau alone cannot prove a compact layout
is memory efficient.

The first packing-only prototype joins adjacent work around the CPU expert
boundary, but still retains an execution per form. It reduces launch separation
without solving that multiplication of executions.

Both final compact methods retain source graph templates and update a smaller
pool of executable graphs with `cudaGraphExecUpdate`. They combine previous
layer tail, current head, an external router-ready event, and the shared expert;
CPU expert scheduling still starts at that event. Graph upload/instantiation is
done before permanent-expert placement, charging real execution storage before
the planner fills the remaining device budget. It is not an extra fixed reserve.
The fallback replaces and destroys the previous executable if an update is
rejected; completed measured runs record zero replacements on this driver.

- **Six-lane (`c493e79c`):** up to six head executions per layer/sequence: whole,
  first half and second half, each with scalar/batched forms. Changing size can
  require updating the same execution again. Its small pool retains fewer
  simultaneous executions.
- **Row-layout (`3847b548`, current):** one execution for each physical row
  count/offset/event layout. Whole and first-half forms share when identical.
  At window 8 this has up to 15 compact head layouts per layer/sequence, reducing
  updates caused solely by switching window widths while retaining more executions.

This limits compact decode heads; other paths still have caches. Multi-sequence
`step()` uses separate forms, and source templates remain cached as new buckets
are visited. Therefore 281/661 are counts in the ordinary scenario, not universal
whole-engine limits. In the broader parity scenario they are 667/1434.

## Actual process consumption

Windows, RTX 4090 24 GiB, CUDA 12.4, i5-13600KF, 128 GiB RAM,
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf. Context 131072, batch 4096, read chunk
32768, staging 2848 MiB, six checkpoints and disk sessions. Every completed
main run has 36 requests, including 32 warm 64-token repeats and two 512-token
outputs. Table Local/Non Local/committed figures are medians of the last ten
warm request-boundary samples, MiB. Speed is the median of all 32 warm repeats.

| Run | Permanent experts | Warm execs | Explicit allocations | Local | Non Local | Total Committed | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| original `control` | 578 | 1924 | 22423.61 | 23385.14 | 491.125 | 23876.27 | 41.01 |
| six-lane `verified` | 567 | 281 | 22305.27 | 22913.12 | 481.125 | 23394.25 | 35.36 |
| original `sixlane-control` | 578 | 1625 | 22423.61 | 23317.12 | 480.313 | 23797.43 | 35.44 |
| six-lane `sixlane-compact` | 567 | 281 | 22305.27 | 22913.12 | 481.125 | 23394.25 | 34.92 |
| original `rowlayout-control` | 578 | 2182 | 22423.61 | 23263.49 | 671.125 | 23934.62 | 33.60 |
| original `rowlayout-control3` | 578 | 1924 | 22423.61 | 23357.20 | 518.313 | 23875.51 | 39.12 |
| original `rowlayout-control4` | 578 | 1625 | 22423.61 | 23317.12 | 481.125 | 23798.25 | 34.97 |
| row-layout `rowlayout-compact` | 549 | 661 | 22109.86 | 22909.12 | 481.125 | 23390.25 | 38.52 |
| row-layout `rowlayout-compact2` | 549 | 661 | 22109.86 | 22909.12 | 481.125 | 23390.25 | 35.46 |

Each control/candidate comparison recorded in comparison.json uses the same
executable SHA256, config hash and expert seed. Different binaries are used for
six-lane and row-layout. These are observational comparisons of sequential runs,
not randomized or alternating replicates. Their timing difference is not a
causal estimate. Outputs match for all 36 requests in every listed comparison.

The old **553 MiB** saving does survive a consumption-based check: maximum
sampled Local Usage is 23467.17 MiB in `control` versus 22914.58 in `verified`, a
552.59 MiB reduction. The later same-binary six-lane comparison saves 506.61 MiB
by that metric. Row-layout comparisons save 381.04, 446.80 or 524.57 MiB, depending
on the recorded control. These are request-boundary maxima, not continuous peaks.

The planner redistributes memory as well: permanent weights are 6319.61 MiB
without compact graphs, 6201.27 with six-lane, 6005.86 with row-layout. Thus the
total saving includes both different graph storage and fewer permanent weights.
It must not be reported as an exact measurement of graph bytes alone. Runtime
driver allocations outside the engine's explicit allocator matter too.

Non Local Usage includes registered source RAM; its legacy JSON name `spilled`
is not proof of evicted VRAM. Reporting both process Local and Non Local avoids
calling a residency change a saving. Shared and Non Local are not added together.
No whole-device-free threshold is used to judge the consumption benefit here.

One original-path run (`rowlayout-control3`) also has a late Non Local step:
518.3125 to 529.4375 MiB over its last three warm requests, with unchanged
execution count and almost unchanged Local Usage. The offline median-growth
screen flags that 11.125 MiB increase. It is a measured consumption change, not
evidence by itself of continuing leakage or a new graph execution. Both final
compact designs pass that finite-window screen in the reviewed runs. The
screen's status is kept in comparison.json rather than hidden by the table median.

## What failed in the speed interpretation

1. The first six-lane comparison really measured 41.01 versus 35.36 tok/s, but
   used different speculative work. Representative control responses (warm 1/16)
   arrive as one token plus nine groups of seven, with 54 accepted guesses.
   Corresponding compact responses arrive as one plus 21 groups of three, with
   42 accepted guesses. The 20 ms arrival-gap grouping is a diagnostic proxy,
   corroborated by the counters, not a measured kernel-launch trace. Identical
   output IDs do not imply identical verification work.
2. The later six-lane pair is 35.44 versus 34.92 tok/s (-1.46%), with matching
   representative three-token group patterns. It does not reproduce the original
   -13.8%. A small difference in one sequential pair is not enough to establish
   that six-lane is slower in general.
3. Row-layout's flag-off runs range 33.60-39.12; flag-on runs are 35.46 and 38.52.
   One available same-binary comparison gives +14.63%, another +1.40%, another
   -1.53%. Those are recorded differences, not three measurements of an isolated
   graph-layout speed effect. The original blanket rejection was unsupported;
   equality of speed or a universal improvement is also not demonstrated.
4. The two compact implementations differ beyond execution-pool size. Row-layout
   marks template updates as setup (`recorded_ = true`), excluding their time
   from adaptive policy training through `Guesses::end()`. Its `main.cpp` also
   caps proposals to remaining output tokens only when compact graphs are active.
   Six-lane lacked that cap. This changes final-round work and subsequent policy
   history, especially in repeated short replies. Comparing them is not an
   isolated test of graph storage with identical policy decisions.
5. Earlier runs lack numeric per-round cost tables. The historical state that
   led to the 7-versus-3 choices cannot be reconstructed exactly. A later policy
   audit demonstrates a stale-cost blind spot and traced choices, but cannot prove
   that precise mechanism caused every old difference. Browser/Windows activity
   was not measured in these records and is not assigned as the cause.

## Correctness evidence and limits

Both saved parity outputs report 179 frames, **23141120 logits, zero differing
bits**, maximum/mean absolute error zero. They cover batch 4096, context 16384,
1/8/31/32/33 boundaries, window sizes, halves/replay, checkpoint import, two
sequences and cancelled prefill. The fixture fixes placement with no permanent
GPU experts, no expert swaps, cpu_share=0 and draft=false. It verifies graph
arithmetic and state handling; it is not full-logit parity of the adaptive
production cache or MTP path. Integration runs separately verify all 255 adaptive
payloads/zero tails, SAVE/RESTORE bytes and generated IDs, including restarts.

The saved binaries still match their recorded SHA256: six-lane `0f11ff30...`,
row-layout `449bc830...`; parity binaries `9e198c01...` and `82cc6a97...` also
match. `source_sha256` in old run metadata hashes the working tree at run time,
not necessarily the source that produced a copied executable. In particular
sixlane-control and sixlane-compact have the same binary hash but different
working-tree source hashes. Those snapshots cannot serve as build attestations.
The later clean-build dependency verification does not retroactively certify
every old object dependency database.

The CUDA event microtests both report that graph upload executed no work and
1000 replays exposed current host data at the external router event. This is
useful focused evidence, not a replacement for full numerical parity.
HIP is explicitly rejected for compact graphs; no new HIP/SYCL validation or
86k-input experiment was done. These synthetic repetitive replies strongly
exercise prompt lookup and do not establish ordinary-prose throughput.

## Earlier prototypes reconsidered

- `packed-trace-full`: 825 executions versus baseline-trace's 1924 at matching
  early request counts. Maximum sampled Local Usage is about 23263 versus 23305
  MiB. Packing helps less than execution reuse in these records. Tracing inserts
  synchronization; its speed is not a clean production comparison. Its old
  rejection solely for low whole-device free memory is not a consumption verdict.
- `bounded`: first process completed 36 requests, about 22951 MiB sampled Local
  maximum and 235 executions after switching conversations, with all 578
  permanent experts retained. It is a promising memory result, not a demonstrated
  failure on consumption. Executions were not eagerly charged before placement,
  and the restart was deliberately interrupted; full restart acceptance is absent.
- `prepared`: stopped before READY, no performance evidence.
- `prepared-upload`: artifact-write failure after 24 requests, engine exited
  normally; completed successor is `verified`. It is not a model failure.
- `timing-verified`: intentionally stopped, incomplete integration scenario.
- `final`: an additional policy experiment, not acceptance of the unchanged
  compact-graph implementation. Its speed cannot isolate graph layout.
- `packed-trace`: stale-object startup failure, no inference performance data.

## Decision supported by this review

Keep compact graphs as a valid memory-saving technique. Withdraw the reason for
discarding six-lane based on the first 13.8% timing difference. Six-lane deserves
comparison with row-layout under equal output-limit/setup behavior and controlled
verification work; then ordinary adaptive decoding should be compared separately.
Neither switching back nor claiming a faster default follows from this review
alone. Existing row-layout config and binaries were not changed.

Reproducible offline analysis:

```
.venv\Scripts\python.exe tools\analyze_poly_compact_graphs.py --source bench/results/2026-10-19-poly-graph-packing --output <NEW_JSON>
```

[comparison.json](comparison.json) contains all nine complete-run consumption
series, exact evidence hashes, five same-binary comparisons and parity results.
[intermediate-consumption.json](intermediate-consumption.json) preserves the
prototype reanalysis. Historical reports and unsuccessful old free-memory
acceptance statuses are unchanged; the process-consumption interpretation here
supersedes those particular rejection arguments.
