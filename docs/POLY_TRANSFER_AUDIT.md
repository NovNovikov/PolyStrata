# Numeric transfer diagnostics for strata-poly

Set `STRATA_TRACE_TRANSFERS=1` for an instrumented run. It is off by default. This does not change
expert placement policy, precision, routing thresholds, staging capacity or the session format.
The diagnostic records no prompts, token IDs, weights or session names. The public-input benchmark
runner separately preserves its synthetic protocol and inputs.

The engine writes aggregated JSON lines beginning with `poly-audit` after a request and at shutdown.
Each line identifies the request, phase, speculative work type, layer, read pass, source file/line,
stream and operation. `detail` is a RAM-arena offset for registration or a numeric expert identifier
for cache lifetimes. Counters are deltas at each reporting boundary; GPU intervals completing later
retain their original request ID. Add matching rows across boundaries when analyzing a request.

## What is measured

- Successful explicit H2D/D2H/D2D/H2H calls, payload bytes, failures and host API time, including 2D
  row serialization. Large async copies and stage/cache batches also have GPU stream intervals.
- `copy2` moves data through a GPU kernel rather than `cudaMemcpy`. Its pointer directions and logical
  payload are counted separately as `kernel-*-copy2`, including graph replays. These are not DMA API calls.
- CUDA graph construction is labelled `capture-template`; it is not executed transfer traffic. Pointer
  memcpy nodes and instrumented copy/permutation kernels are counted at every graph launch/update.
  Unsupported graph payloads produce explicit `unaccounted` counters rather than a false zero.
- Actual host-register/unregister API calls, bytes and source offsets, host time, and whether watched
  streams were busy before/after the call. `busy_cleared` counts busy-to-idle transitions during the API;
  `host_busy_before_ms` permits comparison of busy-entry and idle-entry calls. This is correlation, not
  proof that an internal GPU wait caused all of the API's duration.
- Whole prepare/retain spans, registration verification, identical/covered skips, explicit stream/event
  waits. Registration bytes describe mappings of original RAM, not another H2D transfer. Legacy staging
  counters remain available; the retain aggregate excludes its separately measured drain.
- First/repeated cache admissions, first-use delay, use and eviction without use, prefill invalidation,
  exchange demotions and reciprocal exchanges. Active survivors are not classified as wasted uploads.
  Speculative cache uses observed in the mixed expert route are split into accepted/rejected rows at
  `keep()`. Draft/verification work is labelled separately; offered/accepted token totals are also emitted.
- Both DeepSeek activation permutations: logical VRAM reads/writes and GPU stream intervals. Graph
  template traffic is separate from replay traffic.
- Router copy sites, host grouping/table preparation, capacity versus active table payload, and host
  activation rows unused by the selected CPU-expert branch.
- Explicit CPU memcpy/memset, checkpoint/row capture spans, session pack/unpack, logical capture bytes,
  buffered file API bytes, and live/peak conversation blobs. File API bytes are not physical disk bytes;
  blob gauges do not include every STL container, allocator overhead or driver allocation.

There are no new host synchronization barriers. Timing events and stream queries still perturb the run,
allocate diagnostic resources and add submission work. Use the result to locate costs, not as a speed
comparison with an uninstrumented executable. GPU timings are stream intervals and can contain idle time
or dependencies. Parent spans, host API times and overlapping GPU intervals must not be added together.

The mixed expert route exposes individual cache uses. Entirely GPU-routed graph paths do not return all
expert IDs to the CPU; their copies and graph launches are counted, but an absent per-expert use observation
must not be interpreted as proof that a GPU-resident expert was unused.

## One integrated run

`tools/bench_poly_transfer_audit.py` loads the model once with the configured batch/chunk/staging/context.
It uses two previously validated public 50000-token inputs, generates 256 tokens for each, returns to the
first conversation, and processes a complete server-encoded 1000-token input. All session/profile writes
go into a fresh experiment directory. It refuses an existing run and never retries or kills another engine.
Pass `--inputs`, `--profile-seed`, `--output` and `--run` explicitly. It does not invoke a compiler.

`tools/analyze_poly_transfers.py EXPERIMENT_DIRECTORY` can calculate the same numeric summary afterwards
without loading a model. It creates `transfer-analysis.json` exclusively and refuses to overwrite it.
The original JSON records, per-operation totals, call-site totals and registration comparisons are retained.

The `poly_transfer_audit_test` build target checks payload preservation, graph replay accounting, inherited
worker context, event completion and cache lifetimes without a model. Run it with the diagnostic variable set.
CUDA is validated on the local NVIDIA machine. HIP graph payload accounting emits an explicit coverage gap;
no HIP/SYCL performance or correctness claim follows from the CUDA run.
