# DeepSeek4: explain 41 tok/s and verify graph memory fix, 2026-10-20

## Enabled outcome

The local `strata-dsv4-q3_k_m.json` now adds only `--compact-graphs` (row-layout implementation).
Its arguments and other settings were checked against a byte backup: removing that single argument
reconstructs the original parsed config exactly. `STRATA_REFRESH_DRAFT_COSTS` is NOT enabled in the
config. No production server was started; the setting applies at the next normal launch.

Original config SHA256: `eb0cde6a44416d6d87ac9ba1843558fcfe565478b37ed69b38122ef75a3a50f0`.
Updated config SHA256: `2f51ccb290de4768f89bd3d8af5b1ea229761a7eb6a69532606cb5320c889ea9`.
Backup: `L:\Temp\opencode\strata-dsv4-q3_k_m-before-compact-graphs-20261020.json`.
The config is an ignored machine-local file, not added to Git.

## Where the old 41 tok/s came from

The old `../2026-10-19-poly-graph-packing/control/engine/results.json` contains a median of
41.0143 tok/s for 32 repeated 64-token responses, median decode time 1536.05 ms. This is NOT the
initial 512-token answer's throughput (29.8406 tok/s). The driver computes `(generated - 1)*1000/decode_ms`.

The response token timestamps show one initial token followed by nine groups of seven tokens in
representative old control requests (1, 16 and 32). The counter records 54 accepted guesses. In the
later `sixlane-control/` and `sixlane-compact/`, the corresponding responses have one initial token
followed by 21 groups of three, with 42 accepted guesses. Token IDs match across these runs.
The grouping threshold used for this inspection was 20 ms between arrivals, and the counters and
source loop independently corroborate the grouping. This is a difference in decoding work, not an
incorrect throughput division. There is no recorded system-load evidence justifying attributing
the old difference to other programs on the user's PC.

`DraftPolicy` learns per-window costs from measured rounds and keeps that learning in Session across
requests. Its original stale-size probe retries only the full available proposal. With a proposal of
8, an expensive old estimate for 7 can remain untested indefinitely while 8 is retried. A numeric CPU
fixture reproduces this: of 1200 decisions, 1197 use 3 and three use 8; none uses 7 even though the
fixture's current optimum is 7 (41.1765 tok/s versus 35.7143 at 3). This demonstrates a real policy
blind spot. Original GPU runs did not record policy cost tables, so their exact historical cost state
and the initiating stall cannot be recovered or claimed as proven.

## New diagnostic and experimental fix

Commit `47398cea` adds:

- `STRATA_TRACE_DRAFT_POLICY=1`: numeric per-round proposal lengths, match lengths, confidence,
  chosen widths, accepted guesses, measured/countable time, setup exclusion and cost/sample/age tables.
  No prompt or token IDs are printed by this diagnostic. Output is buffered until decode_ms is sealed;
  collecting the trace still has a small timing cost. It is off by default.
- `STRATA_REFRESH_DRAFT_COSTS=1`: an opt-in policy that retries intermediate available widths and
  replaces stale cost estimates with a current measurement. Original policy remains the default.
- `tools/analyze_poly_policy.py`: summarizes the selected synthetic run; output uses exclusive create.

The first refresh version also filled unmeasured widths with early probes. It led to a cold-start
slowdown and was not accepted. Commit `ff016be6` narrows the change to aged estimates (more than 300
rounds since the last measurement, including priors never measured). Existing cold exploration stays
unchanged. Original choices are exactly equal in the CPU cold-start test before costs age out. Both
estimated costs and expensive measured intermediate sizes can eventually be revisited. Capture/setup
rounds excluded by Guesses do not falsely freshen a cost. Fresh subsequent samples retain normal EMA.

27 CPU policy checks and 7 Guesses checks passed. One intermediate CPU refinement incorrectly omitted
never-measured priors; a regression test failed, it was corrected before the final GPU experiment,
and both the failed and passing outputs are retained here.

## GPU measurements

Windows, RTX 4090 (24563.50 MiB CUDA total), driver 617.14, CUDA 12.4, i5-13600KF, 12 inference threads,
128 GiB RAM. DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf. The original config above, with
`--compact-graphs` passed as an extra argument: context 131072, batch 4096, read chunk 32768,
compact scratch, no-disk, rows in VRAM, staging 2848 MiB, reserve 256 MiB, load margin 256 MiB,
expert swaps 64, six text checkpoints and disk sessions. Freeze-experts removed and GPU staging
payload verification enabled. Public synthetic profile seed is copied from the earlier audit.
Only synthetic input and fresh synthetic work directories were used.

| Run | Policy | First 512 | Edited 512 | Median of 32 x 64 | B switch 64 | Explicit restore 64 | Min CUDA free MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| trace-rowlayout | original | 29.71 | 33.42 | 39.57 | 22.91 | 29.45 | 364.69 |
| refresh-rowlayout | first experiment | 26.85 | 37.61 | 37.58 | 24.78 | 23.39 | 351.47 |
| refresh-v2-rowlayout | refined experiment | 30.16 | 36.47 | 40.21 | 20.29 | 24.68 | 367.41 |

All throughput columns are tok/s. Each process did 10000 input + 512 output, edited input resuming at
4096 + 512 output, 32 requests resuming at 7047 + 64 output, B/A switching and byte-identical
SAVE/RESTORE. All 36 generated responses match between all three new runs, and between the new
original-policy run and the old control. All three functional/memory acceptance checks passed;
all 255 staging slots were filled and their GPU payloads and zero tails verified. Live graph count
plateaus at 661; Local Usage spread in the last ten repeat samples is zero.

The original-policy run additionally tested a fresh-process automatic resume at 9999 and isolated
explicit restore at 10511. Minimum sampled free VRAM after restart: 393.84 MiB; 253 slots used and
verified, capacity 255. Its complete scenario passed. The two experimental policy runs explicitly
used `--no-restart` and do NOT establish restart behavior for the new policy.

The original-policy run itself reached 41.90 tok/s in a warm repeat, so the graph method can reach
the old speed range without the new policy. The refined experiment peaks at 42.65, but its median
is only 1.6% above the new original-policy median. These sequential, different-binary runs are NOT
an alternating replicated speed study; they do not establish a general throughput improvement.
The first experiment's warm timing is additionally confounded by CPU helper builds during part
of that run. No builds overlapped the refined experiment's decode measurement.

The refined policy still loses on the short explicit-restore request: 24.68 versus 29.45 tok/s in
this pair. It remains available for experimentation, NOT enabled in the user config. Profiling
untried/stale sizes costs rounds; this is not a uniformly faster shipping default.

### Binaries and provenance

- trace-rowlayout: `f085f0df137dbe899553d00020ac73c5f108255eb15b551aad85b3a71200036f`;
  trace instrumentation, original cost policy, `STRATA_TRACE_DRAFT_POLICY=1`, refresh unset.
  Saved as `build-poly-cuda12/strata-poly-policy-trace-20261020.exe`.
- refresh-rowlayout: `75f081e1615abfb89cf2d01e9c13d2c2d1356d049200aca0e5c5b38e9443dca4`;
  `47398cea`, both diagnostic/refresh environment variables equal 1.
  Saved as `build-poly-cuda12/strata-poly-policy-refresh-v1-20261020.exe`.
- refresh-v2-rowlayout: `db8659cf0c8bcebfdb79b90d16de46574aa60ee72fe784da736fa980a9ffde1a`;
  `ff016be6`, both variables equal 1. This is the current `strata-poly.exe`; normal launch leaves
  both variables unset and selects original policy, with the config's compact graphs enabled.

The first trace-only binary predates the policy commits; its source hashes are recorded by the
run. Other run source hashes and the two environment settings are in summary.json. All commands,
generated IDs, timings, protocol replies and synthetic engine diagnostics remain in their original
run directories. Synthetic session payloads stay under `L:\Temp\opencode\policy-*-20261020`.
No results were removed or overwritten to make room for these runs.

## Limits and stopping point

The existing graph numerical parity results (23.1 million logits with zero bit differences for
each implementation) were inspected in the earlier audit; engine arithmetic was not changed here.
No new full-logit parity, no-graphs model run, HIP/SYCL build, ordinary-prose speed study or 86k-token
request was performed in this session. Today's model inputs were 10k tokens, despite the configured
131072-token capacity. This does not prove a universal 40 tok/s rate.

All launched experiments completed normally. No test or production model process remains running.
The local config enables only the graph memory fix. The experimental policy and its diagnostic
remain opt-in; old results and other people's uncommitted changes are preserved.
