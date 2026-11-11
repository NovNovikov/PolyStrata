# Bounded DMA coalescing and preparation before attention

Started 2026-10-21, completed 2026-10-22, Windows / RTX 4090 / 128 GiB RAM.
Code commit `7a2cf777`; opt-in `--coalesce-expert-dma`. CUDA executable is at the
existing configured path `build-poly-cuda12/strata-poly.exe`.

Only the changed variant was launched: ONE model load and four requests, no retry,
no baseline rerun. Control is the existing `../2026-10-21-poly-transfer-audit/`.
The input and initial-profile hashes match, as do the 1000-token short input and
the startup placement (10441 RAM / 567 permanent GPU experts). Geometry remains
context 131072, batch 4096, read-chunk 32768, stage 2848 MiB, reserve/margin 256 MiB,
compact scratch/graphs and rows-vram, adaptive cache and CPU share.

## Canonical data

- `combined/engine.stderr`: own synthetic experiment's original numeric trace.
- `combined/results.json`, `samples.jsonl`: timing and per-process telemetry.
- `manifest.json`: executable/input/profile hashes and exact command.
- `transfer-analysis.json`: complete numeric aggregation, 22322 records, zero coverage failures.
- `comparison-v2.json`: correct comparison with the existing control.
- `compare.py`: offline calculation only; never starts a model.
- `build.log`, `self-test-audit.stderr`, `verification.json`: build/test evidence.

`comparison.json` and `comparison-console.txt` are retained INVALID first-draft
calculations: a generator was exhausted after summing calls, leaving other fields
zero. Use v2. Neither raw model results nor transfer-analysis.json were affected.

## Actual changes

1. A whole layer's RAM-source span is registered before submitting its attention,
   only if the page-rounded span fits the existing DMA limit. Already-covered
   routed subsets reuse that registration. A layer that does not fit falls back
   to preparation at staging time. There is no raised memory limit.
2. Selected staging ranges can cover gaps with one bounded registration. When
   the covering span exceeds the limit, exact source pages remain the fallback.
3. Adjacent gate/up are submitted as one H2D, down separately. Destination
   layout, padding, zero tail and source-reader event fencing are unchanged.

No additional weight arena, GPU buffer or CPU repacking was introduced. Default
execution without the flag retains the original policy. The flag is present in
the measured command. After validation it was enabled in the local DeepSeek
launcher JSON for the next normal start; GLM was not changed. `launcher-before.json`
preserves the original configuration; `launcher-activation.json` records that
the flag is the only configuration difference. No model was restarted for this.

## Timing

| Request | Control prompt, s | Changed prompt, s | Control decode, tok/s | Changed decode, tok/s |
|---|---:|---:|---:|---:|
| A, 50000 / 256 | 95.331 | 93.878 | 16.38 | 16.28 |
| B, 50000 / 256, excluding parking | 96.064 | 89.679 | 17.45 | 16.93 |
| Return A, resume=49999/read=1, excluding parking | 2.068 | 1.551 | 21.78 | 17.21 |
| Short, 1000 / 128 | 15.319 | 10.008 | 15.56 | 16.80 |

B raw prompt is 147.283 -> 91.029 s. Control includes 51.219 s of
observer-distorted packing; changed includes 1.350 s parking. Return A similarly
includes 52.598 / 1.255 s parking. That large apparent reduction is the already
fixed scalar instrumentation error, NOT a DMA optimization. Excluded values use
the measured outer session-park-total span; nested packing spans are not added.

This is one changed run compared with an earlier control, not an interleaved
speed study. Long A is only 1.5% shorter; a stable speedup of that size is not
established. Decode does not show a consistent improvement. Adaptive placements,
CPU share and speculative acceptance vary with execution timing; this was not a
fixed-token/fixed-routing test. Full model logits parity was not measured.

## Operations removed and time observed

| Main prefill only | Control A | Changed A | Control short | Changed short |
|---|---:|---:|---:|---:|
| Register calls | 86 | 85 | 1410 | 43 |
| Unregister calls | 86 | 85 | 1390 | 43 |
| Register host API, s | 0.963 | 0.848 | 0.791 | 0.424 |
| Unregister host API, s | 12.518 | 4.240 | 10.349 | 1.372 |
| Expert H2D calls | 62646 | 41764 | 5883 | 8968 |
| Expert H2D bytes | 239896100864 | 239896100864 | 22615162880 | 51482132480 |

On B, register calls remain 86; unregister host API is 12.645 -> 3.959 s.
A saves an additional registration by reusing the covering window left by load.
All changed main unregister calls still enter with watched streams busy and
return with them ready: the GPU-reader fence was not removed.

The much smaller unregister API sum is not equivalent wall-time saved. Some
waiting moved before attention submission; GPU work and copy intervals overlap.
On A the attention-head stream intervals are 60.274 -> 64.280 s and expert-copy
intervals 10.083 -> 12.953 s. These include contention/dependencies and cannot be
added into a wall-time budget. Gate/up API submission itself is 0.385 -> 0.252 s.

The short request stages 1961 -> 4484 experts: automatic CPU share changes its
balance with the new staging cost. Its H2D volume therefore INCREASES from
21.062 -> 47.946 GiB despite two copies per expert. Do not describe this result
as an unchanged-traffic short-prefill comparison. The normal selection algorithm
and configured CPU-share policy were not changed or frozen.

Short registration coverage grows from 22615162880 to 119948050432 aggregate
bytes across layers. These are page mappings of existing RAM, not H2D traffic or
simultaneously allocated new memory. The same approximately 2850 MiB window limit
applies. Larger mapped gaps and their driver overhead remain a tradeoff.

## Memory and correctness

Explicit engine VRAM allocation at placement is unchanged: 22306.09 MiB. Compare
process counters, not global free VRAM: overall sampled GPU asked peak is
25.6267 -> 25.6281 GiB, working-set peak 114.0245 -> 114.1440 GiB, private commit
peak 136.6081 -> 136.6071 GiB. These are different counters, not additive memory.
Short-prefill GPU asked peak rises 24.9300 -> 25.6253 GiB; equal whole-run peaks
do not establish equal driver overhead at each phase. Per-phase counters remain
in comparison-v2.json; no claim of zero mapping overhead is made.

Model exit 0; both long requests read 50000 and generated 256, return A resumed at
49999/read 1 and generated 32, short read 1000/generated 128. The ordinary adaptive
cache remained active; it was not frozen or replaced with permanent experts only.

CUDA build passed. `poly_no_disk_test`: 2485 checks. GPU DMA probe: two streams,
partial-registration rollback, covered-subset reuse, invalid-range preservation,
bounded-span fallback and byte-identical original/merged staging (including gaps
and tail). GPU audit self-test passed. HIP and SYCL were not built/tested. The
canonical exchange path was not model-tested in this series. No full-model
bitwise logits or graph/no-graph equivalence claim is made from these tests.

No baseline/model retry was launched, no other engine stopped, no user session
or private prompt read, no previous raw results deleted or overwritten.
