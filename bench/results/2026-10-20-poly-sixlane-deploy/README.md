# Six-lane selection and deployment, 2026-10-20

The user selected six-lane and requested the executable at the path already
used by the DeepSeek and GLM JSON files. The current compact head pool and eager
preparation now use the six-lane design. The preparation body matches the
historically tested `c493e79c` implementation exactly. Modern setup-time exclusion,
remaining-output proposal cap, policy diagnostics and explicit no-disk staging
size support are retained. Experimental draft-cost refresh remains opt-in.

This changes the implementation behind the existing opt-in `--compact-graphs`;
the flag-off tail index remains the same. It does not change model arithmetic,
staging budget, checkpoint settings, expert swaps or reserves. These are not new
throughput measurements or a claim that six-lane is universally faster.

## Installed executable

`L:\AI_pictures_generate\PolyStrata\build-poly-cuda12\strata-poly.exe`

SHA256: `d25903c10ca00409f83bcf5f0438fe8da6ae8b428cf45a22b9d3fb3e07c191c8`.

Both `strata-dsv4-q3_k_m.json` and `strata-glm-3.0bit.json` already point here;
neither JSON was edited. DeepSeek has `--compact-graphs` enabled, GLM does not.
DeepSeek staging remains 2848 MiB. The installed build supports changing an
explicit `--stage-mib` above one layer in no-disk mode on the next launch.

The new main executable was built in the verified clean tree, then moved to the
normal path; no duplicate main executable remains in the clean tree. The previous
validated working copy remains as `strata-poly-policy-trace-20261020.exe` in the
normal build directory. No production server was started, and no model test
process remains running. The next normal batch launch uses this executable.

## Build and validation

- CUDA 12.4, architecture 89, CUDA C++17, shared runtime, MSVC 14.41.34120.
  Existing known-working clean CMake configuration was reused.
- Fresh build of main executable, parity test, event test and no-disk CPU test
  completed all 18 scheduled steps. Nine main C++ objects retain project-header
  dependencies.
- The normal build tree still had seven objects with missing header dependencies.
  Its nine poly C++ object files were removed and rebuilt; persistent compiler
  launchers now select VSLANG=1049, matching the literal Russian include prefix.
  All 124 scheduled steps passed. Dependencies were checked again: all nine pass.
  Generated objects/build configuration are not benchmark result files.
- A dry run after installation reports `ninja: no work to do`. Installation
  refreshed the moved executable's timestamp, preserving contents and SHA256;
  its original clean-build timestamp preceded the repaired normal-tree objects.
- 1579 no-disk CPU checks passed.
- CUDA event test: upload executed no work; 1000 replays expose current host data
  before trailing GPU work.
- New numerical parity test of this exact six-lane source combination: batch
  4096, context 16384, 179 frames, **23141120 logits, zero differing bits**,
  max/mean absolute difference zero. Both graph paths exercise 1/8/31/32/33,
  window sweeps, checkpoint restore, two sequences and cancelled prefill.
  Reference ends at 2923 executions, candidate at 667; 1016 updates and zero
  replacements. This uses fixed CPU expert placement, not a new adaptive no-disk
  model-load or RAM benchmark. CPU compilation overlapped part of this numeric
  test; no speed is inferred from its wall time.
- Installed `--help` emits the expected options. CLI intentionally returns 2
  for help; the first smoke wrapper incorrectly expected zero. Both outputs
  are preserved. No program failure was inferred from that harness error.

No new GPU throughput, ordinary no-disk integration, GLM model run, or HIP/SYCL
build was performed. Prior adaptive staging tests and original six-lane tests
remain separate evidence; no new 301-slot verification is claimed for this
deployment. Header layout was not changed by the six-lane selection.

[deployment.json](deployment.json) records both unchanged config hashes, binary
and source hashes. [verification.json](verification.json) records the final
checks. Build logs, dependency reports, parity and event outputs are preserved
beside this file. [source-checks.json](source-checks.json) records equivalence
of eager preparation and the retained modern features.

## RAM from previous staging tests

[RAM.md](RAM.md) gives the requested table for the historical row-layout runs.
In the ordinary pair, maximum sampled process RSS was 114.353 versus 114.877
GiB for staging 2848/3360 MiB. Minimum sampled system available RAM was 2.683
versus 2.784 GiB, influenced by other programs. The extra staging increased
process RSS by about half a GiB; it did not save host RAM. The registered source
window is part of resident weights and is not added a second time.

Those inputs were about 10k tokens, not a full 128k request. These values are
not presented as a fresh six-lane RAM measurement or a guaranteed allocation
limit. All source test artifacts are preserved without rewriting their results.
