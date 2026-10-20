# Redistributing permanent expert VRAM to adaptive staging

2026-10-20, Windows, RTX 4090 24 GiB, CUDA 12.4, i5-13600KF, 128 GiB RAM.
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf; public synthetic inputs/profile only.

An explicit `--stage-mib` in no-disk mode now permits a cache spanning layers
(`3b8c1b1f`). Growing staging allocates less space to permanent GPU experts at
load time. It does not migrate weights during generation or allocate a second
cache. Automatic staging and file-backed mode retain the previous cap.

The feature works, including payload verification beyond 255 slots. These
measurements do **not** establish a generally faster or reserve-safe production
setting. The local launch config remains at 2848 MiB, without experimental
draft-cost refresh. No additional benchmark or production server was started
after the four valid runs below.

## Ordinary speculative decoding

Same clean executable, test config, expert seed and generated inputs in both
arms; maximum window 8, selected by the existing policy. Context 131072, batch
4096, read chunk 32768, rows in VRAM, compact scratch/graphs, six checkpoints,
reserve 256 MiB and load margin 256 MiB. Two 512-token outputs and twelve warm
64-token outputs per arm. All **14 responses have identical token IDs** across
arms. Frozen experts and experimental policy refresh are disabled.

| Measurement | control-default | plus512-default |
|---|---:|---:|
| Staging MiB | 2848 | 3360 |
| Permanent GPU experts | 549 | 502 |
| Permanent expert weights, rounded MiB | 6005 | 5491 |
| Adaptive slots, filled and payload verified | 255 | 301 |
| Explicit device allocations MiB | 22109.86 | 22107.80 |
| Warm median tok/s | 36.3490 | 37.4212 |
| Warm median cache hit rate | 54.2974% | 55.3002% |
| First 512 tok/s | 27.2628 | 28.0777 |
| Edited 512 tok/s | 35.1647 | 34.2466 |
| Minimum sampled whole-device CUDA free MiB | 251.9922 | 224.6055 |
| Last ten executable graph counts | 661 | 661 |
| Last ten Local Usage spread MiB | 0.0234 | 0 |
| Functional checks | pass | pass |
| Configured 256 MiB free-reserve check | fail | fail |

The benchmark processes exit unsuccessfully because of the reserve assertion,
after completing every request. Neither engine crashed. The control misses the
threshold by 4.01 MiB, the candidate by 31.39 MiB. Explicit allocations and graph
counts plateau, but that does not identify the cause of changing whole-device
free memory. No safety margin was changed to make acceptance pass.

The ordinary warm median difference is +2.95%, with +1.00 percentage point hit
rate. It is confounded by policy history: control warm requests mostly select
window 6, while the candidate transitions from 3 to 7. The edited long output
is slower with larger staging. These are single sequential runs with timing
noise, not a demonstrated universal speedup. Background free memory also changed;
temporary checkpoint cleanup occurred during candidate loading. Both traces,
including negative acceptance results, are retained without reruns.

Machine-readable results: [comparison-default.json](comparison-default.json),
[control policy](control-default-policy.json),
[candidate policy](plus512-default-policy.json), and each run's `summary.json`,
`engine/results.json`, stderr and protocol trace.

## Isolated comparison and build integrity

[ISOLATED.md](ISOLATED.md) records the earlier fixed-window-1 comparison, where
all 16 responses match and both memory checks pass. Warm median 22.5521 to
23.5211 tok/s (+4.30%), cache hit rate 54.5220% to 56.5122%. First output was
essentially unchanged and B-switch was slower. This isolates speculative window
selection, but remains one noisy pair. The CUDA-registered host source window
also grows with staging, so it measures the complete setting.

All valid runs use SHA256
`9b0089e349bf0bdcb9c846991f19a4d5755a607bd4c6a1e28242daaebd3a487a`
from `build-poly-cuda12-clean/strata-poly.exe`. All nine C++ objects have checked
header dependencies; 1579 no-disk CPU checks passed. The invalid initial control,
failed first clean build and successful rebuild are preserved and explained in
ISOLATED.md. The ordinary production executable path was restored to the prior
validated executable, not the inconsistent incremental build.

No new HIP/SYCL build, full-logit comparison, process restart, or 86k-input test
was performed for this change. Fixed-window runs verify explicit SAVE/RESTORE
byte equality. Ordinary runs retain session settings but do not request an
explicit roundtrip; candidate automatic parking of 9999 tokens was observed.

## Requested temporary-file cleanup

After initially declining relocation, the user explicitly requested cleanup.
Eight synthetic session payloads from the completed fixed-window pair were
deleted: 4,999,137,664 bytes. After the ordinary candidate exited, its temporary
automatic session was deleted: 561,322,500 bytes. Total removed: 5,560,460,164
bytes (5.179 GiB). Files were not relocated; the active run was not interrupted.

Reports, numeric results, traces, stored session hashes and expert profiles are
preserved. The earlier ISOLATED.md describes the payloads as retained at the
time of that report; these later cleanup records supersede that statement:
[fixed-pair cleanup](cleanup-20261020.json),
[ordinary cleanup](cleanup-default-20261020.json).
The deleted payloads can no longer be independently reopened for roundtrip
verification. Unrelated project changes and other people's experiments were
not cleaned or staged.
