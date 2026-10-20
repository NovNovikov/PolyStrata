# Isolated staging comparison: one token per target-model pass

2026-10-20, Windows / RTX 4090 24 GiB / CUDA 12.4 / i5-13600KF / 128 GiB RAM.
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf. Public synthetic input and expert profile only.

The uncapping change is `3b8c1b1f`. An explicit `--stage-mib` in no-disk mode can now exceed one
layer, because the same allocation is the adaptive decode cache. Automatic staging and file-backed
staging retain their old cap. The new allocation exists before resident planning, so its cost reduces
permanent expert placement rather than increasing the allowed VRAM budget. No runtime weight migration
or second copy of permanent experts is introduced. CLI staging values must be positive integers.

## Completed pair

Same binary SHA256 in both arms:
`9b0089e349bf0bdcb9c846991f19a4d5755a607bd4c6a1e28242daaebd3a487a`.
Same test config, public expert seed and generation inputs. Window 1 in both arms prevents the
adaptive choice of 3 versus 7 from confounding this comparison. No experimental policy refresh.
Twelve warm repeats, two 512-token decodes, B/A switching and exact SAVE/RESTORE in each arm.
Both functional and memory acceptance checks passed. All 16 generated responses match exactly.

| Measurement | control-one-clean | plus512-one |
|---|---:|---:|
| Staging MiB | 2848 | 3360 |
| Permanent GPU experts | 577 | 530 |
| Adaptive slots, all filled and payload verified | 255 | 301 |
| RAM expert weights, rounded MiB | 114284 | 114795 |
| Permanent GPU expert weights, rounded MiB | 6309 | 5797 |
| Explicit device allocations MiB | 22407.86 | 22408.25 |
| Warm repeat median tok/s | 22.5521 | 23.5211 |
| Warm median cache hit rate | 54.5220% | 56.5122% |
| First 512 tok/s | 22.8941 | 22.8629 |
| Edited 512 tok/s | 21.6124 | 22.9832 |
| B switch 64 tok/s | 19.3228 | 18.7094 |
| Explicit restore 64 tok/s | 18.4259 | 18.7964 |
| Minimum sampled whole-device CUDA free MiB | 267.34 | 316.22 |
| Live executable graphs, last ten samples | 45 | 45 |
| Last ten Local Usage spread MiB | 0 | 0 |

Measured warm median change: +4.30%; hit rate: +1.99 percentage points. This is a real difference in
cache-hit counts for the same generated IDs, not a change in speculative-window length. It is still
one sequential pair with noisy timing: control warm repeats span 14.53-23.98 tok/s and candidate
18.96-24.20. No claim of a general 4.3% gain or of statistical significance is made. The first answer
is essentially unchanged and the short B-switch answer is slower. Cache warmup remains a tradeoff.
The CUDA-registered source window limit also grows with staging; this experiment measures the full
setting, not an isolated hardware cost for cache replacement alone.

Reserve and loading margin remain 256 MiB each. Context 131072, batch 4096, read chunk 32768,
compact scratch/graphs, rows in VRAM, expert swaps 64, six text checkpoints and disk sessions remain
configured. Non Local/Shared counters include registered host RAM and are not interpreted as displaced
VRAM. The 301 slots cover multiple layers; every payload and zero tail was checked against original
resident RAM. Disk-session roundtrip bytes matched within each arm. No process-restart test was run.

## Build integrity incident

The first control-one attempt was explicitly stopped with the user's permission during model load.
It is invalid, retained under control-one/ with ABORTED.md. Seven old C++ objects had zero Ninja
header dependencies, and a changed Options layout had not rebuilt architecture objects.
The ordinary launch executable was restored to a previously validated copy during the investigation.

A fresh build was made in build-poly-cuda12-clean. The initial clean configuration selected CUDA C++20
and its CUDA front-end failed; build-clean.txt preserves that failure. The successful build matches
the prior CUDA C++17 and shared runtime. Only the Russian MSVC language pack is installed, so the
English launcher attempt could not provide English messages; CMake's cached localized include prefix
was repaired from the known working prefix. Launchers now consistently request VSLANG=1049.
The current generated prefix is the literal Russian include marker, not its mojibake variant.

All 126 clean build steps completed. The dependency check now passes for all nine poly C++ objects
(deps-clean-build.txt), versus seven failures in the old tree (deps-old-build.txt). Required project
headers are recorded, including engine.hpp for architecture files. 1579 no-disk CPU checks passed.
`tools/check_poly_build_deps.py` is committed as 761119b5 and can reject the bad cache before a model
benchmark. Build logs and both the stopped attempt and failed build remain here.

The test config lives at L:\Temp\opencode\stage-split-clean-config-20261020.json; only its exe path
is changed from the current local launch config, selecting the clean build. Engine commands and
source/config/profile hashes are recorded in each results.json and summary.json. Synthetic session
payloads remain in L:\Temp\opencode\stage-split-*-20261020. No old result was deleted or overwritten.

This file records only the completed isolated pair. Ordinary-window validation has separate runs and
will be documented in another file, without changing this result. Production staging has not yet
been increased on the basis of this pair.
