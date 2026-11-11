# The same GLM test on the clean upstream build

2026-10-20, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. The task: the same public GLM test on the
clean upstream, with the same parameters and without the fork's own features (the disk cache, the no-branch
flag), so that the fork and the upstream can be compared.

## What was run

The prompt is the saved one from `../2026-10-20-glm-mmap-vs-no-disk/input.json` (2050 tokens, the public
`docs/GLM.md` rendered with the GLM chat template), so the two runs send byte-identical input. Three
answers of 512 tokens each, temperature 0, seed 42, the same diagnostics
(`STRATA_TRACE_DECODE_COSTS=1`, `STRATA_TRACE_DRAFT_POLICY=1`), a fresh copy of the public expert profile
(`data/expert-profile-glm.txt`) and a fresh session store per run.

The upstream's own parameters that the fork does not have were removed: `--dump-buffers`, `--session-dir`,
`--session-max-mib`, `--session-min-free-mib`, `--conversation-cache-no-branch`. The upstream's poly engine has
`--conversation-cache-mib`, `--prompt-cache` and `--conversation-cache-min-free-mib` (they are not fork-only),
so those stayed, as in the maintainer's control run.

## The numbers

| | request 1 | request 2 | request 3 | median of 2-3 |
|---|---:|---:|---:|---:|
| clean upstream, built here with C++20 | 24.75 tok/s (20649 ms) | 28.74 (17782 ms) | 26.82 (19052 ms) | **27.78 tok/s** |
| the fork, `build-poly-cuda12/strata-poly.exe` | 27.77 tok/s (18404 ms) | 28.10 (18182 ms) | 25.45 (20076 ms) | **26.78 tok/s** |

The drafts accepted/offered: the upstream 305/348, 336/374, 299/341; the fork's run is in
`../2026-10-20-glm-fork-clean/`. Both runs were consecutive on the same machine with the RAM free, so the
file cache held the weights in both.

The fork's VRAM accounting is the same as the maintainer's mmap run: 21985.06 MiB (dense 5622.06, experts
11858.00, KV 72.00, staging 2048.00, rows 0.00, other 2385.00).

## The reading

The fork and the clean upstream differ by 1 tok/s on the median of the two warm answers, and the spread
inside one run is 2.6-4.0 tok/s. So on this test, in these conditions, the fork's changes to the mmap path
do not cost speed. The maintainer's earlier numbers (23.23 tok/s for the mmap control, 22.19 for the
no-disk mode) were taken with 4.75 GiB of RAM free, while the file held most of the weights; these runs had
the machine free, which is why they are higher.

## How the upstream was built

The upstream's poly engine needs C++20 in its CUDA sources, and `cudafe++` 12.4 crashes on C++20 (our
`build-engine.bat` says so), so the upstream does not build with the toolkit on this machine. The compiler
came from NVIDIA's redistributable instead, and the system toolkit was left alone:

- `cuda_nvcc` 12.9.86 (121 MiB, SHA256 227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85, the
  checksum in NVIDIA's manifest for 12.9.1);
- `cuda_cudart` 12.9.79, `cuda_cccl` 12.9.27, `libnvfatbin` 12.9.82 for the headers and the import
  libraries; the rest of the CUDA libraries come from the installed 12.4.

The clone was configured with `-DCMAKE_CUDA_STANDARD=20` and built with that compiler: 123 of 123 targets,
no source change of any kind.

## What is not done

- The upstream's Qwen engine (`strata.exe`) was not built or measured; only the poly engine was, because the
  GLM test needs it.
- The upstream's HIP and SYCL backends were not built; CUDA only.
- The runs are one prompt, one order, three answers; no confidence intervals.
- The fork's no-disk mode was not run here: the upstream has no `--no-disk` in its poly engine, so that
  column of the maintainer's table has no upstream counterpart.
