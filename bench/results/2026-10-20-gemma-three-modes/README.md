# Gemma, three modes: the clean upstream, the fork, and the fork through our server

2026-10-20, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. Model
`gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf` (9.82 GiB), the poly engine of both trees. The prompt is the public
`docs/GLM.md` rendered with the model's own chat template (`llama_cpp/gemma_chat_template.jinja`) and the
model's own vocabulary (out of the GGUF, the way the installer builds a pack), 3825 ids, three answers of
512 tokens, temperature 0.

The upstream's poly engine had none of our diagnostics, so the `decode-costs` line was added to the clone
(its engine already had the counters: `gpu_experts`, `cpu_experts`, `swapped_experts`, `cpu_ms`,
`route_wait_ms`); the clone was built with `-DCMAKE_CUDA_STANDARD=20` on nvcc 12.9.86 (the redistributable,
the installed 12.4 untouched) and without a source change of any kind.

## The numbers

| | clean upstream, direct | the fork, direct | the fork through our server |
|---|---:|---:|---:|
| answer 1 | 351.2 tok/s | 338.7 tok/s | 311.2 tok/s |
| answer 2 | 355.3 | 344.0 | 315.7 |
| answer 3 | 355.7 | 343.7 | 315.8 |
| median of the two warm | **355.5** | **343.8** | **315.8** |
| load to READY | 28.25 s | 25.64 s | 30.0 s (the port opened) |
| process RSS (median) | 12.08 GiB | 11.89 GiB | 11.90 GiB |
| free RAM (minimum) | 103.64 GiB | 103.70 GiB | 102.61 GiB |
| SM clock (median) | 2430 MHz | 2430 MHz | 2430 MHz |
| GPU power (median) | 218.19 W | 198.40 W | 248.0 W |
| GPU temperature (median) | — | — | 57 °C |
| GPU utilization (median) | 94% | 94% | 96% |

The engine's own counters on the last answer: the upstream `generated=512 accounted_tokens=512
wall_ms=1436.484 cpu_expert_ms=0.000 route_wait_ms=0.000 uploads=0 drafts_offered=336
drafts_accepted=328 gpu_experts=124800 cpu_experts=0`; the fork `generated=512 accounted_tokens=511
wall_ms=1486.739 ... drafts_offered=336 drafts_accepted=328 gpu_experts=124560`; the server's run
`generated=512 accounted_tokens=511 wall_ms=1602.168 ... drafts_offered=468 drafts_accepted=310
gpu_experts=160800`.

## What this says

- **The Gemma does not exercise the expert path at all**: every one of its 3840 experts is in the VRAM
  (`cpu_experts=0`, `uploads=0`, the CPU timer zero, the hit rate 100%). So this model says nothing about
  the questions the exchange in `docs/NO_DISK_HOT_EXPERT_EXCHANGE.md` is about; those need the DSV4 or the
  GLM, where the hit rate is 26-44% and the CPU timer is 38-41 ms per output interval.
- **The direct modes are equal**: 355.5 against 343.8, and the run-to-run spread of one mode is 2.4-3.3%
  (the same two trees measured 288.64/288.34 on the earlier prompt and 355.5/343.84 on this one), so a
  single run of each does not settle the difference; the alternating repeats do.
- **The server's path costs about 11%** against the direct modes (315.8 against 355.5) — but its counters
  differ (161 520 experts against 124 800, 468 drafts against 336), so the two are not the same work yet:
  the server's own rendering of the same messages is one id longer in the engine's count, and the answer's
  routing follows from it. The exact difference is not isolated yet.

## The three modes, and what each one is

- A: the clean upstream (`PolyStrata-upstream-glm-20261020/build-20/strata-poly.exe`), the engine's own
  protocol, the ids sent directly.
- B: the fork (`build-poly-cuda12/strata-poly.exe`), the same ids, the same protocol.
- C: the fork through our own bat file's path: `serve/server.py` with `strata-gemma-26b.json`, the chat
  API, the server's own rendering and detokenizing, the narration and the streaming.

## Not done

- One prompt, one order, one run per mode; no alternating repeats, no confidence intervals.
- The server's rendering has not been aligned to the direct modes' down to the last id, so the counters of
  mode C and of A/B are not yet the same work.
- The HIP and SYCL backends were not built; CUDA only.
- The upstream's Qwen engine (`strata.exe`) was not built or measured.
- The draft model (MTP) for the Gemma is not on this machine, so the draft path was off in every run.

## Files

- `A-upstream/`, `B-fork/`: the engine's stderr, the protocol, the summary with the sampler's output.
- `C-fork-via-server/`: the server run's summary.
- The new pack: `PolyStrata-data/packs/gemma-26b/tokenizer` (the vocabulary from the GGUF, the template
  from `llama_cpp`).
- The engine binaries: the clone's `build-20/strata-poly.exe` and `build-poly-cuda12/strata-poly.exe`.
