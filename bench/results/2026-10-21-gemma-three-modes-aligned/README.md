# Gemma, three modes with one prompt: the clean upstream, the fork, and the fork through our server

2026-10-21, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. Model
`gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf` (9.82 GiB), the poly engine of both trees. The prompt is the public
`docs/GLM.md` rendered with the model's own chat template (`llama_cpp/gemma_chat_template.jinja`) and the
model's own vocabulary (out of the GGUF, the way the installer builds a pack), three answers of 512 tokens,
temperature 0, seed 42.

This run replaces the numbers of `2026-10-20-gemma-three-modes`: there the three modes did not get the same
prompt, so they did not do the same work. The reason is below, and it is one flag.

## The same prompt in all three modes

The server renders a chat prompt and encodes it in two steps (`serve/server.py`):

- `render_prompt()` (server.py:4538): the template with the tokenizer's own `bos_token` / `eos_token` and
  `enable_thinking` off, then `gemma_model_turn()`.
- `_encode_rendered_prompt()` (server.py:4574): `tok.encode(prompt, parse_special=True)`.

The `parse_special=True` matters. Without it the tokenizer treats the control literals as text, and the
leading `[gMASK]` is split into three byte pieces instead of the one special token (id 154822): the direct
modes then sent 3825 ids where the server sends 3824. The prompt was one token different, the answer was a
different text, and the routing followed the text: the direct modes routed 124 800 experts with 336 drafts,
the server 160 800 experts with 468 drafts. Those numbers were not a difference between the engines; they
were two different questions.

With the server's own rule (`parse_special=True`), all three modes send the same 3824 ids, all three produce
the same answer (sha256 of the text, first 16 hex: `948d53f8511cc431`, in all nine answers), and all three
show the same engine counters.

## The numbers

| | clean upstream, direct | the fork, direct | the fork through our server |
|---|---:|---:|---:|
| answer 1 | 317.2 tok/s | 316.2 tok/s | 306.3 tok/s |
| answer 2 | 317.6 | 317.1 | 312.0 |
| answer 3 | 316.6 | 316.4 | 310.5 |
| median of the two warm | **317.2** | **316.2** | **311.3** |
| prompt tokens (the engine's count) | 3824 | 3824 | 3824 |
| load to READY / the port opened | 25.24 s | 28.04 s | 29.1 s |
| free RAM (minimum) | 102.64 GiB | 102.82 GiB | 102.48 GiB |
| SM clock (median) | 2430 MHz | 2430 MHz | 2437.5 MHz |
| GPU power (median) | 228.21 W | 232.01 W | 240.27 W |
| GPU temperature (median) | 52 °C | 56 °C | 52.5 °C |
| GPU utilization (median) | 96% | 94.5% | 94.5% |
| answer's text | sha16 `948d53f8511cc431` | sha16 `948d53f8511cc431` | sha16 `948d53f8511cc431` |

The engine's own counters on the last answer, all three modes:
`generated=512 accounted_tokens=511 wall_ms=1641.5..1649.2 cpu_expert_ms=0.000 route_wait_ms=0.000 uploads=0
drafts_offered=468..471 drafts_accepted=310 gpu_experts=160800..161520 cpu_experts=0`.

Mode C's table row is the server's own rate (the chunks of the stream, from the first to the last). The
engine's own `wall_ms` in that run was 1641.5 and 1649.2 ms, against 1609.5 and 1612.3 ms in mode A and
1617.1 and 1615.3 ms in mode B. The engine works the same in all three; what the server adds is the
token-by-token narration and the streaming, about 2% of the answer's time.

## The alternating repeats (A B B A)

One run of each does not settle a difference of a percent or two, so the two trees were run alternately, the
same ids, the same command line (`2026-10-21-gemma-ab-interleaved/`):

| run | tok/s (median of the two warm) |
|---|---:|
| A1 upstream | 317.62 |
| B1 fork | 316.60 |
| B2 fork | 316.44 |
| A2 upstream | 317.13 |

Upstream 317.38, fork 316.52: the fork is 0.27% behind, inside the run-to-run spread of one mode
(316.4-317.6 across the four runs). **The direct modes are equal.**

## What this says

- **The Gemma does not exercise the expert path at all**: every one of its 3840 experts is in the VRAM
  (`cpu_experts=0`, `uploads=0`, the CPU timer zero, the hit rate 100%). So this model says nothing about
  the questions the exchange in `docs/NO_DISK_HOT_EXPERT_EXCHANGE.md` is about; those need the DSV4 or the
  GLM, where the hit rate is 26-44% and the CPU timer is 38-41 ms per output interval.
- **The fork is not slower than the upstream on the direct path** (0.27%, inside the noise). The 3.4% of the
  earlier run was the different prompt, not the fork.
- **The server's path costs about 2%** against the direct path (311.3 against 317.2 tok/s), and its overhead
  is outside the engine: the engine's own `wall_ms` is the same in all three modes.

## The three modes, and what each one is

- A: the clean upstream (`PolyStrata-upstream-glm-20261020/build-20/strata-poly.exe`), the engine's own
  protocol, the ids sent directly: `GEN 512 temperature=0 seed=42 <3824 ids>` on stdin.
- B: the fork (`build-poly-cuda12/strata-poly.exe`), the same ids, the same command line.
- C: the fork through our own bat file's path: `serve/server.py` with `strata-gemma-26b.json`, the chat API
  (`POST /v1/chat/completions`), the server's own rendering, encoding, detokenizing, narration and streaming.
  The request says `chat_template_kwargs: {"enable_thinking": false}` and `seed: 42`, so the server renders
  the same prompt the direct modes do.

## Not done

- The HIP and SYCL backends were not built; CUDA only.
- The upstream's Qwen engine (`strata.exe`) was not built or measured.
- The draft model (MTP) for the Gemma is not on this machine, so the draft path was off in every run.
- Mode C was measured three times in a row on one server start; the alternation was done for A against B.

## Files

- `2026-10-21-gemma-three-modes-aligned/A-upstream/`, `B-fork/`: the engine's stderr, the protocol, the
  summary with the sampler's output.
- `2026-10-21-gemma-three-modes-aligned/C-fork-via-server/`: the server's summary, its stdout, the engine's log.
- `2026-10-21-gemma-ab-interleaved/`: the A B B A repeats, one folder per run.
- The new pack: `PolyStrata-data/packs/gemma-26b/tokenizer` (the vocabulary from the GGUF, the template
  from `llama_cpp`).
- The engine binaries: the clone's `build-20/strata-poly.exe` and `build-poly-cuda12/strata-poly.exe`.
