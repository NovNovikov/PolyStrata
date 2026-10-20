# Why the decode is slow on this machine: the facts, and the hypotheses to test

Written 2026-10-20, for the maintainer. Everything below is measured on this box (RTX 4090, 24 GiB,
128 GiB RAM, i5-13600K), the DeepSeek-V4-Flash-0731-Q3_K_M-custom model (the GGUF is 126.34 GiB), the
engine `build-poly-cuda12/strata-poly.exe` (`f8dd3e5b`), the config `strata-dsv4-q3_k_m.json`
(`--no-disk --rows-vram --compact-graphs --stage-mib 2848 --read-chunk 32768 --prompt-batch 4096
--max-context 131072 --expert-swaps 64 --conversation-cache-mib 0 --conversation-cache-no-branch`).

## The measurements

**Decode speed (tokens per second):**

| run | context | path | tok/s |
|---|---:|---|---:|
| the graph bench (32 repeats, median, fresh engine) | 10 511 | engine direct | 33.60 / 39.12 / 34.97 (control), 38.52 / 35.46 (row-layout compact), 34.92 / 35.36 (six-lane compact), 41.01 (one older control) |
| the session scale bench (64 tokens) | 90 000 | engine direct, old exe | 22.7 / 25.7 / 23.6 |
| the session scale bench (64 tokens) | 90 000 | engine direct, new exe | 23.9 / 24.5 / 25.5 |
| a synthetic prompt through the running server | 1 500 | through the server | 15.6 and 19.8 (the same prompt twice) |
| a synthetic prompt through the running server | 22 446 | through the server | 17.2 |
| the real chat | 1 469 | through the server | 16.0-18.1 |

**Expert cache hit rate:** the real chat 41.7% / 44.1% / 32.3%; the synthetic prompts 49.5-50.1%;
the long prompt 20.7-38.7%.

**VRAM, by the driver's per-process counters and by the engine's own breakdown:**

```
process: used 22.3 GiB   asked 22.8-23.0 GiB   spilled to host RAM 0.5-0.7 GiB
engine:  VRAM allocations total = 22306.09 MiB
         dense 7758.37 | experts 6202.09 | KV 860.00 | staging 2848.00 | rows 2048.00 | other 2589.63
         permanent experts 567, staging_cache 255/255 slots (2837.70 MiB), uploads 292-322 per request
device:  used 24271-24285 MiB, free 278-434 MiB (includes other processes)
```

A peak was also seen: asked 25.6 GiB with 3.1-3.2 GiB spilled, on a server that was serving a chat
while a long second context was read; a clean server with the same prompt asks 23.0 GiB and spills
0.7 GiB. The engine's own breakdown was identical in both cases.

**Host RAM:** the engine's process 113.7 of 127.8 GiB (the weights are the model: resident_weights
115.4 GiB out of the 126.34 GiB file, read into RAM by --no-disk). So ~14 GiB is left for everything
else. A second engine instance does not fit.

**Prompt prefill:** 22 446 tokens read from scratch = 202-212 s (~110 tok/s); the same tokens read
again from the conversation cache = 1.6 s.

## The hypotheses

**H1. The expert cache's miss rate is the main cost: every miss is a PCIe transfer of an expert
from the 113.7 GiB in RAM.** For: the hit rate moves with the speed across every run we have (the
chat at 41.7% -> 16-18 tok/s; the synthetic prompts at ~50% -> 20-24; the graph bench's fresh engine
at 35-39 - its hit rate is not in the results, that is the first thing to add). Against: nothing yet.
Test: log the hit rate and the tok/s in the same run, and the GPU/CPU expert counters (the engine
already prints `expert-cache answer: generated=N GPU=x CPU=y hit=z%`).

**H2. The staging cache is saturated (255/255 slots, 2837.70 MiB) in every run, so every expert
that is not permanent passes through the host staging on its way to the device.** With 567 permanent
experts out of 11008 routed, the rest are staged. For: `staging_cache=255/255` in every run of the
logs, and 292-322 uploads per request. Against: nothing yet.
Test: the tok/s against the staging's occupancy; whether raising `--stage-mib` (or the number of
permanent experts) moves the speed; what "other = 2589.63 MiB" is.

**H3. The six-lane compact graphs cost decode time.** For: the one in-session pair in the graph work,
41.01 -> 35.36 tok/s (-13.8%). Against: the spread between the runs of the same bench was +-16%, so
that single pair proves nothing; the six-lane compact runs measured 34.92/35.36 against the row-layout
compact's 38.52/35.46, which is inside the noise.
Test: the same exe, the same prompt, with and without `--compact-graphs`, several repeats, the engine
direct, on a machine where nothing else holds the GPU. (This A/B was started and abandoned because a
second engine instance was started while the server was running; that is the reason it is not done.)

**H4. The VRAM spill puts part of the expert work on the PCIe.** For: the driver's counters show
0.5-0.7 GiB of the process's VRAM living in the host RAM, and 3.1-3.2 GiB in the peak case; the
engine's 22.3 GiB of tensors plus the CUDA context and the allocator's pool push the request past
the card's 24.56 GiB. Against: the spill's size depends on the other processes' VRAM, so it is not a
property of the engine alone.
Test: the tok/s with the spill reported per request (the narrator has it) against the engine's own
decode time (the DONE line); repeat with the other GPU apps closed.

**H5. The server's streaming and parsing cost 5-8 tok/s.** For: the engine direct measures 22.7-25.7
and the same engine through the server measures 15.6-19.8. Against: the comparison is confounded
(the direct runs were on a 90k context, the server runs on 1.5k), and the same prompt twice through
the server gave 15.6 and 19.8, so the spread is +-20% on its own.
Test: take the rendered prompt's token ids, send them to the engine directly and through the server,
and compare the engine's own decode time (from DONE) with the server's measured rate. The server's
per-request narration already prints both.

**H6. The decode's cost grows with the context (the attention over the whole context per token).**
For: 10 511 tokens -> 35-39 tok/s against 90 000 -> 22-25. Against: the real chat's context is 1 469
tokens and it is the slowest (16-18), so the context length alone does not explain it.
Test: the same prompt content with the context length varied 1.5k / 22k / 90k, through the same path.

**H7. The prefill of a changed context is what the chat actually pays (202-212 s for 22 446 tokens),
not the decode.** For: the prefill number above; in a chat where the tail changes every turn, the
re-prefill of the tail is what the user waits for. Against: the tail is short in this setup, and the
disk checkpoints exist to keep the prefix out of that cost - but they only help while the prefix is
byte-identical.
Test: the time to the first token in the real chat, with and without the conversation cache warm.

**H8. The host RAM at 113.7 of 127.8 GiB squeezes the page cache, so the model's file pages and the
expert weights are re-read.** For: the free RAM is ~14 GiB and the diskcache run reported
`available 0.3 GB`. Against: in --no-disk the weights are copied into RAM, not mapped, so the decode
should not page them; the file's reads happen at load.
Test: the disk throughput during a decode (the narrator prints it) and the page-fault counters.

## What is not the cause

- The disk checkpoints: the decode path does not touch them (the same 64-token decode measures
  22.7-25.7 tok/s on the old exe and 23.9-25.5 on the new one, on the same scenario); what they
  changed is the park (15.3 s -> 1.7 s, 29.5 ms with a delta) and the restore (32 s -> 3.4 s), plus
  the cache's host memory (3.99 GiB -> 17.5 MB).
- The context length alone: the shortest context measured is the slowest (see H6).
- The free VRAM of the card: it is contaminated by the other processes; the process's own consumption
  is 22.3 GiB in every run and is what the engine accounts for (22306.09 MiB in six components).

## The first thing to measure

One A/B, on a machine where nothing else holds the GPU: the same exe, the same prompt, with and
without `--compact-graphs`, 32 repeats, the median of the decode times, with the expert cache's hit
rate printed per run. It settles H3 and gives H1 its missing number.
