# Decode expert cache in the existing staging allocation

Measured on 2026-10-19: Windows, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM,
CUDA 12.4 / MSVC 19.41, architecture 89. DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf,
the same local model and tokenizer used by the earlier VRAM-first measurements.
Only synthetic reference text was used. No private prompts or server logs were read.

## What changed

The no-disk loader still places GPU-only permanent experts in its original arena and keeps every other
expert at a fixed locked RAM address. During small-token evals, the existing 2848 MiB GPU staging
allocation supplies 255 adaptive cache slots, each 11668800 bytes including zeroed kernel padding.
There is no additional GPU weight allocation, host weight mirror or GPU-to-RAM eviction. Permanent
GPU experts cannot be victims. Recent-use decay, promotion threshold and prior tie-breaking follow
the regular cache's policy. Up to 64 direct uploads use registered original RAM sources.

Before batched prefill reuses staging, compute and upload streams finish; only the staging cache's
slots are invalidated. RAM originals and permanent GPU slots survive. Diagnostic allocation totals
count staging once. Per-answer hit statistics exclude the prompt's CPU work.

## Long natural-generation runs

Parameters: context 131072, batch 4096, read chunk 32768, staging 2848 MiB, runtime reserve 256 MiB,
loading allowance 256 MiB, 12 CPU threads, rows in VRAM, prompt/conversation parking budgets zero,
model draft disabled. Default ngram lookup remains enabled (`--lookup 3`). Each input had exactly
50000 tokens and each answer reached 512 naturally generated tokens. The second adaptive request
changed its first input token, forcing a fresh prefill; neither request reused prompt tokens.

| Run | Permanent GPU experts | Staging cache | Prefill tok/s | Decode tok/s | GPU expert hit rate |
| --- | ---: | --- | ---: | ---: | ---: |
| Frozen control | 285 | empty | 595.12 | 16.44 | 16.49% |
| Adaptive, request 1 | 298 | 255 slots | 603.53 | 26.67 | 40.80% |
| Adaptive, request 2 after new prefill | 298 | 255 slots | 608.09 | 26.99 | 42.19% |

The first adaptive answer completed 330 uploads: the pool filled and then replaced entries.
The next prefill explicitly discarded 255 RAM-backed copies. Its answer completed 261 uploads,
refilling and replacing entries again. After EACH answer, all 255 published GPU payloads were
compared byte-for-byte with their original RAM sources; all zero kernel tails passed as well.
This opt-in verification runs after DONE, outside decode timing, using a 64 KiB host buffer.
All 512 naturally generated token IDs of the first adaptive answer matched the frozen control exactly.
The second adaptive answer also produced that same sequence, despite the changed first input token.
This is a measured greedy-output comparison on this synthetic input, not a general numerical-error bound.

First adaptive answer's 128-token intervals: 39.51%, 40.69%, 41.96% hits;
24.31, 27.04, 28.32 tok/s. Corresponding control intervals: 17.31%, 16.46%, 15.43%;
16.75, 16.33, 15.52 tok/s. The adaptive run had STRATA_FREEZE_EXPERTS removed from its environment.

These are practical operating measurements, not an isolated attribution of every speed difference
to caching: available VRAM changed the permanent placement (285 versus 298 experts), freezing also
holds other adaptive execution choices still, and natural continuations need not be identical when
the GPU/CPU sum order changes. There was one load per configuration, not a statistical repeat series.

## Allocation and file-I/O evidence

Adaptive requested GPU allocation was 21937.76 MiB both after load and after the answers:
dense 7758.37, permanent experts 3259.14, KV 860, staging 2848, read rows 2048, other 5164.24.
The staging-cache line separately reported 2837.70 MiB used within that staging allocation.
The 141.41 MiB difference from the control's total is its smaller permanent arena, not a new cache arena.
Driver/library/graph allocations are outside these requested-byte totals.

Adaptive process reads after load: 525921 bytes, exactly the two stdin commands; extra reads **0**.
The model file was closed after READY. The explicit VRAM-resize request was refused and QUIT exited cleanly.
Peak RSS over both adaptive requests was 119525.23 MiB; resident host weights/buffers remained
118745.32 MiB. Registered source pages are part of those resident weights, not another RAM copy.
System-wide disk activity in the raw JSON includes other processes and is not attributed to the model.

## Test-run corrections and binary identity

The first control attempt refused placement before READY, short of RAM by approximately 36.56 MiB.
The runner retained a tokenizer and source text after encoding. Releasing those unused objects before
launching the engine allowed the same configuration to load. No engine memory limit or reserve was bypassed.

The completed control initially requested `--follow`, but the old main assigned that replay pointer only
in once mode. Serve mode ignored it. The runner's equality assertion caught the discrepancy after a complete
512-token DONE. This control is therefore explicitly classified as NATURAL generation; its result JSON was
recovered from the saved complete protocol. The adaptive long run did not request replay. The explicit replay
pointer is now also assigned before entering serve mode, and a separate final-build test exercises it.

Frozen control SHA256: `0f4a53c1379d0125fe5e680a4ddd8af761f7081382b80439049e9ad874f7597b`.
Long adaptive SHA256: `580ef2fb571397fa533e4fafb893c89dc6177dc2e0ca2ff79584c53dd8198bf4`.
The final build adds reclamation at read/run/step entry before `all_cached()` chooses a path, protecting
small models that might fit entirely in the staging cache. The long run already reclaimed inside batched MoE.
Final executable SHA256: `6d222e46fbd15b8168e5bf2b2cc4c6475c62400b9726690c3d444ce8e6c47741`.
Build output explicitly confirmed engine.cpp.obj and no_disk.cpp.obj recompilation; the last incremental
build recompiled engine.cpp.obj again and linked successfully. Binaries were built from the worktree atop
2a9a5d37; the source changes are committed separately from this measurement record.

Existing CPU checks: poly_no_disk_test, glm_conversations_test, glm_conversations_disk_test, 3/3 passed.
Only Windows/CUDA and this DeepSeek model were run. HIP/Linux/SYCL, other models, cancellation and concurrent
batch-slot workloads were not measured in this series. The unchanged mmap path was inspected, not rebenchmarked.

## Final-build phase and replay verification

The final executable above was loaded separately with context 4096, batch 256, read chunk 2048,
2048-token synthetic inputs, 256 forced output tokens, two requests, adaptive cache and byte verification.
It loaded in 199.418 s, with 1041 permanent GPU experts and the same 255 staging cache slots. Both
outputs matched the requested replay IDs exactly (the runner asserts the entire sequence), and both
requests reused zero prompt tokens. Completed cache uploads were 367 and 258, with 51.97% and 55.24%
GPU hit rates. Both payload/tail checks passed; the intervening prefill discarded all 255 borrowed copies.
Post-load process reads were 21239 bytes, exactly stdin; extra reads zero. VRAM resizing was refused,
QUIT returned zero, and no test engine remained. These shorter-context timings are not compared to the
50k natural-generation numbers because the context, batch, permanent placement and replay differ.

This small-batch configuration also exposed an existing no-disk headroom limitation: requested device
allocations stayed at 22491.77 MiB, but whole-device free fell from 453.86 MiB after load to 69.30 MiB
after the second request, below the named 256 MiB reserve. Whole-device figures include driver/libraries
and other processes; this test does not identify their separate contributions or establish WDDM spill.
No-disk's immutable permanent arena still cannot shrink to enforce the reserve during runtime. This
change does not solve that broader headroom policy. In the 128k/4096 long configuration, post-request
free remained above the reserve (399.78 MiB after its first adaptive answer). Keep the two cases distinct
in the separate buffer/driver-overhead audit rather than claiming identical safety margins for every batch.

## Reproduce the long adaptive run

```text
python tools/bench_poly_no_disk.py --exe build-poly-cuda12/strata-poly.exe --model MODEL.gguf --tokenizer TOKENIZER_DIR --profile PROFILE.txt --output RESULTS_DIR --context 131072 --tokens 50000 --batch 4096 --read-chunk 32768 --stage-mib 2848 --reserve-mib 256 --load-margin-mib 256 --threads 12 --rows-vram --skip-default --no-disk-runs 1 --generate 512 --requests 2 --adaptive-experts --verify-stage-cache
```

Raw commands, per-request timings and token timestamps are in adaptive-results.json; protocol and memory
reports are saved alongside. The control protocol and recovered results retain the failed replay assertion's
context so the original natural output is not misrepresented as a forced sequence.
