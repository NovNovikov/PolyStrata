# Why chat decoding is slower than the repetitive synthetic benchmark

2026-10-20, Windows, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM. Current
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf, six-lane compact graphs, staging
2848 MiB, batch 4096, context capacity 131072, read chunk 32768, 12 CPU workers.

The user's screenshot shows approximately 691 tok/s prefill and 15.7-17.2 tok/s
decode, with 20.7-30.9% expert hits. Private prompts, logs, sessions and expert
profile were not read. The existing 8080 server had no model loaded at inspection;
its aggregate counters did not correspond to the screenshot's request. It was
not stopped, restarted or used to run tests. No attribution to browser activity
or to another application's memory is made.

## Main findings

1. **The model GGUF contains no MTP block.** Reading only GGUF metadata/tensor
   descriptors found 43 blocks, 1328 tensors, no nextn metadata and no nextn/MTP
   tensors or `blk.43.*` tensors. The config has no separate `--draft-model`.
   `Engine::load_shape()` therefore leaves `draft_` false. The old fast
   multi-token path was suffix/prompt lookup, not a trained MTP predictor.
2. **Low speed reproduces with normal chat formatting, outside HTTP.** Two
   public inputs of about 22446 tokens, including a repetitive one, produce
   only 14.6-15.8 tok/s cold when the requested output is analysis rather than
   continuation of the old raw repeating token sequence. The diverse case
   remains 15.6-15.7 tok/s in warm repeated requests.
3. **Most expert work goes to the CPU.** Diverse input cold answer: 29.46%
   GPU-expert hits, 70.54% CPU-expert lookups. Over a 35.09-second decode,
   existing CPU-expert scopes accumulated 21.81 seconds and router-event waits
   10.47 seconds. These scopes can overlap GPU work and are not an additive
   exclusive breakdown. Process CPU usage median is about 960% (9.6 logical
   CPUs); device utilization median about 45%, SM clock 2430 MHz.
4. **Lookup saves very few model steps on this output.** The diverse cold
   answer has 11 accepted drafts in 512 output tokens. Of 500 rounds, 459
   choose width 1, 39 width 2 and two width 3. Old fast raw-text repeats accepted
   42-54 drafts in 64 outputs. Their 35-41 tok/s is not a measured general chat
   rate. The new result does not establish that the adaptive policy is optimal;
   it establishes the actual mostly-single-token execution path.
5. **The adaptive expert cache is functioning.** All 255 staging slots are
   filled and payload-verified; cold diverse answer uploads 426 experts, warm
   requests still upload 14/20. Executable graphs stay at 281. Low expert hit
   rate is not by itself evidence of a disabled cache. The reported hit counter
   combines permanent GPU experts and the adaptive cache, not adaptive hits alone.

## Fresh measurements

Public repetitive lines and varied repository documents/code were rendered
through the model's installed chat template with thinking enabled. The user
instruction asks for technical analysis in both cases. This deliberately differs
from the old raw `amber birch...` sequence-completion workload. Public expert seed
`../2026-10-19-poly-graph-packing/seed-expert-profile.txt` is copied to a new work
directory; the user's live profile is neither read nor modified. Placement is
567 permanent GPU experts plus 255 adaptive slots, fixed across these cases.

| Case | Actual output tokens | Read prompt tokens | tok/s | GPU-expert hit % | Accepted/offered drafts | CPU-expert scope s | Router wait s |
|---|---:|---:|---:|---:|---:|---:|---:|
| repetitive cold | 303 (EOS) | 22446 | 15.82 | 30.62 | 5/17 | 10.84 | 6.50 |
| repetitive continuation | 128 | 1 | 19.83 | 27.83 | 81/91 | 4.87 | 0.94 |
| repetitive warm repeat 1 | 128 | 1 | 17.35 | 33.99 | 15/18 | 4.48 | 2.23 |
| repetitive warm repeat 2 | 128 | 1 | 16.67 | 33.84 | 13/21 | 4.60 | 2.36 |
| diverse cold | 512 | 22445 | 14.56 | 29.46 | 11/43 | 21.81 | 10.47 |
| diverse continuation | 128 | 1 | 14.32 | 30.42 | 6/6 | 5.68 | 2.54 |
| diverse warm repeat 1 | 128 | 1 | 15.62 | 33.21 | 3/6 | 4.88 | 2.63 |
| diverse warm repeat 2 | 128 | 1 | 15.71 | 34.16 | 5/11 | 4.80 | 2.64 |

Timing uses the engine's sealed `decode_ms`, not request duration or terminal
printing time. Full-input prefill is about 648/657 tok/s. Diverse cold first
128 token arrivals give 14.91 tok/s; repetitive first 128 give 15.28 tok/s.
The screenshot's range is reproduced without a long disk save or Python HTTP
streaming in the measured path. Exact values need not match the private request.

Warm requests read one prompt token and preserve the staging cache. Their failure
to approach 35-41 tok/s demonstrates that cold cache alone is not a sufficient
explanation for this workload. Full batched prefill does clear the staging cache
in `drop_resident_stage_cache()`, but it is not the whole story. Diverse answer
after 512 generated tokens and its continuation are still slow.

Greedy warm outputs diverge from cold prefixes at tokens 32-56 and from each
other at tokens 32/38. Adaptive placement can change CPU/GPU floating-point sum
order; this diagnostic does not identify the exact reason for those divergences
and is **not** a numerical-parity test. Consequently a warm/cold speed ratio is
not an isolated causal estimate of cache warmup. All IDs and comparisons are
preserved in the raw results and analysis. Prior fixed-placement parity is a
separate test and does not resolve this question.

## Instrumentation and limits

`STRATA_TRACE_DECODE_COSTS=1` prints existing per-answer counter/timer deltas:
wall decode time, CPU-expert scope time, router-event wait, uploads and draft
counts. The snapshot is taken at the first emitted token so prompt work is
excluded. It is off by default; no inference arithmetic, cache policy or draft
policy was changed. CPU-expert scopes include `cpu_experts()` and its optional
GPU submission callback/worker waits, not CPU instruction time alone. Router
wait includes the preceding queued GPU work, not just attention. These timings
must not be summed as exclusive CPU/GPU ownership or converted to memory bandwidth.

The first measured diagnostic called Stats.tokens `evaluated_tokens`. That
counter tracks committed/accounted positions, not rejected verification work.
After measurement the label was corrected to `accounted_tokens` and the help
text added. Raw results retain the original label; no data was overwritten.
Measured versus installed binary hashes and this string-only change are in
verification.json. The installed executable remains at the path used by both
JSON files. Its prior pre-diagnostic binary is preserved as
`build-poly-cuda12/strata-poly-before-decode-diagnosis-20261020.exe`.

The driver samples process CPU/RAM and NVML utilization/clock/power every 0.5 s,
records numeric policy traces, and verifies adaptive cache payloads. Checkpoint
count stays six; disk parking is disabled only in this isolated test by using an
empty session directory. The production JSON is unchanged, including staging
2848 MiB and no-branch setting. GPU verification and trace overheads are present,
so this is a diagnostic, not a claimed universal production-throughput benchmark.

The eight cases are sequential in one process. Policy history carries across
requests; varied cases are not cold starts of the policy. No claim that lookup
selection is optimal, no throughput claim for a working MTP model, no per-layer
DRAM bandwidth measurement and no performance intervention are established.
The installed GGUF is the user's custom variant; compatible MTP weights and a
validated execution path would be required to evaluate trained speculation.
Simply raising max_tokens or setting a flag cannot create missing weights.

CUDA builds completed, all nine C++ dependency checks pass. No HIP/SYCL build
or new full-logit parity was done for this diagnostics-only change. No test
model process remains running. Existing results were not deleted or overwritten.

## Evidence

- [analysis.json](analysis.json): per-case host timers, width histograms, hardware
  medians, cache verification and token-prefix comparisons.
- [model-draft-metadata.json](model-draft-metadata.json): metadata/tensor scan;
  no weight payload was read by the scan.
- [measurements/provenance.json](measurements/provenance.json): input-source,
  config, binary, expert-seed and diagnostic-source hashes.
- [measurements/engine/results.json](measurements/engine/results.json): engine
  timings, token arrivals/IDs, process memory and hardware samples.
- [measurements/engine/engine.stderr](measurements/engine/engine.stderr): numeric
  policy traces, host timers, upload counts and payload checks.
- [verification.json](verification.json): final binary/config/build checks.

Reproduction uses a new work/output directory (the driver refuses existing ones):

```
.venv\Scripts\python.exe tools\bench_poly_decode_diagnosis.py --config strata-dsv4-q3_k_m.json --profile bench/results/2026-10-19-poly-graph-packing/seed-expert-profile.txt --work <NEW_WORK> --output <NEW_OUTPUT>
```
