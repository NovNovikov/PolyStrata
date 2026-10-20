# Where ordinary chat decode spends time

Measured 2026-10-20 on this Windows PC: RTX 4090, CUDA 12.4, i5-13600KF. Windows reports
6 physical cores / 12 logical processors. Two 64-GiB DIMMs report a configured rate of 5600 MT/s.
Model: `DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`. No private chat, prompt, log or expert profile was read.

This continues [the public chat diagnostic](../2026-10-20-poly-decode-diagnosis/README.md), and checks
[Minimax's hypotheses](../2026-10-20-poly-session-cache/SPEED-HYPOTHESES.md). The task was to locate costs,
not to substitute another reproduction of the observed tokens/s.

## What was established

The large CPU-expert timer corresponds to substantial real computation, independently of the server,
CUDA and the graph layout. Four RAM experts in each of the model's 43 layers take about **33 ms**
in the standalone CPU path. About **20.6 ms is gate/up**, **12.2 ms is down**, and **0.43 ms is
preparation and activation quantization**. This is a measured cost of the current implementation,
not a claim that the hardware could never compute it faster.

The separate engine probe locates the GPU work before the router, rather than calling all its
wait time "attention": a single-position pass spends a median **18.29 ms** in that part of the GPU graph.
The largest interval is attention at **5.51 ms**; output projections take **3.75 ms**, the compressor/indexer
interval **3.48 ms**, and Q/K/V projections **2.71 ms**. These are diagnostic timings with event nodes present.

In the instrumented answer, the host CPU-expert timer was **46.78 ms per output interval**, and waiting
for the router **16.63 ms**, against **71.75 ms wall time**. The CPU timer includes the concurrent GPU
submission callback and worker waits. GPU work overlaps it, so the GPU and host columns are **not
independent pieces to add**. Nevertheless, the scalar-token path has a dependency at every layer:
GPU head -> host routing/CPU experts -> add their result -> next GPU head. The shared/cached GPU experts
can overlap the CPU experts; the next layer's dependent head cannot.

This explains why a fast repetitive speculative benchmark cannot establish ordinary chat throughput.
The direct CPU measurement is already longer than a 35-token/s output interval (28.6 ms) for this
four-RAM-expert workload, before the dependent GPU heads. That is not a new assertion about MTP.

There is also confirmed rejected verification work: **127 output intervals evaluated 142 position
equivalents**. `24 offered - 9 accepted = 15` extra positions; independently,
`36636 expert lookups / (43 layers * 6 experts) = 142`. This is **11.8% additional positions**, not a
measured 11.8% latency penalty: batching and CPU/GPU overlap can recover some of that work.

## CPU isolation, including two rejected explanations

`poly_cpu_decode_bench` loads only four experts from every layer, closes the GGUF, then runs the
production `cpu_experts()` routine over those weights. Total selected payload: **1,975,779,328 bytes
(1.840 GiB)**. Input: 4096 deterministic binary fractions. One sweep visits all 43 layers; another
mode repeatedly visits one layer. Gate/up is Q3_K; down is MXFP4 in 34 layers and Q3_K in nine.

The final ISA pair uses identical inputs, compiler options for the harness/dispatcher, and a
test-only replacement of ggml's x86 dot-product object. Each binary tests all five worker counts in
three shuffled rounds, with 12 sweeps per sample. Binaries run sequentially; no other benchmark runs
at the same time. Other user applications were not stopped or controlled.

| CPU threads | stock median, ms/sweep | AVX-VNNI median, ms/sweep |
|---:|---:|---:|
| 1 | 140.97 | 142.22 |
| 3 | 53.51 | 61.07 |
| 6 | 35.30 | 39.58 |
| 8 | 32.65 | 34.31 |
| 12 | 33.15 | 33.39 |

The large 12-thread spread is retained: stock **32.77/33.15/39.97 ms**, VNNI **32.67/33.39/37.99 ms**.
These sequential microbenchmarks do not justify changing the production thread default. Six threads
do not fix the problem; eight and twelve are close. CPU dispatch of 256 empty tasks is roughly
0.01 ms, although this does not bound every worker barrier under load.

AVX2/FMA/F16C are already compiled into production ggml-cpu; there is no scalar fallback here.
`GGML_AVX_VNNI` is OFF in the production cache. CPUID confirms that this CPU supports AVX-VNNI, but
enabling its ggml dot-product branches in the isolated binary **does not give a throughput gain**.
It is not a demonstrated fix. All thread-count comparisons passed bitwise equality. The final
stock/VNNI input and all 43 output vectors also compare **byte for byte**, SHA256
`13521561387a161e2a193b32b4ed18935d158702d281d72e9d4123dcba285aa6`.

The logical weight payload divided by time is about 55.5 GiB/s for the stock 12-thread sweep.
This is **not** a hardware DRAM-bandwidth counter and does not separate memory stalls from dot-product
execution. A claim that the memory bus is saturated would require further measurement.

Raw data: [isa-pair](isa-pair/), [CPU phase CSV](cpu-stages.csv), [first exploration](cpu.csv).
The first two explorations used Gaussian inputs. The final ISA pair uses exact binary fractions to
avoid input changes from compiler-dependent random-distribution math; compare variants within that pair.

## GPU phase probe on the model

`poly_gpu_phase_probe` is a separate, excluded-from-default-build executable. It adds seven timing
events to a DeepSeek layer head. Events use `cudaEventRecordExternal`, so they actually record during
graph replay. The [self-test](phase-self-test.txt) verifies positive elapsed time from captured events.
Samples are collected after `run()`'s existing stream synchronization; no new synchronization is
inserted between model phases. The production engine contains none of these phase event nodes.

The synthetic varied chat has 22,445 input tokens and requests 128 output tokens. The public expert
seed, actual chat template, batch 4096, read chunk 32768, stage 2848 MiB, compact scratch and six-lane
graphs are used. Disk parking is disabled only in this diagnostic. For diagnostic allocations the
test's load margin is **512 MiB**, leaving **543** permanent experts rather than the earlier run's 567.
The user JSON is unchanged. This is not a speed A/B against the previous production run.

The probe completed: prefill **32.628 s**, decode **9.112 s / 13.94 tok/s**, hit **29.02%**, uploads **404**.
All **255** staging payloads were checked against original RAM, including zero tails. Of the scalar
position samples, 101 remain after excluding the prompt's last position; all contain 43 measured layers.
Multi-position speculative/halved passes are not included in the phase median.

| GPU interval, summed over 43 layers | median ms / scalar position |
|---|---:|
| hyper-connection input + norm | 1.121 |
| Q/K/V projection, norm, RoPE and raw-cache store | 2.707 |
| compressors + indexer projections/scores/selection | 3.478 |
| attention weighted sum and softmax | 5.507 |
| output RoPE + grouped output projections | 3.749 |
| hyper-connections, FFN norm, router and host copy | 1.413 |
| full measured head, median of per-position sums | **18.293** |

Medians of individual intervals need not add to the median of their sum. These intervals exclude
shared/routed GPU experts, the final output head, layer tails outside these markers, CPU routing and
cache maintenance. Captured event nodes have overhead; the table locates costs, not uninstrumented
production speed or a precise exclusive hardware occupancy breakdown.

Metadata gives a concrete fixed cost: `attn_q_b`, `attn_output_a`, `attn_output_b` are all Q8_0 and
total **4.283 GiB of weights** across 43 layers. Their matrices are used even if every routed expert
hits the cache. See [dense metadata](dense-metadata.json); no weights were read for that metadata scan.

## Context scaling of the attention kernels alone

`poly_dsv4_decode_bench` uses the actual model geometry: 64 heads, head width 512, raw window 128,
512 selected compressed keys. GPU events time graphs of 64 repeated kernel calls; seven trials,
median, synthetic values. Buffers are warm; dense projections and CPU work are absent.

| context | attention + index scoring + selection, weighted over actual layers, ms |
|---:|---:|
| 1536 | 3.062 |
| 22446 | 4.759 |
| 90000 | 8.316 |
| 131072 | 10.689 |

DeepSeek does not attend to the whole uncompressed context in every layer: it uses the raw window,
selected compressed rows in ratio-4 layers, and all visible ratio-128 compressed rows in those layers.
Context therefore has a measurable cost; it does not by itself explain the reported shortest-context
slowdown. These warm kernel timings are not a context-length speed comparison of complete chat requests.

Correction retained transparently: `gpu-kernels.csv` initially labels the layer multiplicities as
5/19/19. Actual metadata for the **43 active layers** gives **2 ratio-0 / 21 ratio-4 / 20 ratio-128**.
Per-call measured times remain valid. The table above, `analysis.json`, and
`gpu-kernels-corrected-counts.json` recompute the weights from `attention-geometry.json`; the raw file
is preserved. The benchmark source now prints the correct multiplicities.

## Check of Minimax's H1-H8

| hypothesis | what the evidence supports |
|---|---|
| H1: every cache miss is a PCIe expert transfer | **Wrong mechanism for decode.** `Engine::moe()` sends absent slots to `cpu_experts()`. Cache uploads are separately counted. The CPU computation is a major measured cost. |
| H2: a full staging cache makes every remaining expert pass through host staging | **Wrong.** Full occupancy means cached GPU copies occupy all slots. A miss may be computed on the CPU without any cache upload. `follow_resident()` chooses which RAM experts to upload. |
| H3: six-lane graphs cause a slowdown | **Still unmeasured as a controlled A/B.** This phase probe does not settle it. Existing 41 -> 35 figures are not a same-workload regression measurement. |
| H4: NonLocal proves GPU weights spilled | **Not established.** In this probe NonLocal falls from 3316 MiB at READY to 480 MiB after decode while the large registered host source window is released/replaced. Calling the whole counter GPU-weight spill is unjustified. |
| H5: HTTP streaming costs 5-8 tok/s | **Not demonstrated by the cited comparisons.** The earlier public direct chat diagnostic already gives 14-17 tok/s without HTTP. This probe is also engine-direct. A smaller server cost remains possible. |
| H6: longer context makes decode slower | **A real component, not the sole explanation.** The isolated attention pipeline grows from 3.06 to 8.32 ms between 1.5k and 90k. It is compressed/sliding attention, not full uncompressed attention per layer. |
| H7: 202-212 s of prefill is the reported low generation speed | **Separate metrics.** This run's actual DONE prompt time is 32.628 s (~688 tok/s). The other 202-212 s observation cannot be treated as pure prefill without its timing boundary. |
| H8: no-disk decode rereads model pages from disk | **Not the current model path.** Loading reports locked resident RAM and closed model files; DONE reports zero file blobs/MB. Session/checkpoint memory pressure is a separate question. |

Also keep units straight: the older `resident_weights=115411 MiB` is **112.71 GiB**, not 115.4 GiB.
Subtracting engine RSS from installed RAM is not a measurement of currently available system RAM.

## Failed profiler attempts and limits

The first Nsight-wrapped native process ended before READY without usable target protocol;
`gpu-trace/` is preserved. The child-driver attempt reached model load but the load reserve check
refused it: **250.81 MiB free < 256 MiB requested**. It never generated a token and supplies no speed
measurement. `gpu-trace-child/` is preserved. Installed Nsight Systems 2023.4.4's independent preflight
recorded CUDA API calls but no GPU kernel activities, so GPU kernel conclusions are not taken from it.

No user applications were stopped; no server was killed, restarted or contacted. No private expert
profile or conversation cache was used. Nothing in the JSON or production cache policy was changed.
AVX-VNNI and a lower thread count were tested and not adopted. No unmeasured optimization is reported
as an improvement. The outstanding six-lane A/B remains outstanding; these results identify current
costs and do not prove a historical regression or a hardware limit.

## Reproduction and validation

Build the excluded targets `poly_cpu_decode_bench`, `poly_cpu_decode_vnni_bench` (MSVC with VNNI
intrinsics), `poly_dsv4_decode_bench`, `poly_gpu_phase_probe`. Run the phase probe's `--probe-self-test`
before loading a model. The full commands and executable hashes for the CPU pair are in
`isa-pair/provenance.json`; the engine probe's command is in `phase-probe/engine/results.json`.

The drivers are `tools/bench_poly_cpu_isa.py`, `tools/bench_poly_decode_trace.py` and
`tools/analyze_poly_cpu_bottleneck.py`. Result directories and reference files are exclusive-create.
The phase driver requests graceful QUIT and waits; it does not kill a slow shutdown. Analysis refuses
an existing `analysis.json`. Use new output directories for new experiments.

CUDA is the built/tested backend here. Phase instrumentation is compiled only into the separate
CUDA probe. CPU stage instrumentation is compiled only into the CPU microbenchmarks. The installed
engine's optional `STRATA_PROFILE_DECODE=1` merely brackets GEN decode with profiler start/stop APIs;
with it unset no profiler calls are made. There is no production arithmetic/cache-policy change.
