# DeepSeek: rows, reuse of staging and the alleged 3x transfer amplification

2026-10-21. Read-only review of source, the model's GGUF header and the completed public synthetic
benchmarks. **No build, model load or GPU benchmark was run.** No engine/config/launcher was changed.
`analyze.py` reproduces the header arithmetic and scheduling; `analysis.json` preserves its output.
It does not read tensor payloads, private chats, user prompts or private logs. Existing benchmark
files are only read; none are modified. This report is not committed, per the user's preference.

## Main finding

There is a concrete reason to test **more rows in VRAM**, but not the proposed missing-tail fix:

* This DeepSeek has **43 layers, 256 experts per layer, 6 selected experts per token**. It is not
  the 288-expert GLM geometry used in the quoted arithmetic.
* The largest complete expert payload of any layer fits in **exactly 2848 MiB**, the configured
  staging size. Every layer's entire RAM-only subset therefore fits too. There are no alleged
  14 experts beyond the ahead set in this configuration.
* The 50000-token prompt is split into **32768 + 17231**, then a final single-token eval.
  The second read reloads each layer's RAM experts because the shared buffer has already visited
  all intervening layers. This is a normal consequence of splitting the read, not failure to
  reuse the layer inside one read.
* `--read-chunk 53248` would put all 49999 read tokens in one layer-major pass. Rows would grow
  from 2048 to **3328 MiB**, an increase of **1280 MiB**. Under a fixed device budget those bytes
  come out of permanent GPU experts. Estimated main expert H2D falls from **223.42 GiB to about
  112.96 GiB** per prompt. These are source-derived payload counts, **not a measured speedup**.

## 1. What rows actually stores

`Engine::alloc`, `src/poly/engine.cpp:2303-2311` and `:2444`: the capacity is rounded down to a
multiple of prompt_batch, and `v_res_` stores intermediate residual rows between layers:

```
bytes = read_rows * streams * embedding * sizeof(float)
      = read_rows * 4 * 4096 * 4
      = read_rows * 65536
```

The four streams are DeepSeek's **hyper-connections**, confirmed by
`deepseek4.hyper_connection.count=4` and `DeepSeek4::streams()` at `deepseek4.cpp:62`.
Calling them GLM hybrid attention is incorrect. The formula and the 2 GiB size are nevertheless right.

`Engine::read` iterates calls of at most 4096 tokens **inside each layer**. Enlarging the read
does not enlarge these attention/GEMM microbatches. Its benefit is keeping the same layer's
weights through more microbatches before moving to the next layer.

## 2. Exact model sizes: no missing tail in the current configuration

Read from GGUF metadata only:

* architecture `deepseek4`, block_count 43, expert_count 256, expert_used_count 6;
* gate/up per expert: 3604480 bytes each, Q3_K;
* down per expert: either 3604480 bytes (Q3_K) or 4456448 bytes (MXFP4);
* raw expert payload: **10.3125 or 11.125 MiB**;
* all 256 raw experts of a layer: **2640 or 2848 MiB**;
* all routed expert payloads in the main 43-layer model: **117.765625 GiB**.

The ahead capacity uses **raw gate+up+down bytes**, not the separately padded slots of the decode
cache. `plan_ahead` at `engine.cpp:1769-1777` limits to `stage_bytes/(2*gu_bytes+d_bytes)`.
For the largest type this is `2848/11.125 = 256`, exactly. The allocation also reserves padding
and kTail beyond stage_bytes. The decode cache reporting 255 padded slots is not the prefill capacity.

The quoted 275 RAM experts in one DeepSeek layer is impossible: there are only 256 experts total.
Its claimed 14-expert overflow, 261-expert ahead set and related arithmetic were derived from the
wrong geometry. The per-layer counts do not follow by dividing the model-wide GPU count evenly;
placement is global and frequency-based.

## 3. What the reuse code really does

`moe_rows`, `engine.cpp:2942-2954` and `:3118-3180`:

1. The first call of a layer uploads its RAM experts to staging; `early` schedules that upload.
2. Here all of them fit in round 0. No routed RAM experts remain outside L.ahead, so there is no
   second staging round to overwrite it.
3. At the end, `staged_` equals L.ahead, `staged_il_` equals this layer.
4. The next call of the same layer satisfies `mine`; the search finds `kept=0`, and `r==kept`
   performs no upload. This does not depend on the timing heuristic because the first equality
   in the `mine` condition already holds. The model's main placement is fixed during this prefill.
5. A different layer overwrites staging. Therefore a new full read must start the sequence again.

The generic code *can* churn when a layer's RAM subset exceeds staging and multiple rounds overwrite
the same base. That is a valid candidate for smaller staging or other models, but it is **not the
current DeepSeek/2848 MiB case**. There is no reason to implement that fix as an explanation of these runs.

The control starts with 567 permanent GPU experts. Its exact raw RAM payload can be reconstructed
without inspecting expert-frequency contents: the model has two payload/slot size classes; the 567
slots and the printed 6202.09 MiB allocation identify a unique mixture (see analysis.json).
This gives **119948050432 bytes = 111.7103271484375 GiB** in RAM. The two layer-major reads each
copy that set once, i.e. **223.420654296875 GiB** of main staging payload. Small token, routing,
embedding, final-eval and driver transfers are separate and are not included in that number.

## 4. Why 371 -> 126 GiB is not established

`serve/telemetry.py:107-110` calls NVML PCIe throughput. NVML documents a **20 ms interval**:
https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html

The runner samples about every 0.5 seconds. Its arithmetic mean therefore averages short windows
with gaps, not a continuously accumulated byte counter. In the reference first prefill the mean RX
is 4345.61, median 56.65 and maximum 44657.61 in the telemetry's units. Such bursty measurements
cannot establish exact uploaded bytes by multiplying the unweighted mean by full prompt time.
The counters are also device-wide and not restricted to expert payloads or even this process.

Further, the quoted one-pass minimum ignores the current **two reads**. The actual RAM subset is
111.71 GiB, not `42*275*10.905 MiB`. Neither 371 GiB as exact expert uploads nor its claimed 3x
amplification is proven by those samples. To establish extra API transfers, add byte counters at
the actual staging copies, split by layer/read and reuse/miss. That requires separate opt-in
instrumentation; it was not added in this review.

## 5. PP is not a token-prefix clock; 48% attention is not measured

`src/poly/main.cpp:444-449` emits progress as:

```
on_progress(i + m * done / of, n, elapsed, false)
```

Here done/of is completed **layers**, and every such layer has processed all microbatches in that
read. At PP=762 the first layer has handled up to 32768 real positions; it does not mean that only
a 762-token KV prefix has been processed. A fit of seconds against this synthetic progress count
cannot isolate the time of attention versus copies/MMQ. No GPU phase measurements justifying
41.3 seconds / 44.8 seconds appear in these results. The existing gpu_phase_probe only records n=1,
so it does not supply missing prefill timings either.

The reference control reaches 32768 at **47.102 s**, then completes the remaining 17231 read
tokens plus the final eval in **39.015 s**. In the second request: **46.920 + 39.414 s**, including
parking in total prompt time. Both attention context and amortization of a second weights pass
can affect this; those two aggregate intervals do not separate their shares.

DeepSeek also uses local-window and compressed/indexed attention (ratios 0/4/128, raw window 128,
index top_k 512), not full-context attention in every layer. Some cost grows with context, but
the quoted global linear model is not a measured decomposition.

## 6. Concrete next test: 53248, not 16384 or 50000

For a 50000-token prompt, the nonfinal read has 49999 tokens. All choices below still have
**13 microbatches** with the same sizes: twelve 4096 and one 847.

| Requested read-chunk | Actual read capacity | Rows, MiB | Reads before final token | Main staging H2D |
|---|---:|---:|---|---:|
| 16384 | 16384 | 1024 | 16384 + 16384 + 16384 + 847 | three full passes plus routing-dependent short tail |
| 32768 (current) | 32768 | 2048 | 32768 + 17231 | 223.42 GiB |
| 50000 | 49152 | 3072 | 49152 + 847 | still a second read; tail uses another execution branch |
| **53248** | **53248** | **3328** | **49999** | **about 112.96 GiB** |
| 65536 | 65536 | 4096 | 49999 | about 113.71 GiB |

H2D estimates for changed rows assume a fixed device budget and roughly one byte of main expert
VRAM displaced per extra row byte. Exact placement varies by slot rounding and live device budget.
For 53248 that is about 1.25 GiB more host expert payload, so about 112.96 GiB loaded once instead
of 111.71 GiB twice. This removes about **110.46 GiB of main weight payload transfers**, approximately
49.4%, but it does not halve attention/MMQ FLOPs or establish any particular wall-time speedup.

More VRAM rows come at the expense of **permanent GPU experts and host RAM**, so decode can get
slower and RAM pressure can rise. Total requested VRAM need not rise; it is a reallocation inside
the same budget. A permanent resident expert still performs its GEMMs, so treating cached experts
as removed compute is incorrect. A one-pass prefill and decode benefit are different tradeoffs.

The next meaningful run is a new 53248 configuration against the existing 32768 control on the
same valid server-encoded prompt/profile/exe, with timing, process allocations, host pressure,
decode, actual stage byte counters if available, and output/logit correctness checked. No predicted
tok/s, no default change, and no claim that decreasing chunk is universally forbidden follow from
this review. A later implementation could lend expert VRAM to rows during prefill and return it
for decode, but fixed no-disk ownership currently needs explicit safe handling for that.
