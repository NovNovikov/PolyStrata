# VRAM-first loading headroom, 2026-10-19

Windows, RTX 4090 (24563.50 MiB), i5-13600KF, 128 GiB RAM, 12 CPU threads. DeepSeek V4 Flash Q3_K_M,
using the model and tokenizer paths recorded in results.json. This is one fresh-process run, not a sustained
performance study. No user chats or prompts were used.

## Change and build

The permanent-expert budget subtracts both the runtime reserve and a separate loading allowance:
`max(0, free - reserve - load_margin)`. `--no-disk-vram-margin-mib` defaults to 512; 0 disables that extra
allowance. It only affects --no-disk. The allowance is not an allocated buffer and does not guarantee stable
whole-device free memory. The final runtime-reserve check remains, with actual byte counts and no claim that
crossing this threshold establishes spill.

Seven synchronized samples report free bytes, reserve, loading allowance, whole-device deltas since the
previous sample and planning, and the engine's requested allocation total. They bracket planning, RAM arena
allocation, CUDA registration, permanent-expert allocation, zeroing, weight copies, and file closure.

Code commits: 359fd2f1 (headroom/diagnostics/benchmark options) and 1a608e21 (the existing VRAM-first placement
and its tests). The binary was built from these source contents before the commits were recorded. Build hash
and the compiled placement-header hash are in build.json.

A clean CUDA build removed 124 target/dependency outputs with Ninja, then ran build-strata-poly.bat successfully.
The build explicitly compiled no_disk.cpp.obj (step 106/124), engine.cpp.obj, main.cpp.obj and the other engine
sources before linking. Ninja's refreshed dependency record includes no_disk.hpp. Earlier stale-object behavior
was observed, but its original cause was not established by this work.

CPU CTest: poly_no_disk_test, glm_conversations_test, glm_conversations_disk_test all passed. The placement,
reader and DMA-range executable reported 1564 checks. The new margin parser rejected -1, garbage, 512x and
2147483648 before model loading. Benchmark Python syntax passed. Linux/HIP and the mmap default path were not
built or benchmarked here.

## Measured run

Context 131072, batch 4096, read chunk 4096, staging 2848 MiB, runtime reserve 512 MiB, loading allowance
512 MiB; --no-disk --rows-vram, no draft, prompt cache 0, conversation cache 0, frozen experts. The full engine
command is in results.json. The benchmark copied the supplied expert profile into its temporary run directory.

- Load: 203.306 s; READY reached and generation completed.
- Exactly 10000 synthetic prompt tokens from tools/bench_poly_no_disk.py, greedy seed 42, 8 generated tokens.
- Prompt: 41990.7 ms, **238.148 tokens/s**, zero cached tokens.
- Decode: 331.5 ms for the engine's seven post-first-token intervals, about 21.12 tokens/s. Eight generated
  tokens are too short for a sustained decode-speed result.
- Generated IDs: 126889, 77070, 32901, 1229, 79, 38571, 603, 19093; all match the earlier synthetic runs.
- 461 experts permanently in VRAM (5037.17 MiB), 10547 in locked RAM.
- Engine VRAM allocations: 21667.78 MiB = dense 7758.37 + experts 5037.17 + KV 860.00 + staging 2848.00 +
  intermediate rows 256.00 + other 4908.24. Rounded components may differ from totals by 0.01 MiB.
- h_res_ = 0. Host weights and pinned buffers: 116967.70 MiB; registered windows are part of the resident weights.
- Free device memory after loading: 958.07 MiB; request report: 762.98 MiB. Both exceed the 512 MiB reserve.
  A later shutdown message sampled 737 MiB; these are different sampling times.
- Model file closed after READY. Process reads during the request: 52453 bytes, exactly the stdin command.
  Additional process reads: 0. Global system disk reads were nonzero and include other activity.
- Live VRAM resize was refused as expected. The process exited after QUIT; no test engine was left running.

## Where free VRAM changed

| Sample | Free MiB | Device use since previous MiB |
| --- | ---: | ---: |
| Before plan | 6068.00 | 0.00 |
| After RAM arena | 6068.00 | 0.00 |
| After CUDA registration | 6068.00 | 0.00 |
| After expert allocation | 1030.00 | +5038.00 |
| After expert zeroing | 960.95 | +69.05 |
| After expert copies | 958.07 | +2.88 |
| After files close | 958.07 | 0.00 |

The requested expert arena is 5037.17 MiB; whole-device use rose about 5109.93 MiB across loading. The difference
is about 72.76 MiB, including allocation rounding and costs or outside activity not captured in requested engine
bytes. Most of the difference appeared during zeroing; registration had no observed delta in this run. These
samples localize timing, not ownership, and do not separately measure WDDM spill. They do not establish an upper
bound for other runs or hardware. Free memory fell another 195.09 MiB by the request report while tracked engine
allocations stayed constant.

## Comparison limits

The earlier 233.186 tokens/s result used a stale placement object, context 16384, batch 1024, chunk 4096,
reserve 1024 and 422 VRAM experts. The historical 484.809 tokens/s run used context 16384, batch 1024,
full-context read rows and 457 VRAM experts. The 584.099 figure predates proper CUDA registration. None of these
comparisons isolates the cost of placement, chunking or registration. This change fixes loading headroom;
it does not demonstrate restoration of the earlier prefill speed.

Raw request metrics: results.json. Loading and request memory samples: memory.txt. Protocol: protocol.txt.
The original temporary run is L:/Temp/poly-bench-20261019/run-4-margin512. Local launcher configuration was not
edited; its existing 512 MiB runtime reserve remains and the new loading allowance defaults to 512 MiB.
