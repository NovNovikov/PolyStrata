# All model weights in RAM and VRAM

`strata-poly --no-disk` loads the model into locked RAM and fixed VRAM, then closes its GGUF files. It is opt-in.
Without the flag, the existing mmap loader, adaptive expert cache and host staging buffers are used.

The loader reads the GGUF headers without mapping the files. After the dense weights, scratch, full-context KV
and expert staging buffer have been allocated, it divides the routed experts between RAM and VRAM. Permanent
experts fill the remaining VRAM budget after the runtime reserve and loading headroom; RAM holds the rest.
The ranking is the existing cache's use-per-byte order. New use counts are recorded for future loads; they do
not change the placement of this run.

An expert permanently in VRAM has no RAM copy. Its slot is used directly by the existing GPU kernels. An expert
in RAM has its gate, up and down bytes together in an anonymous allocation. CUDA copies them directly from
a registered window of that locked region into the GPU's staging buffer. The two 512 MiB host staging buffers and the swap buffers
are not allocated in this mode. A staging buffer large enough for a layer lets a long read reuse that layer's
uploaded experts across its calls.

The embedding table is read into locked RAM at load, so token lookup also needs no file. DeepSeek's CPU
token-to-expert tables are copied and locked as well. Dense tensors upload through bounded temporary buffers; routed experts read
straight into their chosen RAM destination, or through a bounded upload buffer to their permanent VRAM slot.
No whole-model RAM copy precedes placement. Closing the files leaves their metadata available; a subsequent
attempt to read their payload fails instead of silently falling back to disk.

## Memory and options

| Option | Meaning |
| --- | --- |
| `--no-disk` | Require all weights to fit in locked RAM and fixed VRAM; close model files after loading |
| `--ram-limit-mib N` | Upper bound for resident host weights and known host buffers; 0 uses available RAM |
| `--rows-vram` | Allocate intermediate read rows in VRAM for the maximum read pass, at load |
| `--read-chunk N` | Limit tokens in a layer-by-layer read pass independently of the context; 0 uses existing limits |
| `--vram-reserve-mib N` | Keep this VRAM free, with the existing automatic reserve when omitted |
| `--no-disk-vram-margin-mib N` | Additional headroom deducted from the permanent-expert budget for loading; default 512, 0 disables it |
| `--stage-mib N` | VRAM through which RAM experts pass; at most one layer's experts |

RAM is sampled after the loader's existing allocations. The OS and other programs have already been deducted
from that available figure. The plan subtracts 1 GiB of headroom and, without `--rows-vram`, the maximum host
read-row buffer that may be allocated later. A RAM limit additionally subtracts the host weights and buffers
already allocated and those future read rows. It is not a limit on all process metadata, DLLs or chat caches.

Free VRAM is sampled after dense weights, full-context KV, scratch, staging and optional VRAM read rows exist.
The expert budget is `max(0, free - reserve - loading_headroom)`. The extra 512 MiB is a starting allowance,
not a guarantee against driver costs or changing memory usage by other programs. Each device allocation is checked
against free dedicated VRAM before it is requested, rather than relying on Windows to refuse a spill into RAM.
The loader refuses a model that does not fit, or a host region that cannot be locked in full, with byte counts.
It samples VRAM before planning, after the RAM arena, CUDA/HIP registration, expert allocation, zeroing, copying,
and file closure. Reports include free bytes, the reserve and loading allowance, changes since the previous
sample and the plan, and the engine's requested allocation total. Whole-device changes do not identify their
owner or prove spill. A final reserve refusal prints the actual free, reserve and loading-headroom byte counts.

Windows uses `VirtualAlloc` and the platform's working-set minimum plus `VirtualLock`, as llama.cpp does for
residency. Linux uses anonymous memory and `mlock`; the account must have a sufficient locked-memory allowance
(`ulimit -l`). OS residency alone does not register a DMA source with CUDA. Before an expert upload, its original
source pages are registered with `cudaHostRegister` (HIP: `hipHostRegister`), and the registration is verified.
No second RAM buffer receives the expert. The window is at most the staging capacity plus page rounding;
adjacent or overlapping page ranges are merged. The next window waits for source-copy events on every stream
that used the previous one before unregistering it. An interrupted copy without its final event is synchronized
before releasing its source as well.

Loading first verifies a staging-sized registration window. Any registration or verification refusal is an
error, including during generation; there is no pageable-copy fallback. Windows refused one 110 GiB CUDA
registration on the RTX 4090 test PC even though all 110 GiB could be OS-locked; a 4 GiB window in that same
locked arena registered successfully. Bounded registration avoids requiring CUDA to register the whole model.
This is not a giant `cudaMallocHost` arena. A failed load releases registrations, allocations and OS locks.

Expert swaps, cache growth, shrinking and VRAM resizing are disabled. The line protocol's `VRAM` request is
refused, and the server is told `vram_elastic=0`. The fixed expert placement stays valid throughout the run.
This mode currently requires one card; `--layer-split` is refused. The host/DMA buffer sizes depend on the model
and batch rather than a hard 100 MiB limit.

`--rows-vram` is separate from `--no-disk`. Without it, intermediate read rows retain their existing lazy pinned
RAM path. With it, their size is `max_read * streams * embedding_width * sizeof(float)` in VRAM. Set
`--read-chunk 4096` to limit a read pass without lowering the attention context; a smaller read can cause more
expert uploads between passes. Long reads copy
those rows between device buffers, with the same stream/event ordering as the host path. A larger context can
therefore fail the VRAM budget even when the model itself fits unless the read size is limited.

The flag governs model weights. Session SAVE/RESTORE and expert-profile persistence remain separate features.

The stderr `memory[load]` report shows the context limit, allocated KV in VRAM and the full context's KV size,
read rows in VRAM, and `h_res_` in RAM with its maximum possible size. Without `--rows-vram`, `h_res_` starts at
zero; `memory[h_res_ grown]` reports its allocation when a prompt needs it. `memory[request]` refreshes the totals
after an answer. A refused device allocation or expert placement also prints the partial allocation breakdown.

VRAM allocation totals add dense weights, experts, KV, staging, read rows and other engine allocations. RAM
weight/buffer totals add resident host weights and pinned buffers; `h_res_` is already part of pinned buffers.
These requested-byte totals exclude CUDA library/graph allocations and ordinary process metadata. Separate
whole-device usage comes from `cudaMemGetInfo` and includes CUDA runtime overhead and other processes;
whole-process RSS shows resident RAM pages, including metadata and libraries. The CUDA figures do not separately
measure Windows WDDM spill into RAM. RAM and VRAM totals are reported separately. A reserve is free headroom, not
an allocation, and is not added to occupied memory. Numbers are in MiB, rounded to two decimal places.
The CUDA/HIP-registered window and its peak are part of the resident weights, not additional RAM allocations.

## Checking a machine

CPU tests, built with `STRATA_BUILD_CONVERSATION_TESTS=ON`:

```text
poly_no_disk_test
glm_conversations_test
glm_conversations_disk_test
```

`poly_resident_probe MIB [RUNS]` deliberately allocates and locks that amount of RAM, then frees it. Do not run
a large probe beside a loaded model. On Windows with 128 GiB RAM, 112640 MiB locked in 21.53 s during the
2026-10-18 test (RTX 4090 PC, i5-13600KF). A partial lock is a failure, not a successful smaller allocation.

The GPU build also provides `poly_resident_dma_probe RAM_MIB [WINDOW_MIB] [RUNS]`. It OS-locks the requested
arena, switches a bounded CUDA/HIP registration between its beginning and end, verifies real copies on two
independent streams, and checks cleanup. It needs no model file. The default window is 3072 MiB.

`tools/bench_poly_no_disk.py` creates synthetic text and starts a fresh engine for each run. It copies a supplied
expert profile into the output directory; it does not use user chats or prompts. Run it with the server's Python:

```text
python tools/bench_poly_no_disk.py --exe PATH/strata-poly.exe --model MODEL.gguf --tokenizer TOKENIZER_DIR --profile PROFILE.txt --output RESULTS_DIR --context 16384 --tokens 10000 --batch 1024 --stage-mib 3072 --reserve-mib 1024 --rows-vram --skip-default --no-disk-runs 2
```

With `--baseline-exe OLD_EXE` and without `--skip-default`, the tool
also compares the generated token IDs of the old and current mmap paths. It records read speed, zero prompt
reuse, process memory, system disk counters, process read bytes and whether the model file is still open. The
engine's stdin token command itself counts as process I/O; the tool reports its size separately. System disk
counters include unrelated activity and must not be called model reads without attribution.

Measured results and limits on the RTX 4090 Windows PC are recorded in
[the 2026-10-18 run](../bench/results/2026-10-18-poly-no-disk/README.md).
