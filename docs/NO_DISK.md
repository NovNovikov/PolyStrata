# All model weights in RAM and VRAM

`strata-poly --no-disk` loads the model into locked RAM and fixed VRAM, then closes its GGUF files. It is opt-in.
Without the flag, the existing mmap loader, adaptive expert cache and host staging buffers are used.

The loader reads the GGUF headers without mapping the files. After the dense weights, scratch, full-context KV
and expert staging buffer have been allocated, it divides the routed experts between RAM and VRAM. Only the
minimum prefix of the profile's ranking needed to fit the remaining experts into RAM becomes permanent VRAM.
The ranking is the existing cache's use-per-byte order. New use counts are recorded for future loads; they do
not change the placement of this run.

An expert permanently in VRAM has no RAM copy. Its slot is used directly by the existing GPU kernels. An expert
in RAM has its gate, up and down bytes together in an anonymous allocation. CUDA copies them directly from
that locked region into the GPU's staging buffer. The two 512 MiB host staging buffers and the swap buffers
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
| `--rows-vram` | Allocate intermediate read rows in VRAM for the entire requested context, at load |
| `--vram-reserve-mib N` | Keep this VRAM free, with the existing automatic reserve when omitted |
| `--stage-mib N` | VRAM through which RAM experts pass; at most one layer's experts |

RAM is sampled after the loader's existing allocations. The OS and other programs have already been deducted
from that available figure. The plan subtracts 1 GiB of headroom and, without `--rows-vram`, the maximum host
read-row buffer that may be allocated later. A RAM limit additionally subtracts the host weights and buffers
already allocated and those future read rows. It is not a limit on all process metadata, DLLs or chat caches.

Free VRAM is sampled after dense weights, full-context KV, scratch, staging and optional VRAM read rows exist.
Only the reserve and permanent experts come out of that remaining space. Each device allocation is checked
against free dedicated VRAM before it is requested, rather than relying on Windows to refuse a spill into RAM.
The loader refuses a model that does not fit, or a host region that cannot be locked in full, with byte counts.

Windows uses `VirtualAlloc` and the platform's working-set minimum plus `VirtualLock`, as llama.cpp does for
residency. Linux uses anonymous memory and `mlock`; the account must have a sufficient locked-memory allowance
(`ulimit -l`). CUDA transfers take their source directly from the locked host region; this is not a giant
`cudaMallocHost` arena. A failed load frees its allocations and unlocks its regions.

Expert swaps, cache growth, shrinking and VRAM resizing are disabled. The line protocol's `VRAM` request is
refused, and the server is told `vram_elastic=0`. The fixed expert placement stays valid throughout the run.
This mode currently requires one card; `--layer-split` is refused. The host/DMA buffer sizes depend on the model
and batch rather than a hard 100 MiB limit.

`--rows-vram` is separate from `--no-disk`. Without it, intermediate read rows retain their existing lazy pinned
RAM path. With it, their size is `context * streams * embedding_width * sizeof(float)` in VRAM. Long reads copy
those rows between device buffers, with the same stream/event ordering as the host path. A larger context can
therefore fail the VRAM budget even when the model itself fits. Lower the context or omit `--rows-vram` if needed.

The flag governs model weights. Session SAVE/RESTORE and expert-profile persistence remain separate features.

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
