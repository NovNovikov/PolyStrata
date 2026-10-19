# Windows no-disk RAM admission

Windows, RTX 4090 24 GiB, 128 GiB RAM, DeepSeek V4 Flash Q3_K_M. All inputs are synthetic.
No private logs, conversations or prompts were read. This change concerns host RAM admission only.

## Change

In Windows `--no-disk`, `ullAvailPhys` is diagnostic, not a hard limit for expert placement.
Without an explicit `--ram-limit-mib`, the loader attempts the required anonymous allocation and full
working-set/VirtualLock residency. Allocation or incomplete locking still fails loading. It does not
fall back to pageable weights. A failed VirtualAlloc reports the captured Windows error code.
The initial Windows rejection for an unavailable/zero physical-memory snapshot is removed as well.

An explicit cap still includes existing resident weights, known pinned host buffers and future host
read rows, and can reject the plan before allocating the expert arena. The pure policy tests also
exercise the unchanged available-memory admission for other platforms, including future-row underflow
and host-byte overflow. The mmap path and VRAM placement/reserves are unchanged.

## Validation

`cpu-tests.txt`: the existing no-disk suite, including the new RAM policy cases, passed.
`resident-probe.txt`: two independent 1024 MiB allocations, with every page checked through
QueryWorkingSetEx for both Valid and Locked. Three provisional allocation/release cycles preserve byte
counts and restore working-set quotas. An impossible address-space allocation reports a Windows error.

`build.txt`: Windows/CUDA build of strata-poly, poly_no_disk_test and poly_resident_probe succeeded.
HIP, SYCL and Linux were not built or run. No claim of measured cross-platform behavior is made.

## Actual loader under RAM pressure

The test creates a separate, unlocked process, allocates and writes 8192 MiB of pageable anonymous
memory, and keeps it alive while testing both binaries. It retouches its pages before each load.
It frees that allocation and exits after testing. Its working set is managed by Windows; an 8192 MiB
allocation does not imply an 8192 MiB reported RSS. Measurements retain the actual RSS and availability.

Both binaries use context 131072, prompt batch 4096, read chunk 32768, staging 2848 MiB, rows in VRAM,
compact scratch, runtime VRAM reserve 256 MiB, loading margin 256 MiB, 12 threads, no draft and lookup 0.
The existing benchmark provides 10000 synthetic prompt tokens and requests 128 output tokens, with the
adaptive expert cache enabled and payload verification outside timing. Results are a functional test,
not a throughput comparison: the old process fails admission and has no generation timing.

The previous binary is the already measured compact build, SHA256
`96c402130e67d20de09217ea7652ed6a8e9adf3d297da68ddaa0bdec83328d82`.
The exact new hash and commands are recorded alongside the results.

| Observation | Previous binary | New binary |
| --- | ---: | ---: |
| Expert arena required in RAM, MiB | 114274.69 | 114274.69 |
| Available RAM at placement, MiB | 110292.73 | 110321 (integer diagnostic) |
| Result | Refused before arena allocation | Full allocation/locking, READY, generation complete |
| Load time, seconds | No completed load | 220.20 |
| Permanent GPU experts | No completed placement | 578 |
| Locked resident weights including embeddings, MiB | No completed placement | 115293.56 |
| Resident weights plus known host buffers, MiB | No completed placement | 115686.38 |
| Available RAM after load, MiB | Not applicable | 7382.61 |

The previous planner rejected a 4175364096-byte (3981.94 MiB) shortfall against the snapshot.
The new loader passed full locking despite a roughly 3954 MiB snapshot shortfall. It stabilized placement
on its first attempt. This establishes that this Windows load can succeed beyond the initial availability
snapshot; it does not establish which other process or Windows memory list supplied every reclaimed page.

The new process consumed all 10000 prompt tokens (19437 ms, 514.48 tok/s), generated all 128 requested
tokens (6349.4 ms decode, 20.00 tok/s excluding the first token), and quit successfully. Decode hit rate was
50.60%, with 305 adaptive uploads and 254/255 staging slots populated at the end. All 254 GPU cache payloads
matched their RAM originals byte for byte, with zero kernel tails. This is a short functional run rather
than a warmed cache or throughput comparison.

Model files were closed after loading. During the request the process read exactly its 52455 stdin bytes,
with zero additional process read bytes. System-wide disk reads/writes are retained but are not attributed
to the engine. There is no claim that other processes or Windows made no disk accesses.
VRAM reserves remained 256 MiB plus 256 MiB loading headroom. Free VRAM after the request was 411.07 MiB;
whole-device changes are not a measurement of their owner or of spill.

The pressure process exited with code 0, its allocation was released, the engine exited, and a final
process-name check found no strata-poly process. `reclaim-summary.json` records 123397.73 MiB available
after cleanup. The exact temporary driver is retained in `driver-used.py.txt`; it was run from build-poly-cuda12.
It deliberately allocates 8 GiB and needs unused output paths when rerun.

No full-model inference with host read rows (`--rows-vram` absent), an explicit host cap, or other models
was run in this change. Future host rows and explicit caps are exercised by the pure arithmetic suite.
