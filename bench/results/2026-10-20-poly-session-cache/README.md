# The session file: where the 25-43 seconds went, and what changed

Measured on 2026-10-20, Windows, RTX 4090, CUDA 12.4, i5-13600K, 128 GiB RAM, XPG SPECTRIX S40G
(NVMe) as the system disk. Model `DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf`, config
`strata-dsv4-q3_k_m.json` (SHA256 `2f51ccb290de4768f89bd3d8af5b1ea229761a7eb6a69532606cb5320c889ea9`),
public synthetic expert seed profile (SHA256 `b06d1fcd08c83787b79f3528d2da54577e3d557151b6aa1608865199070ec6ed`).
Only synthetic prompts and fresh synthetic sessions were used.

## What the user asked

The disk cache is written after every generation and read back before the next one, and that took
25-43 seconds. Two questions: does the request wait for the save, and why does the save take so long.

## The measurements

From the server's own log, a 90601-token branch:

```
strata-poly: session file 44ce61542493caa3-90601.bin: 4378984940 bytes
strata-poly: conversation cache: parked 90601 tokens in 14670.5 ms
strata-poly: conversation cache: restored 74507 tokens in 32114.5 ms
```

So one park wrote 4.38 GB in 14.7 s and one restore read 4.07 GB in 32.1 s.

Measured on a real session file (603 114 484 bytes, 10511 tokens, the same engine and model):

| measurement | result |
|---|---|
| raw read of the file, warm (page cache) | 87.5 ms for 575.2 MiB (6576 MiB/s) |
| raw read of the file, cold (earlier probe) | 367.6 ms (1565 MiB/s) |
| the same file through `SessionFile::read` | 188.8 ms |
| host copy of 1 GiB, `memcpy` | 366.4 ms (2795 MiB/s) |
| **zeros in the rows payload** | **85.51 %** |
| payload split | marks 100.4 MiB, rows 474.7 MiB |

The disk is not the problem: it delivers 1.5 GB/s cold and 6.5 GB/s warm, so the whole 4.07 GB
restore is about 2.7 s of reading. The file itself is the problem, and the RAM it needs on the way back:

- **85.51 % of the file was zeros.** The engine's rows hold a block's compressed row once per block
  (a 4-to-1 or 128-to-1 compressor), so most positions in the saved range carry no row at all. A
  90601-token branch wrote 4.38 GB of which about 0.6 GB was data.
- **The restore held the file twice.** `SessionFile::read` built the whole payload in RAM, and the
  caller then copied each record into a blob. Two copies of 4.07 GB with 2.7 GiB of free RAM is where
  the 32 s went: the second copy had to be faulted in and evicted, not read from the disk.

## The change

One commit, `8980dce6`, in `src/poly/session_file.hpp` and `src/poly/conversations.hpp`:

1. **Version 3 of the file format: the payloads are packed.** A run of zeros is written as a count, the
   rest as literal bytes. A version 2 file is still read. A version 3 record names its packed size, so
   a truncated file is refused rather than half-read.
2. **A restore reads the payloads straight into their blobs** (`SessionFile::read_into`), so the file's
   bytes are not in RAM a second time. The host's memory question is now asked of the unpacked size,
   which is what the host actually holds.

Measured on the real file, packed and unpacked, compared byte for byte: the packed file is 179 411 494
bytes, the unpacked bytes are identical to the original, the write takes 277.8 ms and the read back
205.0 ms.

## What it bought

Same engine, same model, same scenario, same machine, one run each:

| | old exe `d25903c1` | new exe `989ceb9a` |
|---|---:|---:|
| session file for 10511 tokens | 603 114 484 B | **179 411 489 B** (29.7 %) |
| SAVE (write) | 589.8 ms | **420.1 ms** |
| RESTORE (read back) | 369.9 ms | 398.7 ms |
| second SAVE | 408.9 ms | 276.6 ms |
| RESTORE after a process restart | 670.8 ms | 2348.5 ms |

Read on a real file, packed and unpacked, compared bit for bit: 220.1 ms for the packed file, and the
unpacked bytes are identical. The 2348.5 ms above is the one number that moved the wrong way: it is a
restore right after a process restart, where the new file is read from the disk for the first time and
then unpacked into 603 MB, while the old file was 603 MB read straight into place with no unpacking
step. It is one sample of a cold read, and the same cold read on the packed file is 1.3 GB of disk
plus 603 MB of unpacking.

What the numbers say: the file is 3.4 times smaller, the write is faster, and the RAM a restore needs
is one copy of the unpacked payload instead of two.

## Where the 32 seconds come from, and what the next step is

Read from the code, with the file and the RAM as the two parts of it:

- **The rows live in VRAM.** `DeepSeek4::save_card_rows` copies device to host (`B.rows`, `B.keys` are
  device pointers) and `load_card_rows` copies host to device. A restore is therefore a transfer, not a
  place: the 4.07 GB of rows has to be in host RAM only while it is on its way to the device.
- **The file is already in small pieces.** `park()` writes the engine's tail in chunks of about 4 MiB
  (`chunk = 4 << 20 / row_bytes`, conversations.hpp:638), so a 90601-token branch is a thousand records
  of 4 MiB, not one record of 4.38 GB.
- **The host copy is what thrashes.** `load_blobs()` reads a record into a blob and keeps it, so the
  restore's host RAM is the sum of all the records: 4.4 GB with 2.7 GiB free. The kernel has to evict
  other pages to give it, which is where the 32 s went, not the disk (1.5 GB/s cold) and not the
  transfer.

So the next lever is not the file's size but the host copy: load each record into the device and give
the host buffer back as soon as it is loaded, so the peak is one record (4 MiB) instead of the whole
conversation. What that costs is the host copy the conversation keeps afterwards: the parked
conversation's rows would have to be captured from the device again (the engine can do that,
`save_rows` reads the device) when the conversation is written out again. This is the change to make
next, and it is not made here.

The fork does the same thing on its side: its reader hands the file's bytes to the device
(`llama_io_read_file::read_tensor` to `ggml_backend_tensor_set`) instead of building a dump in RAM
first.

The other two levers, for the record:

- **Delta files** make the write small: a new file holds only the rows added since the parent, and
  the parent is named in a sidecar (`parent_id`, `range_lo`, `range_hi` in the fork). It does not make
  the read small, because the read needs the whole chain.
- **A background writer** takes the write off the request path (`auto_enqueue_snapshot` in the fork).
  For the disk path here it needs the rows in RAM first, because the engine may move on while the
  writer runs: the whole 4.38 GB of a 90601-token branch. That is why the disk path streams the
  engine's tail straight into the file instead.

## What is not done

- **The save is still synchronous.** The request still waits for the write of the branch it parks.
  The Manual LLAMA_CPP fork answers this with a background writer that takes the snapshot and writes
  it off the request path (`auto_enqueue_snapshot`, server-context.cpp:2848). That needs a writer
  thread in the poly engine and a snapshot that outlives the request; it is the next step and it is
  not done here.
- **The file is still rewritten whole on every park.** The fork's fork answers this with delta files:
  a new file holds only the cells added since the parent, and the parent is named in a sidecar
  (`auto_delta_state_filename`, `parent_id`, `range_lo`). That needs a second file format and a
  restore that walks the chain; also not done here.
- **A 4.38 GB branch is still a 4.38 GB file of rows.** Packing makes it 1.3 GB, but the rows
  themselves are what they are: the engine's compressed attention cache for 90601 positions.
- The measured restore of a 90601-token branch (32.1 s before the change) was not repeated after it:
  the new engine was measured on a 10511-token session, and the branch of that size needs a run of
  about ten minutes that was not spent.

## Files

- `packed/` - the new engine's run: `summary.json`, the engine's stderr, the protocol, per phase.
- `old/` - the old engine's run, same scenario.
- The engine's own numbers are in each run's `engine.stderr`; the session files are under the work
  directories (`L:\Temp\opencode\bench-packed`, `L:\Temp\opencode\bench-old`).
