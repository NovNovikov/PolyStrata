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

## What it bought, at the scale of a real chat

The scenario above is 10511 tokens. The scale that matters is a long conversation, so the same test was
run with a 89999-token conversation (`tools/bench_poly_session_scale.py`): one long synthetic prompt,
a different prompt that branches off it (which parks the conversation to a file in the session store),
and the first prompt again (which brings it back from that file). The old engine and the new one, the
same machine, the same conversation:

| | old exe `d25903c1` | new exe `2b9165ce` | |
|---|---:|---:|---|
| the session file | 4 280 290 660 B | **652 459 431 B** | 6.6 times smaller |
| park (the first) | 7 765.7 ms | **2 537.2 ms** | 3.1 times faster |
| park (the second, the RAM is fuller) | 15 271.2 ms | **2 352.8 ms** | 6.5 times faster |
| restore | 16 595.2 ms | **3 207.9 ms** | 5.2 times faster |

The user's own log, on their real conversation, is the same order as the old engine here: 14 670.5 ms
to park 90601 tokens and 32 114.5 ms to restore 74507 tokens. On the same scale the new engine parks
in 2.5 s and restores in 3.2 s.

And on the smaller scenario, with the file's size alone:

| | old exe `d25903c1` | new exe `2b9165ce` |
|---|---:|---:|
| session file for 10511 tokens | 603 114 484 B | **179 411 487 B** (29.7 %) |
| SAVE (write) | 589.8 ms | **389.3 ms** |
| SAVE (again) | 408.9 ms | **251.1 ms** |
| RESTORE (read back) | 369.9 ms | **196.7 ms** |
| RESTORE after a process restart | 670.8 ms | **319.5 ms** |

So the restore is now 1.9 times faster than the old file's, and the write 1.5 times faster, on top of
the file being 3.4 times smaller. The other gain is not in the numbers above: the restore no longer
holds the file's bytes in RAM a second time, so the host memory a restore needs is one copy of the
unpacked payload instead of two (4.4 GB instead of 8.8 GB for a 90601-token branch).

## The reader that was slower, and why

The first packed build (`989ceb9a`) read a packed file *slower* than the old one: 398.7 ms against
369.9 ms warm, and 2348.5 ms against 670.8 ms after a process restart. The file was smaller, and the
restore was slower, which is not a trade worth making.

The cause was in the reader, not in the packing: the decoder read the file's bytes run by run (an
8-byte header for every run, then the run's bytes), which is 339 247 reads for one session file. A
cold file then costs more in small reads than the whole unpacked file cost in one read.

Measured on the real file (171.1 MiB in the file, 575.1 MiB unpacked), the same file read two ways:

| | cold | warm | reads |
|---|---:|---:|---:|
| a read for every run (the first build) | 429.3 ms | 229.7 ms | 339 247 |
| one read for the block, the decode out of memory | 160.0 ms | 166.7 ms | 30 |

The reader now takes the file's bytes in blocks of 1 MiB and decodes them out of the block, so the
reads are 30 for the whole file. That is the `2b9165ce` build, and the numbers above are its.

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
- **A 4.38 GB branch is still a 4.38 GB file of rows.** Packing makes it 622 MiB, but the rows
  themselves are what they are: the engine's compressed attention cache for 90601 positions. The
  restore still brings those rows into the host RAM (4.4 GB), because the engine's rows are in VRAM
  and the file is the only other place they are.
- The host copy of the rows is the last big thing: the restore reads the file (622 MiB now) and unpacks
  it into 4.4 GB of host memory, which the engine's window does not need. What that would take is in
  `NEXT-STEPS.md`.

## Files

- `packed/` - the new engine's run: `summary.json`, the engine's stderr, the protocol, per phase.
- `old/` - the old engine's run, same scenario.
- The engine's own numbers are in each run's `engine.stderr`; the session files are under the work
  directories (`L:\Temp\opencode\bench-packed`, `L:\Temp\opencode\bench-old`).
