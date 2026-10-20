# Next step for the session cache: where the rows live

Written 2026-10-20, after the packing commit `8980dce6`. What the code says now, and the three ways
forward, with what each one costs. Nothing here is implemented; the packing and the read without the
second copy are done and measured (`README.md` in this directory).

## What the code says

- `DeepSeek4::save_card_rows` copies device to host and `load_card_rows` copies host to device
  (`src/poly/deepseek4.cpp`): `B.rows` and `B.keys` are device pointers. The rows of a conversation
  live in VRAM; a host buffer is only the way to them.
- `mount()` reads every record of the file into a blob (`load_blobs`), hands each blob to
  `e_.load_rows(...)` and then keeps the blobs in the slot (`s.rows = std::move(rows)`,
  conversations.hpp:381). So after a restore the rows are in the device **and** in the host RAM, and
  the host copy is the size of the whole conversation: 4.4 GB for a 90601-token branch.
- `park()` writes the rows of the blobs and the engine's tail in chunks of about 4 MiB
  (conversations.hpp:638). The file is therefore a thousand small records, not one big one.
- A slot whose blobs are in its session file counts nothing towards the host memory
  (`droppings()`, conversations.hpp:470): the design already has a "the rows are on the disk" state
  (`Slot::disk`, `Slot::disk_marks`).

## The three ways forward

**1. Load a record into the device and give the host buffer back.** The peak host memory of a restore
becomes one record (4 MiB) instead of the whole conversation. What it costs: the slot no longer holds
the rows, so a later park has to capture them from the device again (`save_rows` reads the device),
and it can only do that while the device still holds them. For a conversation longer than the
context the device has dropped the oldest rows, and the conversation could not be parked afterwards.
The rows would have to live in the file, which is the state `Slot::disk` already describes.

**2. Delta files.** A new file holds only the rows added since the parent, and the parent is named in
a sidecar (the fork: `parent_id`, `range_lo`, `range_hi`). The write becomes small; the read does
not, because the read needs the whole chain. It also needs a second file format and a restore that
walks the chain.

**3. A background writer.** The write leaves the request path (the fork: `auto_enqueue_snapshot`).
For the disk path here it needs the rows in RAM first, because the engine moves on while the writer
runs: 4.38 GB for a 90601-token branch. That is why this path streams the engine's tail straight
into the file instead, and a background writer would give that up.

## What I would do next, and why

The packing (done) makes the file 3.4 times smaller, so the write and the read of the file are smaller.
The `RESTORE` command (done) reads the file's records one at a time and hands each to the engine, so it
needs room for one record and not for the whole conversation.

What is left is the conversation cache's own path, which is the one a client's next prompt takes:

- `materialize()` reads a parked conversation's file into blobs: the rows of the whole conversation (4.4 GB
  at the 89999-token scale) in the host RAM. `enter()` then copies them into the engine. The rows are
  already in the engine's VRAM after a restore, and the file is where they are between the runs, so the
  blobs are a third copy that nothing needs.
- The fix is the same as for `RESTORE`: read the file's records one at a time and hand each to the
  engine. The order matters: `enter()` parks the leaving conversation (which reads its rows from the
  engine) *before* the incoming conversation's rows are loaded, so the incoming rows must be read after
  that, and the file's checkpoints (the states) must be in the host before it, because `state_at()` wants
  them.
- The catch is the same one the `RESTORE` path has: a conversation longer than the engine's window has
  rows the engine no longer holds, and then the file is the only copy and the blobs are needed. So the
  change has to ask the engine whether it still holds the rows (`kv_rows_` and the reach), and keep the
  blobs when it does not.

2 (the delta files) comes after that, if the write is still the slower half. 3 (the background writer)
needs the rows in RAM, which is what the above removes.

## The delta files, worked out (2026-10-20)

The park writes the whole conversation again every time: 4.3 GiB of rows for 89999 tokens, of which
622 MiB reach the disk after the packing. A delta file holds only the rows added since the parent, and
names the parent. The Manual LLAMA_CPP fork does this (`parent_id`, `range_lo`, `range_hi`), and it is
the step that makes the write small enough not to matter.

**The format.** The header (56 bytes now: magic, version, n_seq, prompt, sb, rb, n_marks, n_rows) gets
the parent's name in a fixed 64-byte field; empty means there is no parent. The version goes to 4. The
offset where the rows begin is not needed: a delta's rows begin where the parent's end, and the chain is
read by the names.

**The writer** (`src/poly/session_file.hpp`, `write_stream`): a `parent` parameter, written into the
header. The marks and rows are written as they are, so a delta file carries only [k, end).

**The reader**: `read_head` gives the `parent`; a new `read_chain(path)` walks the parents to the root and
reads the chain from the root down, joining the marks and the rows in position order. One pass over the
bytes, as now; the host holds one record at a time.

**The cache** (`src/poly/conversations.hpp`):
- `Slot` gets `disk_rows_end`: how far the rows in its file reach;
- `park()`: when the slot has a file and its `disk_rows_end` is below the end, the new record is a delta
  [disk_rows_end, end) whose parent is the old file; otherwise a whole file;
- `take()` and `branch()` carry `disk` and `disk_rows_end` over (the rows of a copy are the same bytes:
  they are a function of the tokens);
- `materialize()` and `enter()` read the chain, not one file.

**The traps:**
1. The eviction. `disk_fit()` drops the oldest file; a parent cannot go while a child lives, so the
   leaves are dropped and the chains are counted whole. Otherwise a delta is orphaned and the
   conversation is lost.
2. The branch. A new branch off an old one shares a prefix but its file is not that branch's delta: a
   delta is only valid from a file with the *same* prefix, which the cache checks with `common()`.
3. The marks. A delta carries only the marks after the parent's end; else the chain holds two of them
   and `state_at()` finds the wrong one.
4. The names. A file's name is a hash of the whole token sequence, so every park gives a new name and
   the chain grows. The parent link has to be by name, not by hash.
5. An empty delta. When there are no new rows, no file is written at all.

**The order, so that every step is testable:**
1. The format: the parent's field, version 4, the writer takes the parent, `read_head` gives it. The
   format's own tests stay green and the old files are still read.
2. The chain reader: `read_chain`, used by `mount()` and `materialize()`. Nothing writes a delta yet, so
   nothing changes.
3. The cache: `disk_rows_end`, the delta in `park()`, `take()` and `branch()`.
4. The eviction with chains, and the prefix check.
5. The measurement: the 89999-token scenario, where the park should go from ~1.7 s to tens of
   milliseconds; the files on the disk add up to about the same as before (the chain is as big as the
   whole file was).

Roughly 300-350 lines in two files, plus the tests. It is a change of its own.


The same scenario as the numbers in `README.md`: a 90601-token branch parked to a file and read back,
on a machine with the RAM the user has. The number to beat is 14.7 s for the park and 32.1 s for the
restore, and the RAM peak during the restore, which is 4.4 GB now.
