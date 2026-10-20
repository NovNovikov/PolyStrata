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

1 first, in this order:

- The packing (done) makes the file 3.4 times smaller, so the write and the read of the file are
  smaller.
- 1 then: the host copy of the rows is the only thing left that is large, and it is large because the
  device already has it. Make the restore load a record, hand it to the device and let it go, and
  let the file be the home of the rows of a conversation that is not in the device's window. The
  change is in `mount()`, `load_blobs()` and `park()`, and the parked conversations' file handling
  has to be able to write a conversation whose rows are in the file and in the device's tail.
- 2 after that, if the write is still the slower half: the delta file is the format that makes a
  park cheap, and the fork has one to copy.

3 last, and maybe not at all: a background writer needs the rows in RAM, which is what 1 removes.

## The test that says it worked

The same scenario as the numbers in `README.md`: a 90601-token branch parked to a file and read back,
on a machine with the RAM the user has. The number to beat is 14.7 s for the park and 32.1 s for the
restore, and the RAM peak during the restore, which is 4.4 GB now.
