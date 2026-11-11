# Diagnostic DeepSeek prefill run — invalid chat encoding

**Do not use this series as an ordinary-chat performance comparison.** The benchmark called
`Tokenizer.encode(rendered)` without `parse_special=True`, omitted the template's BOS parameter,
and bypassed the server's literal-control protection. The engine therefore received ordinary BPE
pieces spelling the User/Assistant markers instead of their control-token IDs. Five of six answers
immediately returned EOS. Source code, traces and results are retained intact.

The mistake is in the benchmark input construction, not a discovered difference in numerical launch
parameters. Offline reproduction: the old first IDs are `[30, 28217, 6756, 28217, 32, ...]`; the actual
server starts this DeepSeek request with BOS `0`, User `128803`, and ends on Assistant `128804` and
thinking-open `128821` under its defaults.

The earlier GLM input in `2026-10-20-glm-mmap-vs-no-disk/input.json` has the same mistake: 2050 old
IDs versus 2032 IDs when its saved public rendering is encoded with control recognition. Its old
first IDs `[32365,49544,33974,...]` differ from `[154822,154824,154826,...]`. The previously quoted
approximately +6% GLM result is withdrawn as evidence of ordinary-chat throughput. Unit tests and
payload-transfer tests are independent of this prompt-construction error.

## Recorded diagnostic numbers, not ordinary-chat benchmarks

The numerical parameters are those in `strata-dsv4-q3_k_m.json`: context 131072, batch 4096,
chunk 32768, stage 2848 MiB, rows-vram, compact-scratch, compact-graphs, VRAM reserve 256 MiB,
no-disk loading margin 256 MiB, swaps 64. Native `--serve`; fresh isolated session directories and
identical opaque copies of the configured expert statistics. User prompts, logs and conversations
were not read. The mmap variant only removes `--no-disk`; the exchange variant only adds
`--no-disk-expert-exchange` (default maximum 1).

All six requests read 50000 IDs with resume=0. At the requested cap of 256 outputs, the first five
returned EOS after one output; the last mmap request generated 256 outputs at 6.631 tok/s. That is
not a usable decode comparison. No automatic exchange occurred in the exchange variant.

| Mode | READY, s | First full read, tok/s | Second full read, tok/s | Second read excluding park, tok/s |
|---|---:|---:|---:|---:|
| Bat no-disk | 201.62 | 582.69 | 579.38 | 587.76 |
| Bat plus exchange | 207.49 | 591.50 | 586.90 | 593.20 |
| Bat minus no-disk (mmap) | 40.09 | 234.45 | 700.70 | 710.17 |

Progress at exactly 32768 IDs: no-disk 706.0/698.1, exchange 718.6/707.5, mmap 223.8/803.0 tok/s.
These describe only the malformed token streams. Second-request park durations: 1231.5 / 905.8 /
951.1 ms. Excluded-park timing is measured from RESUME 0 arrival to the final PP arrival; full
prompt timing is the engine's DONE timing and includes parking.

Mmap first prefill: 36,178,148 process page faults, 136.973 GiB system-wide disk reads. Second
prefill: 7427 faults, 0.269 GiB system-wide reads. Faults include soft faults; disk figures are not
exclusive process I/O. These distinguish first and warm file access but do not cure the chat-format
error or establish a real-request throughput result.

Own no-disk device allocations stayed 22306.09 MiB (dense 7758.37, experts 6202.09, KV 860,
staging 2848, rows 2048, other 2589.63), with 567 main GPU experts in both variants. Mmap started
with 658 experts/22568.00 MiB and ended at 22510.00 MiB (experts 6848, KV 418; other categories
unchanged). GPU clock medians during all six prefill phases were 2430 MHz. This was one fixed-order
series; no external applications were stopped and no cold OS-cache state was forced.

`manifest.json` records source/binary/config hashes and the launch command. `summary.json`, all raw
traces and per-request files are preserved. `final-verification.json` confirms the original exe,
JSON, bat and configured expert profile are unchanged, and all model processes exited.

## Prepared correction

`../2026-10-21-deepseek-bat-prefill-specials/run.py --prepare-only` has prepared two full 50000-token
public requests using the actual `Service.encode_prompt` method, server template BOS parameters,
default thinking and protection of control-token literals inside user text. Cached and full server
encoding match exactly for both. See that directory's `input-validation.json` and `public-inputs.json`.
No corrected model run has been launched. Repetition needs the user's authorization under AGENTS.md.
