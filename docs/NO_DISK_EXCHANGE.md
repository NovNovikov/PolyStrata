# Adaptive main expert placement in no-disk mode

`--no-disk-expert-exchange` enables an experimental exchange of expert ownership between locked RAM
and the main GPU expert arena. It requires `--no-disk`. Without this flag the existing staging copy
cache and its policy are retained. Configuration JSON files and launch scripts are not changed.

A frequently selected RAM expert takes the main GPU slot of a less frequently selected expert of
the same layer. The displaced GPU expert takes the former RAM slot. Gate, up and down remain in
their original quantized format. Model files remain closed. The exchange borrows temporary bytes
inside the existing staging buffer and does not allocate another weight arena.

In this mode staging is used for prefill and temporary exchange storage; by default it does not
also keep the old persistent decode copies. Thus the initial number of main GPU slots stays fixed
and can be smaller than the old main slots plus staging copies. Exchange changes their composition,
not their capacity. A layer with no initial GPU slots cannot gain one in this first implementation.

`--no-disk-exchange-cache` (opt-in, needs the main flag) additionally lets the staging bytes keep
disposable copies of RAM experts during decode, the way the plain `--no-disk` staging cache does:
the swap transaction writes the first staging slot only, copies live from the second slot on. An
expert whose copy stands on staging or is on its way there takes no part in an exchange until the
cache lets it go, so the two mechanisms never claim the same expert at once; a completed exchange
leaves a copy of the demoted expert valid, because its RAM twin receives the bytes the permanent
slot had. A prompt batch drops the copies, as in the plain mode. Measured 2026-11-06 with MiMo
V2.6 Flash MOPD Q2_K on an RTX 4090 (`--stage-mib 2048`, warm agent repeats, two runs): decode
27.5-27.7 -> 29.5-29.7 tok/s (+7.1..7.2%), expert hit 44-45% -> 50-52%, exchanges completed
unchanged, none failed; the cache phase's first two requests wait 0.13-0.2 s once for copies
already in flight.

Options (all opt-in through the main flag):

| Option | Default | Meaning |
|---|---:|---|
| `--no-disk-exchange-max N` | 1 | Maximum completed exchanges per eval boundary, 1 through 64 |
| `--no-disk-exchange-cooldown N` | 0 | Evaluated decode rows before either expert may exchange again |
| `--no-disk-exchange-budget-pct F` | 10 | Soft time budget for additional exchanges relative to the preceding eval |
| `--no-disk-exchange-cache` | off | Staging keeps RAM-expert copies during decode next to the exchanges |

The first transaction is always allowed so a cold registration or diagnostic measurement cannot
permanently stop adaptation. Further transactions stop when the budget is consumed. The budget
is a soft bound: one indivisible transaction may exceed it. The default of one exchange bounds
the batch independently of this estimate. The preceding eval measurement covers host compute
and submission, not an exclusive GPU-time measurement.

The policy uses the existing recent-use counts, with a half-life of 128 evaluated decode rows,
and the threshold `recent(hot) >= 2 * recent(cold) + 2`. Profile counts break ties. Like the
existing adaptive cache, evaluated tentative MTP rows contribute to these counts; ownership is
global and does not roll back when speculative token state or a conversation is restored.
The cooldown is configurable; 256 rows is not claimed to be an optimal measured setting.

Readers finish before weights are overwritten. A transaction saves the cold GPU payload to
staging, copies hot RAM weights to that GPU slot, copies the cold payload back to the RAM slot,
then publishes both owners. On an error it restores the original payloads where possible. If
restoration fails the engine refuses further expert evaluation. Registered RAM pages are reused
within a bounded window; removing any registration waits for its DMA users.

`exchange:` diagnostics separate completed/failed/restored transactions, plan/registration/copy/
reader-wait time and rejected candidates. `committed_H2D`, `committed_D2H`, `committed_D2D` count
successful committed payload bytes; rollback and verification traffic are not included in those
three byte counters. Registration counters with `_total` are engine-lifetime counts.

For diagnostics only, `STRATA_VERIFY_EXCHANGE=1` hashes both complete payloads before and after
each exchange. It adds substantial host and device reads and must be off for throughput tests.
`STRATA_TRACE_EXCHANGE=1` reports the expert identities and counts. `STRATA_FREEZE_EXPERTS=1`
disables automatic adaptation in both cache modes.

The native serve protocol exposes `EXCHANGE [fault]` only with `STRATA_EXCHANGE_TEST_MODE=1`.
It forces a same-layer pair and verifies both payloads; fault points -1, 1, 2, 3 exercise rollback.
It is a test hook, not an HTTP endpoint. `poly_exchange_test` checks planning without a model;
`poly_exchange_gpu_test` exercises 1000 in-place exchanges, rollback, payload hashes, tails,
canaries and bounded registration reuse without loading model weights. `poly_exchange_cache_test
MODEL.gguf` runs both mechanisms together on the real engine with `STRATA_VERIFY_EXCHANGE` and
`STRATA_VERIFY_STAGE_CACHE` on: after every 32 decoded tokens it walks every expert and requires
the permanent-slot / host-twin / staging-owner witnesses to agree, and at the end it fails if
either mechanism stayed idle.
