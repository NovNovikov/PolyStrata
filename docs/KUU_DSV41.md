# DeepSeek V4.1 engine: the kuu-Strata port

Nine optimizations and two correctness fixes moved from the donor fork
[`midokino/kuu-Strata-fork`](https://github.com/midokino/kuu-Strata-fork) (verified HEAD `2b584305`, common
ancestor `146df3c4`) into this repository's `src/dsv41` on branch `strata-0.2.12`, on the
`experiment/kuu-dsv41` branch. Every option is off unless its environment variable says otherwise; with all
of them unset the engine reads and computes exactly as the pre-port binary. The donor's speed numbers quoted
below are the donor's, measured on the donor's machine - not measurements of this build.

## What is in, and where from

| Change | Donor commit(s) | Here | Switch (default: off) |
| --- | --- | --- | --- |
| `in_ram()` counts the RAM store's slots (correctness) | `faff292baa` | `a9affb2e` | none (a fix) |
| RAM store slots sized per layer (size classes) | `00cf5dddab` | `255d587b` | `DS4_STORE_UNIFORM=1` restores one class |
| Expert Deferral: a decode layer's CPU-lane part lands later | `8beec7ab3d`, `d8b413e933`, `234c8bffc4`, `e0983ffd7f`, `c9d536c544` | `c24cca3d` | `STRATA_DS4_DEFER=0..4` |
| Router kept in BF16 (half the VRAM, same bits) | `16ba4156b6` | `aed9735c` | `STRATA_DS4_ROUTER_BF16=1` |
| `DSpark` drafter VRAM cap, split over its blocks | `19cbf65602` | `7133880c` | `DS4_MTP_VRAM_GB=<GB>` (unset: old behavior) |
| Lazy DSpark: head rows past the confidence cutoff not computed | `e6a49369fb` (its lazy part) | `2a7e8f94` | `STRATA_DS4_DSPARK_LAZY=1` |
| Engram rows of a prompt read ahead into a pinned ring | `8cbdff9e19`, `3642b3af0b` | `a9323e86` | `STRATA_DS4_EG_RING=n` (n>=2) |
| Next-layer expert read-ahead during this layer's chunks | `b54bdb608c` | `42f71876` | `STRATA_DS4_LM_AHEAD=1` |
| Expert shards' buffered handles closed after the load | `f62520a2aa` | `50a3033c` | `STRATA_DS4_DROP_SHARDS=1` |
| DEFER mode and knobs join the session identity | - (task spec §6) | `98487870` | follows `STRATA_DS4_DEFER` |

Defect fixes carried with the port: the donor's `gemv_bf16` is a no-op under `STRATA_METAL` yet it stored only
BF16 there - on Metal the switch is now inert and the F32 copy kept (`aed9735c`); the donor's Engram ring
left its old pinned slots allocated when the chunk grew - they are now released and taken out of the teardown
list (`a9323e86`).

Not ported, per the task spec: ACE norm skipping, the adaptive deferral time budget (`DEFER_BUDGET`,
`DEFER_US_MAX`), the verification window cut (`WIN_CUT`), draft-row-only aggressive skipping (`DS4_EST`,
`DRAFT_RAM_W`, `DRAFT_DISK_W`), the measured-and-rejected routing switches (`ROUTE_KNORM`, `SKIP_RENORM`,
`E2_*`, `A2_*`), and the other section-5 list. Two of the rejected ones rode inside ported commits'
auto-merged hunks and were removed during review - the tree contains none of them.

## Environment variables this port adds

| Variable | Values | Default | What it does |
| --- | --- | --- | --- |
| `DS4_STORE_UNIFORM` | `0`/`1` | off (classes) | `1`: the old uniform store, one slot size |
| `STRATA_DS4_DEFER` | `0`..`4` | `0` | defer the CPU-lane part: `1` whole, `2` light experts, `3` light into the next route, `4` landing `DEFER_K` layers late |
| `STRATA_DS4_DEFER_W` | float | `0.2` | modes 2/4: under this share of the route an expert is light |
| `STRATA_DS4_DEFER_K` | 1-3 | `1` | mode 4: layers late |
| `STRATA_DS4_DEFER_COLD` | `0`/`1` | `0` | mode 4: light means not-in-RAM |
| `STRATA_DS4_DEFER_EARLY` | `0`/`1` | `0` | apply the deferred part at the next layer's entry |
| `STRATA_DS4_DEFER_MAX` | int | all | cap the trunk layer deferral applies to |
| `STRATA_DS4_ROUTER_BF16` | `0`/`1` | `0` | keep a BF16 router as its own bits (half the VRAM) |
| `DS4_MTP_VRAM_GB` | GB | unset | the drafter's per-block VRAM byte cap |
| `STRATA_DS4_DSPARK_LAZY` | `0`/`1` | `0` | stop drafting rows the confidence cut discards |
| `STRATA_DS4_EG_RING` | int | off | Engram reader ring slots (>= 2) |
| `DS4_LM_AHEAD_Q` | float | `0.5` | read-ahead: the pick-chance to read an expert ahead |
| `DS4_LM_AHEAD_CHUNKS` | int | `4` | read-ahead: the shortest run of chunks that turns it on |
| `STRATA_DS4_LM_AHEAD` | `0`/`1` | `0` | next-layer expert read-ahead |
| `STRATA_DS4_DROP_SHARDS` | `0`/`1` | `0` | close the expert files' buffered handles after the load |

The pre-existing skip knobs (`STRATA_DS4_RAM_SKIP_W`, `STRATA_DS4_DISK_SKIP_W`, `STRATA_DS4_DISK_SKIP_AUTO`,
`STRATA_DS4_PROMPT_SKIP` / its older name `DS4_YOCO`, read first and overriding) are unchanged and combine
with these.

## Launch sets for the user's comparison (starting points, not tuned results)

Baseline - the port's binary with nothing approximate and the old allocator:

```ini
DS4_STORE_UNIFORM=1
```

(and `DS4_MTP_VRAM_GB` unset). For the pre-port behavior exactly, run the pre-port binary: even this set
carries the correctness fixes.

Optimized Exact - no intended change to the model's math (`--experts 6`, `--dspark on` for a DSpark check;
note `DS4_YOCO` wins over `STRATA_DS4_PROMPT_SKIP` if both are set):

```ini
STRATA_DS4_ROUTER_BF16=1
DS4_MTP_VRAM_GB=0.5
DS4_STORE_UNIFORM=0
STRATA_DS4_DSPARK_LAZY=1
STRATA_DS4_EG_RING=8
STRATA_DS4_LM_AHEAD=1
STRATA_DS4_DROP_SHARDS=1
STRATA_DS4_RAM_SKIP_W=0
STRATA_DS4_DISK_SKIP_W=0
STRATA_DS4_DISK_SKIP_AUTO=0
STRATA_DS4_DEFER=0
STRATA_DS4_PROMPT_SKIP=1
```

Optimized Approximate (the donor's kuu fast reference - changes results; never default):

```ini
DS4_MTP_VRAM_GB=0.5
DS4_STORE_UNIFORM=0
STRATA_DS4_DISK_SKIP_W=0.26
STRATA_DS4_RAM_SKIP_W=0.10
STRATA_DS4_DEFER=3
STRATA_DS4_DEFER_W=0.20
STRATA_DS4_PROMPT_SKIP=2
STRATA_DS4_DROP_SHARDS=1
STRATA_DS4_DSPARK_LAZY=1
STRATA_DS4_LM_AHEAD=1
STRATA_DS4_ROUTER_BF16=1
STRATA_DS4_EG_RING=8
```

plus the donor's `--experts 5 --dspark on --dspark-conf 0.65`. Do not copy machine-specific flags
(`--reserve-mib`) without re-checking them. The MTP cap and ring size are the donor's starting numbers, not
tuned for the 4090.

## Building and running the tests

```sh
cmake -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON && cmake --build build --target strata-dsv41
# host tests (no GPU, no model): cl/cl- equivalents work too; here through CMake:
cmake --build build --target ds4_vram_plan_test ds4_draft_cut_test   # plus store_classes_test/ds4_io_test compiled directly:
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/store_classes_test.cpp && store_classes_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/ds4_io_test.cpp        && ds4_io_test
# device tests (EXCLUDE_FROM_ALL, need a GPU):
cmake --build build --target ds4_router_bf16_test ds4_route_guard_test && ./ds4_router_bf16_test
# the pre-existing dsv41 host tests (unchanged, run as regression):
cmake --build build --target ds4_slot_test ds4_marks_test ds4_ring_link_test ds4_evt_test \
    ds4_experts_rule_test ds4_path_policy_test ds4_pick_rule_test
```

## What was actually run, and with what

All on the porting machine: RTX 4090 24 GB (sm_89), i5-13600KF, 128 GB DDR5, Windows 11, CUDA 12.9.
strata-dsv41 compiles and links clean after every group.

| Run | Result |
| --- | --- |
| `ds4_router_bf16_test` (device) | 19 checks, 0 failures: BF16 rows and the F32 copy give bit-identical logits at the router's real 256x7168 (T=1 and 16) and other shapes; `route()` picks identical |
| `store_classes_test` | 127 checks, 0 failures; same 50 MB budget, 24 experts x 3 layers differing ~4x: the class store holds 72 experts before its first eviction, the uniform one 48 |
| `ds4_vram_plan_test` | 19 checks, 0 failures (drafter split, cap compliance through every stage, freed room to the trunk, zero/min/unlimited budgets, no-cap behavior) |
| `ds4_draft_cut_test` | 12 checks, 0 failures (cutoff boundaries inclusive, cumulative rule, window cap, rounds back to back) |
| `ds4_io_test` | 27 checks, 0 failures (buffered/par/unbuffered/direct agree aligned and unaligned; direct survives the buffered closes; closed handle and past-EOF fail loudly) |
| `ds4_route_guard_test` (device, regression on the touched kernels) | all ok |
| `ds4_slot_test`, `ds4_marks_test`, `ds4_ring_link_test`, `ds4_evt_test`, `ds4_experts_rule_test`, `ds4_path_policy_test`, `ds4_pick_rule_test` | all pass (session slot/meta, marks, ring-link, events, rules) |

## Not run, and why

- Any end-to-end inference, prefill/decode timing, KL or quality measurement: excluded by the task spec -
  the model was never loaded, so the donor's percentages are not restated as results of this build.
- Engram ring wraparound timing, LM_AHEAD hit rates under demand, DEFER overlap quality: the code paths are
  inside a prompt run and need the loaded model; reviewed in place (slot-reuse guard waits on the H2D event;
  demand reads queue first; the deferred part is dropped at a window boundary).
- CUDA events and H2D ordering of the ring were verified by reading, not running; a model-free harness would
  be a parallel implementation, which the task spec rules out.
- HIP/Metal builds: this machine is CUDA; the dsv41 path is CUDA-only in this tree (and the Metal hazard for
  BF16 was made inert). A HIP build of the touched files happens wherever `STRATA_BUILD_DSV41` is on.

## Open questions for the model check

- The pre-existing approximate knobs (`RAM_SKIP_W`, `DISK_SKIP_AUTO`, `PROMPT_SKIP=2`, `--experts`) are not in
  the session fingerprint - predates the port; DEFER is now (adding the others would change existing users'
  identity strings, so it wants its own decision).
- `DS4_MTP_VRAM_GB` and ring sizes for the 4090 with the user's pack: the donor's numbers are starting points.
- `DSPARK_LAZY` falls back to eager when the head group spans devices, a markov split runs, or a draft dump is
  active - whether the user's geometry hits that, the first `--dspark on` run will show.
