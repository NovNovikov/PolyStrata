# DeepSeek V4.1 engine: the kuu-Strata port

Nine optimizations and two correctness fixes moved from the donor fork
[`midokino/kuu-Strata-fork`](https://github.com/midokino/kuu-Strata-fork) (verified HEAD `2b584305`, common
ancestor `146df3c4`) into this repository's `src/dsv41` on branch `strata-0.2.12`, on the
`experiment/kuu-dsv41` branch. Every option is off unless its environment variable says otherwise; with all
of them unset the engine reads and computes exactly as the pre-port binary. The donor's speed numbers quoted
below are the donor's, measured on the donor's machine - not measurements of this build.

After the port a hostile review (GPT-assisted, 8 items) read every group against its defect surface; the
seven code findings are each fixed in their own commit, with model-free tests over the production code.
See "Review round" below. A second review (15 items) read the port and the first round's fixes against
memory ordering, failure paths and what the logs actually prove; its results are in "Second review round".

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

## Review round (one commit per finding, model-free tests)

| # | Commit | The defect, and what the fix does | Test |
| --- | --- | --- | --- |
| 1 | `889bdc19` | The CPU lane decided deferral from env + a window-count check of its own, and a window cut mid-request could leave a deferred part pending into the next forward. The verdict now travels with the request: `publish_kernel` stamps `Req.defer` under the seq fence, the lane splits per `lane_splits(r.defer, mode)`, and the four forward entries bump an epoch that drops pending parts from an older forward. Pure logic extracted to `ds4_defer_policy.hpp` | `ds4_defer_policy_test` (21) |
| 2 | `b4e27d3c` | Session files written by an older engine could carry knobs the new one never parsed and be trusted anyway. The fingerprint is `strata-ds4-session-5`: session-4 files are refused, and the identity string carries the parsed effective values (DEFER mode, and cold for every splitting mode, not only 4) | `ds4_identity_knobs_test` (29) |
| 3 | `e2d01b6b` | A size class could starve on a full budget: all-free slots of its size with the room held by another class. `ExpertStore::rebalance()` recasts the coldest fully-reclaimable chunk (all-free, or ready-unpinned-not-visible; never loading or pinned) to the starved class; the `rebalances` counter reports it | `store_classes_test` (153) |
| 4 | `f248c64f` | `STRATA_DS4_DROP_SHARDS` closed file handles while queued IO still held raw fds - reads after a close hit foreign descriptors. Jobs now hold `FdRef` slots resolved at dispatch, and a swap drain-closes the old handle only after every acquired read finishes | `store_close_test` (14) |
| 5 | `5adc9bb7` | A failed Engram read that the window needed twice copied its destination slot's stale bytes into the duplicate (zeroing came after the copies) and warned only under Metal. Failed rows zero first, everywhere, counted as `eg_bad`; the ring's slot book and the batch finisher (`EgRing`, `eg_complete`) moved to `ds4_engram_io.hpp` | `engram_ring_test` (40) |
| 6 | `2ca4f887` | `LM_AHEAD` could retarget its second buffer while the old target's guesses were still in flight (a skipped or RAM-served layer never adopts), tearing slots and counts; a failed guess ended the request. A reuse drains the side's in-flight reads first (`PfSide::drain`), and a failed guess re-reads on demand like any miss (`ds4_lm_ahead.hpp`) | `lm_ahead_test` (13) |
| 7 | `80ac8352` | The deferral late-add and the lazy drafter's batch plan were only checkable with the model loaded. Both moved to headers the device and host share (`ds4_defer_math.hpp`; `dspark_lazy_batches` in `ds4_draft_cut.hpp`): the late add is now checked bit-for-bit against a CPU reference for the on-time add, and the drafter's computed-row plan against its bounds over every cut position | `defer_math_test` (1792), `ds4_draft_cut_test` (26) |

## Second review round (items 1-15, one commit per finding)

A second hostile review read the port and the first round's fixes again - this time against memory
ordering, failure paths, and what the logs actually prove. Status of every item; each fix is its own
commit, each test runs without a model.

| # | Status | Commit | Finding, and what the fix does | Test |
| --- | --- | --- | --- | --- |
| 1 | FIXED | `301ea4a3` | `ExpertStore::find()` hands out a slot whose bytes could be recast the next microsecond: callers read them without a pin, and a full slot whose read had died left the caller waiting. `acquire_ready()` now returns only READY slots, pinned for the caller (`unpin` gives it back); a miss falls back to a demand read through the caller's own `ReadAt` | `store_classes_test` (166) |
| 2 | FIXED | `4ff378d6` | A slot whose fill had failed (dead entry) was still counted live, chosen for eviction, and reported as room: dead entries are now skipped by accounting, eviction and the free-list alike | `store_classes_test` (166), `store_close_test` (20) |
| 3 | FIXED | `45b59e2a` | `FdRef` swap could lose a wakeup (reader between check and wait) and `replace()` waited for readers of the NEW generation: the wait is generation-scoped (a ring of four counters indexed by generation), `waiting_` guards it under seq_cst, acquire/release carry the generation | `store_close_test` (20, section 4: old generation waited out, a new holder not waited, a 500-round lost-wakeup watchdog) |
| 4 | FIXED | `327d1de5` | An Engram row whose read failed was served as zeros silently, the ring published refused rows, and a retry re-read a batch that was already complete. Failed rows are sticky (`EgBatch::bad_n`), the producer fails the ring item instead of publishing it, `eg_complete` retries once through a hook, refused rows fail the REQUEST, and a ring short of pinned room runs narrower or falls back - with prints | `engram_ring_test` (49) |
| 5 | FIXED | `88cca029` | Two miss paths could both demand-rerun the same failed read-ahead slot (two writers, one destination). `PfSide::recover()` elects one re-reader through the exchange the old path already used; the others wait on it, and the item carries its own side so an adoption mid-flight cannot misroute it | `lm_ahead_test` (23; the test drives the production recovery function itself) |
| 6 | FIXED | `741199eb` | The identity printed knob floats with `%g`: locale-dependent, six digits, two different weights could print the same string. `fp_txt()` writes the shortest round-trip text with `std::to_chars` (locale-free), and the format went to `strata-ds4-session-6` - session-5 files cannot prove their floats and are refused | `ds4_identity_knobs_test` (34) |
| 7 | FIXED | `ab62a9fb` | An Engram row cache or ring out of memory could throw past the load (the process died) and reader threads that failed to start left a join that never returned. The cache catches and freezes its capacity, `EgReader::start` reports how many readers came up (load fails by name at zero), `DS4_ENGRAM_THREADS` is clamped to 256 | `engram_ring_test` (49) |
| 8 | FIXED | `596ae05c` | The DSpark auto-fit rule refused the drafter BEFORE the drafter's VRAM cap was consulted - and the cap only exists while the drafter is loaded, so setting `DS4_MTP_VRAM_GB` (the knob for exactly such a tight box) did nothing. With the cap set the auto rule stands down and says so; kept, refused and inert cap each print their witness | engine build (ordering is load logic; refusal text unchanged byte-for-byte) |
| 9 | VERIFIED NOT APPLICABLE | - | Claim: MTP experts stay resident past the cap. Traced instead: `plan_group_vram` respects the cap in every stage (`min(budget, mtp_cap)` in stage 1, MTP skipped in stage 3), `prof` covers `(nl+n_mtp)*n_exp` rows, tail-lending and the recent-slot carve-out exclude MTP blocks, `n_host = n_exp - n_vram` holds under the cap, and the lend-back demotion touches trunk only | `ds4_vram_plan_test` (19) |
| 10 | FIXED | `a1ddb8af` | `rc_loop`'s recent-slot fills copied the GPU straight from the file's mapping while the store could recast it (and read a null mapping under `DROP_SHARDS`): every MXFP4 fill now stages the blob through pinned memory first - the store's copy under a pin the length of one memcpy, the file otherwise. The lane's copies take the store's copy too, and `lane_warm` counts a store-held blob as the RAM it is (and never samples a null mapping) | engine build; mailbox protocol and pins unchanged |
| 11 | VERIFIED, documented here | - | Deferral and autotuning against the session identity - see "Identity of the moving parts" below | `ds4_identity_knobs_test` (34) |
| 12 | FIXED | `80aaa607` | The lazy DSpark counters grew in a member nothing read: `Stats` publishes `lazy_rounds`/`lazy_rows` (counted only while the mode is on - the default path pays nothing) and the path prints its activation once | engine build |
| 13 | ADDED | - | "Approximate modes" below: what each knob changes and where its activation shows | - |
| 14 | FIXED | `af16a157` | `kuu_features_test.bat` graded survival, not activation: every run now greps its log for the feature's witness line (PASS/SKIP/FAIL with the reason), and the two silent modes - `STRATA_DS4_ROUTER_BF16` and a load banner arming `STRATA_DS4_DEFER` - print one | the bat is NOT run here (real pack, exclusive GPU); witness strings taken verbatim from the engine |
| 15 | SWEEP | this commit | full host-test re-run and engine build after all of the above; numbers in "What was actually run" | |

## Third review round (items 1-15, P1 blockers first)

A third hostile review read the second round's fixes and the paths around them - handle lifetimes,
abandoned runs, service threads that die, buffers overwritten mid-DMA. Seven P1 findings came first
(items 1-7): each could corrupt weights, hang a request forever, or kill a later request on a dead
thread. Stage B (8, 9, 12, 13, 14) and Stage C (10, 11, 15) follow the same numbering as before. One
fix per item, every test model-free; the acceptance line was: no silent weight corruption, no infinite
wait on a dead service thread, engine links, all tests pass.

| # | Status | Commit | Finding, and what the fix does | Test |
| --- | --- | --- | --- | --- |
| 1 | FIXED | `1b43fb5e` | `FdRef::acquire` read the handle and counted itself as a reader in two steps: a `replace()` between them returned a CLOSED fd, and the reader was counted against a generation that never used it. Handle and reader count now live in one refcounted `Ctl` block - acquire takes both under a short mutex, `replace()` still never waits on a reader, and a retired fd closes at the LAST reference, after every in-flight read. Teardown audit: the current block's last reference is deliberately NOT dropped at `~FdRef` - Pack owns the handles in the table, and a destructor that closed them (or left a reader's release to) would re-open the double-close class this item exists to end; what remains is one small block per file table, reclaimed at process exit | `store_close_test` (22: a close/replace loop racing readers that read through the handle they hold) |
| 2 | FIXED | `ef5b8d76`, `1204b919` | The recent-slot fill (`rc_stage`) rewrote and freed a pinned staging buffer while an earlier copy from it could still be queued: two experts' bytes in one slot. The production class is extracted (`ds4_rc_stage.hpp`): every member stream carries notes, a rewrite DRAINS the previous fill first, a poisoned stage is leaked rather than freed under a live DMA, a failed fill DECLINES (the RAM lane serves) and a pinned buffer that will not pin falls back to heap - never onto a recastable store slot. HOTFIX: the copies `record()` notes were enqueued BEFORE it ran, so ANY failure inside record (bad device, event creation, event record) means a live DMA with no note - every failure path now poisons the stage for good: no further notes, `drain()` false forever (refills decline), the blob leaked at teardown, never freed under that copy | `ds4_rc_stage_test` (device, new: barriers force the two fills to overlap and check every slot's bytes; hotfix: record on an absent device and on a destroyed stream's handle both poison, and the poisoned stage leaves the copy's bytes UNTOUCHED - 31 checks) |
| 3 | FIXED | `8508bf9b` | An Engram ring that fell back for pinned room left `eg_ring=false` while the ring PRODUCER started over the unreset book (`slot_of()` on n==0) and the consumer `get()`-ed a future nobody started. Requested (`eg_want`) and effective (`eg_ring`) are decided apart, half-made ring resources are freed on every fallback (out of `host_allocs` too), a thread that cannot start is a mode change not a `std::terminate`, and exactly one producer - the one the effective mode names - starts | `engram_ring_test` (62, fallback shapes) |
| 4 | FIXED | `5b3a9e77`, `9193c617` | A lane thread's death was reported like a recoverable failure, and `flow_pump`'s `break` could strand the service loop: later requests queued to a thread that no longer existed and hung. The failure state splits (`ds4_lane_gate.hpp` `LaneFail`): REPORT clears per run, DEATH is sticky forever (`clear_run` re-marks, never reopens), every forward/feed/draft entry refuses on death, and a refused `forward` rolls its position and token book back instead of advancing over a lane that will not compute. HOTFIX: the request that RACES the death still hung its kernels - the device pipelines layers, so the next requests were published while the thread left the loop. `serve_lane` now enters TERMINAL mode instead of breaking: it stays and answers every still-published request with the same failed zeros (`mail_terminal_step`), until teardown's quit - and, per the last review, the drain answers the DEFER late ring too: a `STRATA_DS4_DEFER=2/3/4` request the lane split waits on `resps` AND `resps_late`, so the step asks `lane_splits` and publishes the zeroed late answer as well (the one-ring-deeper hang closed). And a run whose lane died mid-flight commits nothing: `step_runs`/`verify_multi` refuse at the entry, the multi window fails on an outstanding note, and `checkpoint`/`mark`/`slot_save` make no restore point of a zero-poisoned state | `ds4_lane_gate_test` (36: a killed service thread fails the NEXT request rather than hanging it; hotfix scenario: a device thread four requests deep, healthy answers before the death and failed zeros after, the requests published AHEAD of it all released, quit ends the drain; the DEFER scenarios drive the 5-argument step for modes 2/3/4 - split requests answered on both rings from the drain, unsplit ones never answered on the late one) |
| 5 | FIXED | `ada97f1a`, `124995b6` | `StoreHold::release()` keyed on the caller's `ok` flags: a partial read left its slots pinned forever (the budget never saw them back), and the hot batch resolved experts by raw slot instead of the hold's `dmd()` - the lease a dead read still owed. Ownership is the slot list alone (`sl[i] >= 0` IS the pin), and the hot path resolves through the hold. HOTFIX (the mirror defect): a read that FAILED while holders stood kept its pin count but lost its state guard - `finish()` made it `kFree`, which `take_slot` hands out without looking at pins, and `begin()` pins a LOADING slot by `where_` - so two holds of one loading expert shared a slot a third fill was given mid-hold, and their releases cut THAT owner's pin short. A failed read with holders now lands in `kFailed`: dead (never findable, handed out or recast) but held, and the last pin - all four decrements run through one `drop_pin` - retires it to the free pool | `store_classes_test` (190, scenarios 8-9: an abandoned hold returns its pins whole; a file truncated mid-geometry, two holders of one failed slot, and a third fill that is never given the held one) |
| 6 | FIXED | `5da95c8b` | After a successful `PfSide::recover`, the election's loser-winner race could pick a SECOND writer for the just-restored slot. The winner re-checks through `recover_owned`: a slot that arrived while you were elected is taken as a HIT, never rewritten | `lm_ahead_test` (30, the production exchange driven by two threads) |
| 7 | FIXED | `dbb3b181` | Every way out of a flow run moved on while its CPU reads were in flight: `flow_release_pins` unpinned slots a running read copies out of, `run_dev_chunks` dropped `items` - the counters the completions decrement - under them, and `~Impl` `cudaFreeHost`ed the staging the queued jobs write into (the `IoQueue` destructor joins its threads only AFTER that body). The wait `halt_end`/`lm_end` had is now inside `flow_release_pins` (before the unpin, keyed on `c.flow && c.io`), and `~Impl` resets the chunks' queues before any pinned arena is freed | `ds4_io_test` (32, abandon-then-drain: 200 rounds, counters zero, bytes right, >100 rounds caught jobs in flight) |
| 8 | FIXED | `d731d20f` | A speculative read whose layer ARRIVED was a demand the pump waits for - and it sat in the low queue behind later guesses while the workers take high work first. `IoQueue::promote(left)` moves the queued part-jobs of that owner to the high queue (a part running is untouched, a part taken is off the queue, repeat calls are idempotent); `flow_pump` promotes the item it stands at | `ds4_io_test` (32, one worker, five guesses: the promoted one provably lands FIRST) |
| 9 | FIXED | `97929ffa` | Speculative LM_AHEAD passes nobody adopted still read over the link: reuse drained for them, and `lm_end` waited for a run's own queued guesses. `PfSide` grows a per-slot cancel flag - posted jobs carry it, the queue drops a flagged job UNREAD - set by `discard_all` before the reuse drain and before `lm_end`'s drains; `begin()` resets it, which is safe against the drain-before-reuse invariant | `lm_ahead_test` (30, scenario 6: discarded queued guess lands failed with the sentinel slot untouched) |
| 10 | FIXED (diagnostics) / DEFERRED (reserve) | `df43ef64` | A size class that starves was invisible: one total `rebalances` counter and a silent `-1`. Per-class `starved`/`relieved`/`denied` counts, one-time notices naming the slot MB, a `class_report()`, and the load banner's class table. A per-class RESERVE is deferred: the recast already rebalances the shared budget between classes and no measurement says a fixed floor pays - building it would sit unused as often as not | `store_classes_test` (174: fresh store reports clean; the starve/recast/refuse block leaves all three counters non-zero) |
| 11 | FIXED | `e2b05c1e` | The preload RANKING was byte-aware (`count per blob byte`) but the `DS4_STORE_BG` head/tail SPLIT divided 48 GB by LAYER 0's blob - off by the blob ratio on a mixed pack, so the "before the first token" pile was not 48 GB; the background report made the same assumption. The cut accumulates each listed expert's own blob; the report sums the tail's own bytes | engine build (the split needs profile + loaded geometry; uniform packs - where both formulas agree within one expert - unchanged) |
| 12 | FIXED | `31838938` | "No mapped view" is a fact about a FILE: `DROP_SHARDS` keeps mixed shards' mappings, yet one dropped file set the global `views=false` and every starved expert of every layer took the demand rescue - a second blob of IO plus an anon alloc - where its own matrices are still mapped. The rescue asks `Pack::at` per expert (null exactly for dropped files); the global flag is gone, bin packs keep their old answers via the `pk.gguf` guard | engine build (rescue lives with a loaded pack; the per-file primitives `at`/`read_at` are covered by `ds4_io_test` and `store_close_test`) |
| 13 | FIXED | `ea79e43f` | The ring pinned without a ceiling (`cudaHostAlloc` says yes past comfort on Windows) and its events had no teardown owner. A byte budget (4000 MB default, `STRATA_DS4_EG_RING_MB`) clamps the slot count BEFORE pinning, trims a ring a grown chunk pushed over it, and falls under two slots; `~Impl` destroys the ring's and the two-buffer path's events where it already frees the flow's queues | `engram_ring_test` (62: the clamp arithmetic - request bounds budget and vice versa, floors, zero/negative shapes, both engine compositions) |
| 14 | FIXED | `fa152cbb` | `kuu_features_test.bat`'s witness greps had no text: the script handed the engine's stderr to a temporary it UNLINKED, so each run's log carried six prints, no engine line. Every engine stderr line is now teed live into the .bat's log AND kept whole in `KUU_ENGINE_LOG`; grading tightened (a 15-token DONE, a QUIT that hangs, a non-zero exit all fail) and a failure prints the log's tail | `short_prompt_check_test.py` (new: seven mock-engine scenarios grade the checker - exit codes, witness passthrough, log completeness, post-QUIT tail) |
| 15 | FIXED | `509bad51` | `lazy_rounds`/`lazy_rows` lived only in the `/stats` JSON; the console showed a lazy run and a silently-eager run identically. STAT grows `lazy_r=`/`lazy_rows=` deltas, the serve summary names the lazy share of its drafts, the dspark harness prints its line | engine build |

Items 16 and 17 of the task spec were constraints, not findings: the accepted first- and second-round
fixes stand (the sweep below re-ran them all), and no new allocator, scheduler or engine was introduced -
every change is inside the machinery the port already had.

## Identity of the moving parts (review 11)

Everything that changes what a layer computes is in the session identity as the ENGINE resolved it - and
what about the parts that move by themselves?

- **Deferral is pinned by configuration.** Mode, `DEFER_W`, `DEFER_K`, `DEFER_COLD`, `DEFER_EARLY`,
  `DEFER_MAX` all enter the fingerprint (`ds4_identity_knobs.hpp`), so a session saved under `DEFER=3`
  refuses to resume anywhere the deferral differs; `ds4_identity_knobs_test` (34 checks) covers the
  emission and the parsing.
- **A deferred part cannot cross a session boundary.** It belongs to the forward that made it: the epoch
  bumped at every forward entry drops parts from an older one, the landing layer must exist inside the
  same forward, and saves happen after a token - a finished forward - so a session file never holds a
  half-deferred residual.
- **The autotuned skip (`DISK_SKIP_AUTO`) pins its settings, not its measurements.** Auto on/off, window,
  settle and budget are in the fingerprint; the decision to skip is a function of this machine's page
  cache, and identity pins rules, not machine state - the same contract the plain skip knobs always had
  (`in_ram()` reads the page table too). The mode is approximate and says so below.
- **The pre-port autotuners pick transports, not numbers.** `tune_lane` and `calibrate_paths`/`pick_path`
  choose thread layouts and which path runs an expert; both lanes round a low-bit expert's activations
  alike (ggml's Q8_K, unless `DS4_IQ_Q8K=0`), and `DS4_LANE_CHECK=1` measures one expert of each class
  through both at load and prints how many outputs are bit-identical. The port changed none of that.

## Approximate modes: what each knob changes, and where its activation shows (review 13)

None of these is default; the fast reference in "Launch sets" is marked approximate for exactly this
list. "Identity" means the value (as the engine resolved it) is in the session fingerprint: same
fingerprint - same rules. The rules themselves may consult machine state (the skip knobs ask the page
cache), so equal fingerprints promise equal settings, not equal numbers across machines.

| Knob | Changes the answers? | Identity | Activation witness in the log / stats |
| --- | --- | --- | --- |
| `STRATA_DS4_DEFER=1..4` | yes - the CPU-lane part lands a later layer (mode 4 also remaps stream weights) | yes (all six knobs) | `STRATA_DS4_DEFER=N is armed` at load; the arithmetic is tested bit-for-bit against the on-time add (`defer_math_test` 1792) |
| `STRATA_DS4_DISK_SKIP_W` (+`_FROM`) | yes - low-weight cold experts are not read at all | yes | `skipped` in stats |
| `STRATA_DS4_DISK_SKIP_AUTO` | yes, while on - on/off decided by the measured cold share | settings yes; runtime state cannot be | `DISK_SKIP auto: off after ...` / `on again for request ...` |
| `STRATA_DS4_RAM_SKIP_W` | yes - the same rule for every routed expert | yes | `skipped` in stats |
| `STRATA_DS4_PROMPT_SKIP=1/2` (older name `DS4_YOCO`, read first and overriding) | yes - prompts skip layers | yes (mode and keep, as the geometry permits) | per-chunk counts under `DS4_CHUNK_LOG=1` only |
| `--experts N` | yes - fewer routed experts | yes (`experts_used`) | the load's expert count line |
| `STRATA_DS4_DSPARK_LAZY` | no - rows past the confidence cut are never verified; the cut rule and row plan are tested (`ds4_draft_cut_test` 26) | not needed (drafter output is verified) | `DSpark lazy: ...` once; `lazy_rounds`, `lazy_rows` in stats (review 12) |
| `DS4_MTP_VRAM_GB` | no - drafter residency only | not needed | the per-block `DS4_MTP_VRAM_GB: drafter ...` line prints to every log; `... no drafter is loaded` says when the cap is inert (review 8) |
| `STRATA_DS4_ROUTER_BF16` | no - the logits are bit-identical to the F32 copy (`ds4_router_bf16_test` 19) | not needed | `routers kept in BF16` (or the named inert case) |
| `STRATA_DS4_EG_RING`, `STRATA_DS4_LM_AHEAD`, `STRATA_DS4_DROP_SHARDS`, `DS4_STORE_UNIFORM` | no - reads and placement only, bytes tested (`engram_ring_test` 49, `lm_ahead_test` 23, `ds4_io_test` 27, `store_classes_test` 166) | not needed | the ring's pinned line, the read-ahead's counts line, the shard-drop line, the store-class banner |

## Environment variables this port adds

| Variable | Values | Default | What it does |
| --- | --- | --- | --- |
| `DS4_STORE_UNIFORM` | `0`/`1` | off (classes) | `1`: the old uniform store, one slot size |
| `STRATA_DS4_DEFER` | `0`..`4` | `0` | defer the CPU-lane part: `1` whole, `2` light experts, `3` light into the next route, `4` landing `DEFER_K` layers late |
| `STRATA_DS4_DEFER_W` | float | `0.2` | modes 2/4: under this share of the route an expert is light |
| `STRATA_DS4_DEFER_K` | 1-3 | `1` | mode 4: layers late |
| `STRATA_DS4_DEFER_COLD` | `0`/`1` | `0` | splitting modes (2, 3, 4): light means not-in-RAM, weights aside |
| `STRATA_DS4_DEFER_EARLY` | `0`/`1` | `0` | apply the deferred part at the next layer's entry |
| `STRATA_DS4_DEFER_MAX` | int | all | cap the trunk layer deferral applies to |
| `STRATA_DS4_ROUTER_BF16` | `0`/`1` | `0` | keep a BF16 router as its own bits (half the VRAM) |
| `DS4_MTP_VRAM_GB` | GB | unset | the drafter's per-block VRAM byte cap |
| `STRATA_DS4_DSPARK_LAZY` | `0`/`1` | `0` | stop drafting rows the confidence cut discards |
| `STRATA_DS4_EG_RING` | int | off | Engram reader ring slots (>= 2) |
| `STRATA_DS4_EG_RING_MB` | MB | `4000` | the ring's total pinned-byte ceiling: over it the ring runs narrower, under two slots it is off (round three) |
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
cmake --build build --target ds4_vram_plan_test ds4_draft_cut_test   # plus the host tests compiled directly:
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/store_classes_test.cpp && store_classes_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/ds4_io_test.cpp        && ds4_io_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/store_close_test.cpp   && store_close_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/engram_ring_test.cpp   && engram_ring_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/lm_ahead_test.cpp      && lm_ahead_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/defer_math_test.cpp    && defer_math_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/ds4_defer_policy_test.cpp
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/ds4_identity_knobs_test
#   cl /std:c++17 /I src/dsv41 /I include tests/dsv41/ds4_lane_gate_test.cpp  && ds4_lane_gate_test
#   python tests/dsv41/short_prompt_check_test.py          (mock engines - no model, no GPU)
# device tests (EXCLUDE_FROM_ALL, need a GPU):
cmake --build build --target ds4_router_bf16_test ds4_route_guard_test ds4_rc_stage_test && ./ds4_router_bf16_test
# the pre-existing dsv41 host tests (unchanged, run as regression):
cmake --build build --target ds4_slot_test ds4_marks_test ds4_ring_link_test ds4_evt_test \
    ds4_experts_rule_test ds4_path_policy_test ds4_pick_rule_test
```

## What was actually run, and with what

All on the porting machine: RTX 4090 24 GB (sm_89), i5-13600KF, 128 GB DDR5, Windows 11, CUDA 12.9.
strata-dsv41 compiles and links clean after every group and after every review fix; the counts below are
the sweep re-run after the final second-review-round build.

| Run | Result |
| --- | --- |
| `ds4_router_bf16_test` (device) | 19 checks, 0 failures: BF16 rows and the F32 copy give bit-identical logits at the router's real 256x7168 (T=1 and 16) and other shapes; `route()` picks identical |
| `store_classes_test` | 166 checks, 0 failures: size classes over one 50 MB budget (72 experts vs the uniform store's 48 before first eviction), the starvation relief - a class takes a cold chunk from another and never a loading or pinned one - and round two's lease discipline: a ready slot is only handed out pinned, a demand read fills around a miss, and dead entries leave the counts, the eviction scan and the free list |
| `store_close_test` (new, review 4) | 20 checks, 0 failures: handles swapped under readers in flight, queued jobs resolved at dispatch, drain-close ordering; round two's section 4: a swap waits its OLD generation's readers out, never a holder of the new handle, and a 500-round lost-wakeup watchdog over the release/swap race |
| `ds4_vram_plan_test` | 19 checks, 0 failures (drafter split, cap compliance through every stage, freed room to the trunk, zero/min/unlimited budgets, no-cap behavior) |
| `ds4_draft_cut_test` | 26 checks, 0 failures (cutoff boundaries inclusive, cumulative rule, window cap, rounds back to back; the review's lazy batch plan: shape over a grid of bounds, rows paid per cut position, a cut at draft 0 computes zero head rows) |
| `ds4_io_test` | 27 checks, 0 failures (buffered/par/unbuffered/direct agree aligned and unaligned; direct survives the buffered closes; closed handle and past-EOF fail loudly) |
| `ds4_defer_policy_test` (new, review 1) | 21 checks, 0 failures: the deferral verdict and the lane-split rule as pure functions |
| `ds4_identity_knobs_test` (new, review 2) | 34 checks, 0 failures: session-6 identity strings, parsed effective values, cold for every splitting mode, and the `fp_txt` round-trip rule (the shortest text that reads back as the same float) |
| `engram_ring_test` (new, review 5) | 49 checks, 0 failures: ring slot book walked by two threads (no refill under a live copy), row cache clock, and `eg_complete` over synthetic files - a failed read zeros its row AND its duplicate, failures never cached, `stop()` drains a submitted batch; round two's: sticky bad rows retried through the completion hook, a refused row failing its request not its lane, a producer that fails the ring item instead of publishing it, and the pinned-room fallback |
| `lm_ahead_test` (new, review 6) | 23 checks, 0 failures: the read-ahead pass book (target/skip/reset), real IO reads showing a reuse begins only after the drain and the slot then holds the new expert whole, a failed speculative slot landing-and-failed with the demand retry restoring its bytes; round two's: the production `PfSide::recover` electing one re-reader with the rest waiting on it |
| `defer_math_test` (new, review 7) | 1792 checks, 0 failures: the device late-add's exact functions vs a CPU reference for the on-time add, bit-exact on exact dyadic values - plain landing, one and two mixings, mode 4's per-layer remap, mode 3/4 coefficient agreement, device-shaped strided adds; hc 1/2/4/8, 64 seeds |
| `ds4_route_guard_test` (device, regression on the touched kernels) | all ok |
| `ds4_slot_test`, `ds4_marks_test`, `ds4_ring_link_test`, `ds4_evt_test`, `ds4_experts_rule_test`, `ds4_path_policy_test`, `ds4_pick_rule_test` | all pass (session slot/meta, marks, ring-link, events, rules) |

Second review round, re-run after the final build (2026-11-22): `store_classes_test` 166, `store_close_test`
20, `engram_ring_test` 49, `lm_ahead_test` 23, `ds4_identity_knobs_test` 34, `ds4_io_test` 27,
`defer_math_test` 1792, `ds4_draft_cut_test` 26 - 0 failures each; the CUDA engine built clean from these
commits (sm_89). One store_close timing assertion (the swap's wait measured against a 150 ms holder)
missed its 400 ms bound once while eight builds shared the machine and passed 20/0 on a quiet re-run; its
lost-wakeup watchdog is a named-timeout check, not a bound. `ds4_defer_policy_test` (21),
`ds4_vram_plan_test` (19) and the device `ds4_router_bf16_test` (19) were not rebuilt: the code they
compile against is unchanged since they passed.

Third review round, re-run after the final build (2026-11-22), same machine: `store_classes_test` 174,
`store_close_test` 22, `ds4_io_test` 32, `lm_ahead_test` 30, `ds4_lane_gate_test` 18 (new),
`engram_ring_test` 62, `defer_math_test` 1792, `ds4_draft_cut_test` 26, `ds4_identity_knobs_test` 34 -
0 failures each; the device `ds4_rc_stage_test` 18 (new), and `short_prompt_check_test.py` all-good
(mock engine, no model). The CUDA engine (`strata-dsv41.exe`, sm_89) built and linked clean at every
commit of the round. `kuu_features_test.bat` itself was NOT run - it loads the real pack on an exclusive
GPU, out of the task spec; what round three fixed in it is the machinery its verdicts stand on, tested
against mock engines instead.

Hotfix pass on the same round (three P1 leftovers found after the sweep: `1204b919`, `124995b6`,
`9193c617`, plus the late-ring follow-up below), re-run after the final build (2026-11-22), same
machine: `ds4_rc_stage_test` 31, `store_classes_test` 190, `store_close_test` 22, `ds4_io_test` 32,
`ds4_lane_gate_test` 36 - 0 failures each; the untouched suites stand at their round-three numbers
above (the headers they compile against did not change), and `strata-dsv41.exe` (sm_89) built and
linked clean at the final tree. The live-pack feature sweep that followed found one more hole in the
terminal drain - split DEFER requests wait on `resps_late` too, and the first shape released only
`resps` - fixed as above and covered by scenarios for modes 2, 3 and 4.

## Live-pack feature sweep, and what it fixed (2026-11-22)

First time any of these switches ran against a real pack: the owner ran `tests/dsv41/kuu_features_test.bat`
(10 engine loads) on the IQ1_S pack `deepseek-v4.1-1017-ud-iq1-s`, sm_89 24 GB + 128 GB RAM, on the hotfixed
build. Baseline and `DEFER=3` passed clean (10-24 tok/s generating); the router, uniform-store and lazy
switches SKIPped with their own named reasons, as designed. Five FAILs came out of it, and they were one
crash and three silences - none of them a hotfix regression, all of them first-contact bugs of paths the
host tests could not reach:

- **The cap crash (real)**: on this pack the auto rule refuses DSpark (103 GB of routed experts against
  24 GB of GPU), and baseline lives; `DS4_MTP_VRAM_GB` stands that rule down (second review 8.1), the drafter
  begins to load, and its Markov step demands `mtp.2.markov_head.embed.weight` in BF16 and the head in
  F16/BF16 - an IQ1_S pack keeps them quantized, so the loader returned false DEEP in placement and the whole
  engine died at init. A documented switch killed the run. The fix is a refusal, not a new quant reader:
  where the auto fit rule already refuses on size, the loader's Markov condition is now asked up front and
  the pack's own dtypes go into a named line - `DSpark not loaded: the drafter's Markov head is not in the
  bits this loader reads (pack: ... = IQ1_S, wanted BF16 ...)` - and the run goes on without the drafter,
  exactly as it does under the auto refusal. Reading a quantized drafter is a separate feature with its own
  acceptance-rate validation; it is not part of this fix.
- **Three named silences**: `EG_RING` and `LM_AHEAD` live inside the chunked pass, which this box's smoke
  prompts (40 tokens, threshold 96) never reach, and `DROP_SHARDS` steps only over a demand store, which a
  copy-everything-into-RAM machine (`--mmap-experts auto`, "no demand store") never builds. All three were
  legitimately idle - and printed nothing, so the sweep could only call the silence FAIL. Each now names
  itself once at the load's path banner: `Engram ring armed: the chunked pass owns it - prompts under N
  tokens run in windows ...` (a table-less pack says THAT instead), `next-layer read-ahead armed: ...`,
  `STRATA_DS4_DROP_SHARDS armed but idle: the CPU lane's experts came into RAM at load ...`. The prints
  change no feature code; the sweep's witness checks gained the matching SKIP branches, and `all_exact`
  accepts a named reason as a trace, as it always did for the router's `the switch does nothing`.

Their real tests stay what this smoke never was: a prompt past the flow threshold exercises the ring and
the read-ahead, and `--mmap-experts on` with a limited `DS4_STORE_GB` gives `DROP_SHARDS` a store to drop
through. Speed claims still wait for a repeated, one-process benchmark - eight loads of the same 103 GB at
~0.38 GB/s is why the sweep spends ~45 minutes before it measures anything.

The measurement that answers them is `tests/dsv41/kuu_perf_test.bat`: one run per parameter in its own
process (the windows group `baseline defer3 eg_ring lm_ahead all_exact`, the streaming group
`store_base ring_mmap ahead_mmap uniform drop_shards` on `--mmap-experts on` + `DS4_STORE_GB=32`, each
compared with its own control - the `_mmap` pair because the ring and the read-ahead work only where
the chunked pass streams experts from the files), a load sized like the user's own benches (the qwen
A/B benches take a 32,675-token prompt with GEN 512; the mimo bench `--gen 1024/2048`): a 32,768-token
prompt, KUUPERF_WARM warm-up GENs then KUUPERF_GENS measured GENs of 512 (defaults 2 and 5), every run
one RESULT line - cold load, prefill,
decode tok/s per GEN, and the activation witness, so no number is read without knowing the switch
reached its code. `kuu_perf_check.py` is the runner; its mock test (`kuu_perf_check_test.py`, run green 2026-11-22) grades the verdict table without a model.
A/B/A, from the first full sweep the same day: decode drifts run after run (the store group's median
walked 9 -> 31 tok/s across the sweep in launch order alone), so each group runs its control first
AND last (`baseline ... baseline2`, `store_base ... store_base2`) - the two controls' own gap is the
noise estimate, and a switch counts as an effect only against their average; measured GENs went 3 -> 5
and warm GENs 1 -> 2 for the same reason. The RESULT's `prefill` is the first GEN's cold read - the
first version reported the last GEN's 0.1 s of `32767 reused`, a reporting bug, fixed. Twelve pack
loads want an idle machine: the owner runs it, as always.

The A/B/A ran the same evening (2026-11-22, the owner's box: one GPU, 128 GB RAM, pack
deepseek-v4.1-1017-ud-iq1-s; logs %TEMP%\kuu-perf-logs). Two questions, two answers:

- **DEFER=3, the usual copy-to-RAM geometry: no effect.** The two controls came out equal to the
  last digit - mean decode over the five measured GENs 26.8 and 26.8 tok/s, cold prefill 74.2 and
  83.8 s - and defer3 landed inside that band at 25.5 tok/s (prefill 73.6 s). The morning's +15 %,
  measured without the second control, was noise. The flag does nothing at this geometry on this
  hardware and stays off by default.
- **DROP_SHARDS, mmap + 32 GB store: real, and only for streaming reads.** Cold prefill 137.2 s
  against controls of 188.7 and 205.4 - **minus 30 %** against two controls that agree within 4 %.
  The mechanism is the switch's own: shards already shipped into the RAM store are unmapped and
  their buffered handles closed, so the streaming path stops double-buffering what the cache holds.
  Decode stayed undecidable even with five GENs - the controls themselves diverged (32.4 against
  17.7 tok/s) while the experts kept migrating mid-run.
- Neither switch does anything for this box's daily geometry: the engine already copies the 103 GB
  of experts into 128 GB of RAM, where a 32K prompt reads in 74 s; the minus 30 % belongs to
  machines that must stream from disk.

## Not run, and why

- Any end-to-end inference, prefill/decode timing, KL or quality measurement: excluded by the task spec -
  no timing was claimed from the sweep's single-shot GENs (warm-up alone moved baseline 10 -> 19 tok/s),
  so the donor's percentages are not restated as results of this build.
- CUDA event waits, H2D record order, and the actual overlap quality of the ring and the read-ahead: these
  need the device's timeline. Their decisions (which slot may be refilled, when a buffer may be retargeted,
  when a part must die) are now the tested classes; the waits around them were verified by reading.
- `DSPARK_LAZY` sampled-draft token identity through the RNG path: the cut rule and the row plan are tested;
  the sampler itself needs a device and runs identical code to the eager loop.
- HIP/Metal builds: this machine is CUDA; the dsv41 path is CUDA-only in this tree (and the Metal hazard for
  BF16 was made inert). A HIP build of the touched files happens wherever `STRATA_BUILD_DSV41` is on.

## Open questions for the model check

- ~~The pre-existing approximate knobs (`RAM_SKIP_W`, `DISK_SKIP_AUTO`, `PROMPT_SKIP=2`, `--experts`) are
  not in the session fingerprint~~ - settled in the second review: they joined it (all as the engine
  resolved them, `ds4_identity_knobs.hpp`), and the format went to `strata-ds4-session-6` because a
  session-5 file cannot prove its knob floats. Session-5 files are refused, not trusted; see
  "Approximate modes".
- `DS4_MTP_VRAM_GB` and ring sizes for the 4090 with the user's pack: the donor's numbers are starting points.
- `DSPARK_LAZY` falls back to eager when the head group spans devices, a markov split runs, or a draft dump is
  active - whether the user's geometry hits that, the first `--dspark on` run will show: with a drafter
  loaded, `lazy_rounds` staying at 0 names the fallback - since the third round the STAT line and the serve
  summary print it beside the drafts, no JSON open needed (review 12, round-3 item 15).
