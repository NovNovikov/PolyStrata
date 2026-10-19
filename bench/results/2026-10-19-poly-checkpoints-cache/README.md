# DeepSeek: text checkpoints with the full adaptive expert staging cache

Windows, RTX 4090 24 GiB, 128 GiB RAM, DeepSeek V4 Flash Q3_K_M. All prompts and session files in
this test are synthetic. No private logs, prompts or existing user sessions were read.

## Launch configuration

The local ignored `strata-dsv4-q3_k_m.json` now explicitly sets `--expert-swaps 64` and uses
`L:\AI_pictures_generate\PolyStrata-data\packs\dsv4-q3_k_m\sessions` instead of the GLM session directory.
The existing `--stage-mib 2848` and `--prompt-cache 6` are retained and tested enabled together.
`--conversation-cache-mib 0` disables the RAM parking pool, not the active conversation's checkpoints
or automatic disk parking through `--session-dir`. No additional expert allocation was introduced.

Other options are unchanged: context 131072, batch 4096, read chunk 32768, rows in VRAM, compact scratch,
VRAM reserve 256 MiB and loading headroom 256 MiB. Unlike earlier memory benchmarks, this test uses the
launcher's default lookup/draft options: it does not inject --no-draft or --lookup 0. Adaptive experts are
not frozen, and STRATA_VERIFY_STAGE_CACHE=1 checks GPU payloads against immutable RAM originals after requests.

The binary is unchanged from the RAM admission fix: SHA256
`a91c6783e180947fe3c15ed1e5766d4a1cb2601b1a6de66fcca112cb8bb480bf`.
Tested config SHA256: `eb0cde6a44416d6d87ac9ba1843558fcfe565478b37ed69b38122ef75a3a50f0`.
The test replaces only the expert profile and session store paths with its own fresh synthetic-test paths;
it never scans the configured user store. Large session payloads remain under build-poly-cuda12/checkpoints-cache-20261019.

## First process

The model loaded in 206.83 s, with 578 permanent GPU experts and 255 staging-cache slots.
The 2848 MiB staging allocation fits 255 slots of 11668800 bytes: 2837.70 MiB of expert payload capacity.
The remaining 10.30 MiB cannot hold another full slot. The table reports whole-request averages, including
cache warm-up, rather than a comparison against the earlier lookup-disabled throughput tests.

| Request | Prompt tokens | Reused | Read | Generated | Decode tok/s | Expert hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, initial | 10000 | 0 | 10000 | 512 | 29.40 | 50.77% |
| A, continuation | 10576 | 10511 | 65 | 64 | 23.39 | 26.94% |
| B, unrelated prompt | 10000 | 0 | 10000 | 64 | 26.32 | 44.98% |
| A, changed after token 5000 | 7048 | 4096 | 2952 | 512 | 35.99 | 52.06% |
| A, after explicit RESTORE | 10576 | 10575 | 1 | 64 | 28.75 | 33.27% |

Semantic checkpoint positions supplied over the engine protocol were 2048/4096/6144/8192, plus the
conversation's end markers. A continuation marker was also supplied at 10511. This tests the engine's
checkpoint mechanism; the server's rendered-template boundaries are covered by the three separate
LastTurn unit tests, not by an HTTP chat replay.

The GPU cache reached 255/255 slots (2837.70 MiB), including after the interior-checkpoint prefill.
Five payload verifications passed with 255, 251, 255, 255 and 255 published slots respectively; zero
kernel tails also matched. A batched prefill borrows staging and invalidates those disposable GPU copies,
then decode refills them. This is expected when continuing or switching conversations, not persistent
serialization of the expert cache into the text checkpoint. Permanent GPU experts and RAM sources remain.

Automatic parking wrote separate session files for A and B. The interior edit restored the checkpoint at
4096 and read only 2952 tokens. Explicit SAVE wrote 603114484 bytes at position 10511. After RESTORE and
another SAVE, both files had SHA256
`5d1f68a63b27dd5047442c2502414dce448f5f055814d035b062fea8fd332dca`.
This is serialized-file equality; SAVE can reuse imported host blobs, so it alone is not a fresh readback
of every restored GPU byte. The second-process isolated continuation exercises the restored device state.

The two requests with the identical continuation produced exactly the same 64 natural token IDs.
Their resume positions differ: after RESTORE the conversation cache found an even deeper parked match
at 10575, and used it in preference to the explicitly restored position 10511.

## Test harness correction

The first driver expected exactly 10511 for that last resume and stopped on an assertion, after all five
requests and the file round-trip had completed. `initial-runner.txt` retains the failure. It was a test
expectation error: the cache is allowed to choose the deepest matching parked checkpoint. The corrected
driver accepts that result, checks the identical output IDs, and continues from these retained synthetic
artifacts using --resume instead of repeating a successful first-phase load. Its second phase uses a
new suffix to force selection of the explicitly restored state, avoiding this ambiguity.

## Second process: restart and isolated explicit restore

The next process loaded in 200.40 s and scanned only the test's own automatic session directory.
It resumed A at 9999/10000, read one token and produced 64 tokens at 24.51 tok/s. These 64 IDs exactly
match the first 64 output IDs of the original request A. It then explicitly restored the 10511-token
snapshot and received a different continuation suffix, preventing reuse of the deeper parked continuation.
This selected position 10511 as required, read 65 tokens and produced 64 at 21.42 tok/s.
Both requests passed GPU cache payload/tail verification (252 and 249 published slots). Cache capacity
remained 255 slots, and occupancy reached 2804.32 MiB in these short post-restart requests.

Both engines exited with code 0; no strata-poly processes remained after the successful final phase.
The initial driver itself exited 1 solely for the over-strict assertion documented above; the resumed
driver exited 0 and printed CHECKPOINT_CACHE_INTEGRATION_PASSED. The retained summary explicitly records
that this result continued from the initial run. All seven generation requests completed.

## Memory observations and limits

The engine's explicit device allocations stayed at 22423.61 MiB: dense 7758.37, permanent experts 6319.61,
KV 860, staging 2848, rows 2048 and other 2589.63. These totals exclude CUDA/driver allocations.
In the first process, whole-device free VRAM fell from 493.34 MiB after loading to 0.00 MiB at the end.
All requests and GPU byte checks completed, but this is not evidence that the runtime reserve stayed
available or that there was no migration. No VRAM planning policy was changed by this configuration task.
The first process was not sampled through PDH, so no per-process attribution of that drop is claimed.

The second process records raw per-process Local/Non Local/Shared/Total Committed counters through the
existing telemetry helper. Its legacy key `spilled` means the raw Non Local Usage counter; it must not
be interpreted as isolated evicted VRAM bytes. Registered host RAM is also deliberately GPU-accessible.
Windows accounts for system RAM used by the GPU as shared memory; see Microsoft's description:
https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/

| Second-process PDH snapshot, MiB | Local | Non Local | Shared | Total Committed |
| --- | ---: | ---: | ---: | ---: |
| READY, DMA source window 2848 MiB | 22864.67 | 3316.00 | 3316.00 | 26180.67 |
| After automatic resume/generation | 22995.16 | 500.56 | 500.56 | 23495.72 |
| After isolated explicit restore/generation | 23159.24 | 536.75 | 536.75 | 23695.99 |

Non Local and Shared were equal in these samples; neither isolates migrated device allocations from
the GPU-accessible host sources. The second process ended with 159.02 MiB of whole-device free VRAM,
also below the configured 256 MiB reserve. The functional test passes; preservation of the runtime
VRAM reserve does not pass this observation. The cause and prevention of that runtime headroom loss
were not resolved in this configuration-only task. In particular, CUDA graph costs have not been
isolated by a matching --no-graphs run, and external GPU usage is not attributed to the engine.

Functional coverage is Windows/CUDA, this model/configuration and the engine line protocol. It is not a
concurrent-client soak, full HTTP test or a guarantee against runtime VRAM pressure from longer histories
and changing desktop GPU usage. `semantic-boundaries-tests.txt` has three passing server unit tests;
`conversations-tests.txt` has both passing generic conversation/checkpoint suites. No C++ rebuild was needed.
