# Correction: compare engine consumption, not whole-card free memory

This analysis supersedes the reserve-based rejection in README.md. It uses
the four preserved synthetic runs, without rerunning an engine or changing
their original logs, summaries or exit statuses. Whole-device free VRAM
cannot identify memory consumed by this engine process. The original benchmark
incorrectly used its minimum as a pass/fail criterion for this comparison.

## Measured consumption

MiB, ordinary adaptive-window pair:

| Measurement | staging 2848 | staging 3360 | Candidate minus control |
|---|---:|---:|---:|
| Explicit device allocations, warm median | 22109.86 | 22107.80 | -2.06 |
| Process Local Usage, warm median | 22845.25 | 22843.25 | -2.00 |
| Process Local Usage, maximum of all recorded samples | 22862.67 | 22861.68 | -1.00 |
| Process Non Local Usage, warm median | 545.125 | 545.125 | 0 |
| Process Total Committed, warm median | 23390.37 | 23388.37 | -2.00 |
| Last ten Local Usage spread | 0.0234 | 0 | |
| Last ten Non Local Usage spread | 11.9375 | 0.8125 | |
| Last ten executable graph counts | 661 | 661 | 0 |

Warm statistics use the last ten identical-request repeats. Local and Non Local
are the process PDH counters read by the existing telemetry; neither is computed
by subtracting whole-card free memory. Non Local is read from the legacy raw
JSON field `spilled`; that field name does not prove eviction of VRAM. Registered
host pages also contribute to host GPU-memory accounting. Total Committed is
reported separately and is not called physical VRAM consumption.

At READY, Non Local Usage is 3316 versus 3828 MiB. The extra 512 MiB coincides
with the explicitly enlarged registered RAM source window (2848 to 3360 MiB).
It is gone from the difference between warm samples: both have median 545.125
MiB. There is no measured increase in the process Local Usage for the candidate.

The fixed-window-1 comparison agrees: both warm medians are 22917.11 MiB Local,
477.125 MiB Non Local and 23394.24 MiB Total Committed. Explicit allocations are
22407.86 versus 22408.25 MiB (+0.39 MiB from expert packing/alignment).

**Larger staging does not increase recorded steady consumption in these runs.**
The exchange is +512 MiB staging for about -514 MiB permanent expert weights in
the ordinary pair. No accumulating allocation or executable-graph growth is
observed in the ten warm repeats. Request-boundary samples do not measure
transient peaks inside a request or prove absence of every possible long-run
leak. Local residency alone would also miss a move into host memory; this report
therefore includes Non Local and Total Committed.

## Benchmark correction

`tools/bench_poly_graph_memory.py` now uses the same consumption analyzer as
this reanalysis, `tools/poly_memory_report.py`. It checks aligned process,
allocation and graph samples, and rejects missing counters rather than treating
them as zero. For ten warm samples it retains the previous 4 MiB Local Usage
tolerance, requires stable explicit allocations (0.05 MiB allowance for printed
rounding) and graph counts, and screens for sustained Non Local/committed growth
by comparing medians of the first and second five samples with a 4 MiB tolerance.
The median screen distinguishes bounded DMA-window fluctuations from sustained
growth; it is a finite-run heuristic, not a general leak detector.

All four preserved runs satisfy these consumption checks. Cache correctness and
token equality results still hold. Existing unsuccessful exit statuses remain
recorded as historical outcomes of the old whole-device criterion. This does
not establish guaranteed free headroom during arbitrary external GPU workloads.

Offline regression checks used copies of the preserved data: replacing every
whole-device free-memory reading with zero leaves consumption results identical;
injected Local, Non Local, committed or explicit-allocation growth is detected;
missing counters and unaligned samples are rejected. Python syntax was checked.
No C++ build, new GPU run or production-config change was made.

Full samples and check outcomes: [consumption-review.json](consumption-review.json).
Timing conclusions remain separate: ordinary +2.95% has a speculative-policy
confound, while fixed-window +4.30% is one noisy pair. The memory-based reason
previously given for rejecting staging 3360 is withdrawn.
