# The CPU share of the experts under a prompt batch, on the same DeepSeek inputs

The same prepared inputs of `../2026-10-21-deepseek-bat-prefill-specials/`, the same executable, the same
configuration, the same opaque copy of the configured expert statistics, the same two full inputs of 50000 tokens
with `resume=0` and answers up to 256 tokens.

**What is measured and why.** The prefill of this workload is GPU-bound: the card is 84.6% busy while the process
uses about one core of the fourteen, and the CPU experts are not computed at all. The knob that decides how much of
the expert work the CPU takes under a batch is `--cpu-share`, and with a batch of 4096 tokens it does nothing:
the speculative copy-ahead (`--stage-ahead`, 1024 by default) is on for a batch this size, and the CPU share is
only used when the copy-ahead is off (`engine.cpp:2902`). So the three variants below turn the copy-ahead off and
give the CPU a share:

| mode | flags | what it is |
|---|---|---|
| `bat-cpu-0` | `--stage-ahead 0 --cpu-share 0.0` | the copy-ahead off, the CPU takes nothing: the GPU does all the experts |
| `bat-cpu-50` | `--stage-ahead 0 --cpu-share 0.5` | the CPU takes half of the uncached routed experts, in parallel with the card |
| `bat-cpu-75` | `--stage-ahead 0 --cpu-share 0.75` | the CPU takes three quarters of them |

The stock `bat-no-disk` of the earlier series is the fourth point of the comparison: the copy-ahead on and the CPU
share never applying. Its numbers are in `../2026-10-21-deepseek-bat-prefill-specials/`.

The decode is a control: `--cpu-share` is used only in the batch path (`moe_rows()`), so the decode of the three
variants must be the same as the decode of the stock mode. A difference there is the machine's, not the flags'.

```powershell
.\.venv\Scripts\python.exe -u bench/results/2026-10-21-deepseek-cpu-share/run.py --validate-prepared
.\.venv\Scripts\python.exe -u bench/results/2026-10-21-deepseek-cpu-share/run.py --run-prepared --modes bat-cpu-0 bat-cpu-50 bat-cpu-75
```

The executable, the configuration, the bat file and the original profile are not changed, and the result files are
created exclusively.
