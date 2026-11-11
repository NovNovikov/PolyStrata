# The same DeepSeek comparison, the modes in the reverse order

The prepared inputs of `../2026-10-21-deepseek-bat-prefill-specials/`, unchanged, and the same three modes
in the reverse order: **bat-mmap, bat-exchange, bat-no-disk**. The first series ran them as
bat-no-disk, bat-exchange, bat-mmap.

One series does not separate a difference between the modes from a drift of the machine over the run, and
the first series put the slowest mode first and the fastest last. Running the order backwards puts the
fastest first: if the machine drifts (the page cache of the model file, the drive, the thermals, another
job), the two orders show it in opposite directions, and the modes can be compared on equal ground.

The prepared inputs are the same bytes: the same `public-inputs.json`, the same opaque copy of the
configured expert statistics, the same configuration and the same executable. The runner checks the
sha256 of the configuration, the server encoder and the template before it starts, and refuses to run if
they changed.

```powershell
.\.venv\Scripts\python.exe -u bench/results/2026-10-21-deepseek-bat-prefill-specials-reverse/run.py --validate-prepared
.\.venv\Scripts\python.exe -u bench/results/2026-10-21-deepseek-bat-prefill-specials-reverse/run.py --run-prepared --modes bat-mmap bat-exchange bat-no-disk
```

**One confound to state in the report:** the first series' mmap mode read 129.87 GiB of the model file
from the disk at the end of the run, and the operating system's file cache may still hold part of it when
this series starts with mmap. The second mmap prefill of the first series still read 6.25 GiB, so the cache
does not hold the whole file; but the first prefill of this series may start with more of it cached than
the first series' did. The two orders are therefore not perfectly symmetric in cache state, and the mmap
first prefill of this series must be read with that in mind.

Everything else is as in the handoff: the same parameters from `strata-dsv4-q3_k_m.json`, no change to
the executable, the configuration, the bat file or the original profile, both inputs of 50000 tokens with
`resume=0`, answers up to 256 tokens, and the result files created exclusively.
