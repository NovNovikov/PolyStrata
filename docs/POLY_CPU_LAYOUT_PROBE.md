# Standalone CPU expert layout probe

`poly_cpu_layout_probe` is excluded from the default build. It compares task sizes,
separate gate/up traversal and expert-major traversal of a down-row tile in the
production CPU expert routine. Experimental branches exist only when the standalone
target defines `STRATA_CPU_LAYOUT_PROBE`; there is no engine option or default change.

Build the target with the project's configured CMake build, then run:

```
poly_cpu_layout_probe MODEL.gguf NEW_OUTPUT_DIRECTORY 8
```

The output directory must not exist. The probe copies the first four routed experts
of every layer into RAM and closes the GGUF before checks and timing. It does not
load the inference engine, use the GPU, run a server or read user text/profiles.
The selected bytes and tensor shapes are recorded in `geometry.csv`.

Eight variants share the same weights, ggml dot products, deterministic input and
12-thread worker pool. Five shuffled rounds test 1- and 4-row calls; an additional
stock control finishes each round/row-count block. `timings.csv` records wall and
CPU stage spans, not a predicted end-to-end tokens/s gain. Other machine activity
is not controlled by the probe; assess control drift before accepting a small win.

Before timing, output vectors are checked byte-for-byte for 1/2/4/8 ragged rows,
reused experts, SiLU and GELU, and 1/12 threads. A synthetic GPU-submission callback
must execute exactly once. Timed calls have no callback: they isolate CPU execution,
not GPU/CPU overlap. `validation.txt` is written only after the full series completes.

Expert-major down traversal changes weight-reading order but preserves each output's
expert sum order. Gate/up traversal uses the same independent dot products and
activation expression. No precision change or weight repacking is involved.
