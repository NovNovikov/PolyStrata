# RAM consumed in the previous staging comparisons

These are the preserved row-layout runs of 2026-10-20, not new measurements
of the six-lane deployment. Windows has about 127.84 GiB visible physical RAM.
DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf, context capacity 131072, batch 4096,
read chunk 32768, rows in VRAM, six checkpoints and disk-session settings enabled.
Inputs were 10000 tokens, with edited/continued requests; this is not a 128k-input
RAM headroom measurement. Window-1 runs include explicit SAVE/RESTORE and B/A
switching; ordinary-window runs have fewer requests and no explicit roundtrip.

All values except staging are GiB (2^30 bytes). Process RSS is physical working
set consumption at recorded request boundaries. Its maximum is a sampled maximum,
not a continuous transient peak. Private commit is reported separately and must
not be subtracted from physical RAM capacity.

| Run / window | Staging MiB | Owned weights + host buffers at load | Process warm RSS median | Maximum sampled process RSS | Maximum sampled private commit | Minimum sampled system available RAM |
|---|---:|---:|---:|---:|---:|---:|
| control-one-clean / 1 | 2848 | 112.981 | 114.105 | 114.136 | 136.833 | 3.227 |
| plus512-one / 1 | 3360 | 113.480 | 114.573 | 114.604 | 137.329 | 2.768 |
| control-default / adaptive up to 8 | 2848 | 113.281 | 114.350 | 114.353 | 136.948 | 2.683 |
| plus512-default / adaptive up to 8 | 3360 | 113.782 | 114.876 | 114.877 | 137.475 | 2.784 |

The ordinary pair's owned weights/buffers rise by 0.502 GiB, warm process RSS by
0.527 GiB, and maximum sampled RSS by 0.523 GiB. The fixed-window pair shows
0.500 GiB greater owned allocation and 0.468 GiB greater sampled maximum RSS.
Larger GPU staging displaces permanent GPU experts into RAM, increasing host
weights accordingly. The registered RAM source window is part of those resident
weights; its 512 MiB increase is not another 512 MiB host allocation to add twice.

System available RAM is a separate contextual measurement including all other
programs and the OS. In the ordinary pair it is slightly higher in the larger
staging run despite the engine consuming more RAM. This is not RAM saved by larger
staging: background memory usage differed. Likewise, total RAM minus process RSS
is not the amount available to a new allocation while other programs are running.

The observed available physical headroom during these requests was roughly
2.7-3.2 GiB. It is a description of those samples, not a hard Windows allocation
limit or a guarantee for longer contexts, other applications, or temporary
checkpoint serialization peaks. Existing allocate-and-lock behavior is unchanged.
The source tests used row-layout; the six-lane deployment's full no-disk RAM use
has not been inferred as a new measurement from these values.

[ram-staging-comparison.json](ram-staging-comparison.json) records precise values
and SHA256 hashes of the source results and synthetic diagnostic logs. Original
artifacts under `../2026-10-20-poly-stage-cache-split/` were read without changes.
