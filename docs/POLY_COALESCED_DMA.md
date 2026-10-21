# Bounded expert DMA for no-disk prefill

`--coalesce-expert-dma` is opt-in and requires `--no-disk`. Without the flag the
registration policy and three-copy staging path are unchanged.

For batched MoE, the engine prepares a page-rounded span containing the layer's
RAM experts before submitting its attention head. It does this only when the
span fits the existing `ResidentDma` budget. Staging reuses a covering registered
window for smaller routed subsets rather than replacing it.

If a layer's whole span does not fit (including scattered host owners after an
exchange), registration stays at staging time. Selected stage sources are merged
into one span only if that span fits; otherwise only their exact pages are
registered. The limit, source arena, staging VRAM allocation, expert selection,
CPU share, precision and row geometry do not change. Registering gaps does map
more existing RAM pages and can change driver overhead; it is not a new RAM
weight copy or additional weight H2D traffic.

When gate and up are adjacent in resident RAM, a single H2D transfers both to
their existing adjacent staging destinations. Down remains a separate transfer;
padding and the zero tail are unchanged. Nonadjacent gate/up retain separate
transfers. Source-reader event fences and exception cleanup remain in place.
The exchange mode retains its registration eviction policy and transaction fences.

Enable `STRATA_TRACE_TRANSFERS=1` for per-request API bytes/times and
`dma-before-head`, `dma-stage-covered-skip`, `dma-layer-span-over-budget` records.
`tools/bench_poly_transfer_audit.py --coalesce-expert-dma --run ...` adds the flag
to the existing integrated synthetic sequence. It runs only the changed variant,
does not launch a baseline, and refuses to overwrite an earlier experiment.

`poly_no_disk_test` checks rounded covers, exact-budget fallback and source bounds.
`poly_resident_dma_probe 16 8 1` checks covered-subset reuse, unchanged registration
on invalid input, two outstanding reader streams, partial registration rollback,
cleanup and byte equality of merged/original staging including padding and tail.
These small synthetic checks do not load a model. Hardware measurements and
backend validation limits belong in the corresponding experiment report.
