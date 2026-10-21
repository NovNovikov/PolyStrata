# Gemma, the direct modes with the 3825-id prompt (a repeat of 2026-10-20)

2026-10-21, RTX 4090 24 GiB, i5-13600KF, 128 GiB RAM, Windows. This is the repeat of the 2026-10-20 run of
the two direct modes, with the same 3825 ids (the prompt rendered without `parse_special=True`, so the leading
`[gMASK]` was three byte pieces instead of the one special token). The run went into the 2026-10-20 folder by
mistake; the original files of that run were restored from git, and this copy keeps the repeat.

| | clean upstream | the fork |
|---|---:|---:|
| median of the two warm | 353.76 tok/s | 352.18 tok/s |
| prompt tokens (the engine's count) | 3825 | 3825 |

The engine's counters on the last answer: the upstream `generated=512 accounted_tokens=512 wall_ms=1443.892
cpu_expert_ms=0.000 route_wait_ms=0.000 uploads=0 drafts_offered=336 drafts_accepted=328 gpu_experts=124800
cpu_experts=0`; the fork `generated=512 accounted_tokens=511 wall_ms=1426.767 ... drafts_offered=336
drafts_accepted=328 gpu_experts=124560 cpu_experts=0`.

These counters (`336` drafts, `124800` experts) are the ones of a prompt that is one token longer than the
server's; the run with the server's own prompt is `2026-10-21-gemma-three-modes-aligned/`.

The prompt's ids: `L:\Temp\opencode\gemma-ids.json` (rendered by `L:\Temp\opencode\gemma_ids.py`).
