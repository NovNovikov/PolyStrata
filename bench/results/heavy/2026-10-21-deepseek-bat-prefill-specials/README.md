# Prepared DeepSeek comparison through server prompt encoding

Status: **prepared offline; no model run started**. The user explicitly declined another run in
this chat and requested this package for Minimax. The previous series is retained in
`../2026-10-21-deepseek-bat-prefill/` and explicitly marked as invalid ordinary-chat encoding.

`run.py --prepare-only` called the actual `Service.encode_prompt` with the configured tokenizer and
template, the server's BOS/EOS template parameters, its default thinking, and literal control-token
protection. No model, HTTP server, user session, user log or private prompt was loaded for preparation.
Both public requests contain exactly 50000 IDs. The first IDs are BOS `0`, User `128803`; the last
two IDs are Assistant `128804`, thinking-open `128821`. There is exactly one User and one Assistant
control token. The server's cached and full encoding produce identical token streams.

`public-inputs.json` records the actual synthetic messages, IDs, source hashes and the configuration
and server-encoder hashes. `input-validation.json` records the checks. This fixes both the absent
control parsing and the previously missing BOS template parameter.

The prepared launch is:

```powershell
.\.venv\Scripts\python.exe -u bench/results/2026-10-21-deepseek-bat-prefill-specials/run.py --run-prepared
```

It will run bat no-disk, bat plus opt-in exchange (default maximum 1), and bat minus no-disk/mmap,
in that order. Each receives both full inputs and requests up to 256 generated tokens. The same
initial opaque expert-profile copy is used for every mode; session stores are isolated. The launch
arguments are checked against the JSON, allowing only these mode flags, diagnostic dump-buffers and
test profile/session paths. The original profile, JSON, bat and exe are not written.

No freezing, verification hash loop, fault injection, graph disabling, reserve override, batching
override or build is part of this series. Measurements include both full prompt time and the interval
after RESUME before final PP, separating parking. Every input must report resume=0 and read=50000.
Existing results are neither deleted nor overwritten; each artifact is created exclusively.

Use `--validate-prepared` for an offline check without loading a model. The runner validates the
prepared inputs before launching any mode. `--run-prepared --modes bat-no-disk bat-mmap` selects
just the requested pair if the human narrows the scope. Existing results are created exclusively;
an unexpectedly short answer ends the suite gracefully after preserving its result.

See `HANDOFF_MINIMAX.md` for the exact task, corrections, commands and acceptance conditions.
