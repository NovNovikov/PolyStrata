# MiMo-V2.6-Flash (`strata-poly`)

`strata-poly`, PolyStrata's multi-architecture GGUF engine, runs Xiaomi's MiMo-V2.6-Flash (GGUF architecture
`mimo2`) for text: prefill and generation read the original GGUF in place, with no requantizing. The tested file is
`MiMo-V2.6-Flash-MOPD-Q2_K` from ggml-org: 48 layers, 256 routed experts per MoE layer with 8 active per token,
Q2_K gates and ups and MXFP4 downs, 117.7 GB in two files (the first holds the metadata and the tokenizer, the
second every tensor). Vision, audio and the MTP head are not run; text only.

## What the architecture needs

- **Attention whose two sides differ.** Every layer reads one fused projection: 64 query heads of 192 key values
  over 4 (global layers) or 8 (window layers) key heads, whose keys are 192 values but values 128. Queries, keys
  and values leave turned by the position (NeoX rotary over 64 of the 192 values, base 1e7 on global layers and
  1e4 on window ones) and with no per-head norm. `kernels::attend()` indexes both caches with one dimension and
  `head_norm_rope()` always norms, so MiMo gets its own kernels, `mimo2_rope_split` and `mimo2_attend`, which keep
  the key and value widths apart and pass the values through unnormed.
- **39 of the 48 layers attend a window of 128 tokens** from a ring and add a learned sink logit a head to their
  softmax: the sink takes weight from the rows and weighs no value. The global layers (0, 5, 11, 17, 23, 29, 35,
  41, 47) attend every token so far.
- **The attention's output joins the residual times the GGUF's `attention.value_scale`** (0.707), which the engine
  applies after the output projection.
- **Routing is a sigmoid plus a selection bias, and the weights are the bare sigmoids made to sum to one**:
  pick the 8 largest of `sigmoid(logit) + bias`, weigh by `sigmoid / their sum`. No shared expert; the first
  layer's FFN is dense (16384 wide) and read where the expert tensors are absent.

## Running it

Start the server with the run config: `python serve/server.py --config strata-mimo-mopd-q2k.json`. The config's
args: `--no-disk --no-disk-vram-margin-mib 256 --max-context 131072 --rows-vram --coalesce-expert-dma
--read-chunk 16384 --prompt-batch 2048 --stage-mib 2048 --expert-swaps 64 --vram-reserve-mib 512`. The weights go
to locked RAM and fixed VRAM and the model files close: the dense part and the attention fit in 24 GB of VRAM,
the 108 GB of routed experts live in system RAM and the card's expert cache keeps the busy ones, as on every
`--no-disk` model. The tokenizer pack is `PolyStrata-data/packs/mimo-mopd-q2k/tokenizer` (written by
`tools/strata_tokenizer.py` from shard 1; its pre-tokenizer is `qwen2`). Tool calls are the template's XML form
(`<tool_call><function=name><parameter=p>value`), which the server's `function` call format parses.

## What was checked before the first run

`poly_mimo2_attn_test` compares both new kernels against a separate scalar double reference: the split and the
turn at several positions, causal edges at the window's 127/128/129, the model's window of 128, the ring's wrap,
GQA for 16 and 8 queries per key head, sinks in and out, and the position held on the card for recorded launches.
All 30 cases pass on an RTX 4090. `poly_route_test` checks the card's routing against the hosts' for gate 1 with
256 experts and 8 chosen: 0 picks differ, weights within 2.9e-07. `poly_mxfp4_test`: 0 failures. The GGUF's
geometry reads through `serve/models.py` as `mimo2 / Q2_K / available`.

## Open

Nothing above ran the model itself: expert parity against llama.cpp, real prefill and decode speed, and long
conversations on 4090 + 128 GB are measured the first time the model runs for real. There is no expert profile
for MiMo yet; `--expert-profile` takes one when a trace exists.
