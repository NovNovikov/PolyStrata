# MiMo-V2.6-Flash (`strata-poly`)

`strata-poly`, PolyStrata's multi-architecture GGUF engine, runs Xiaomi's MiMo-V2.6-Flash (GGUF architecture
`mimo2`) for text: prefill and generation read the original GGUF in place, with no requantizing. The tested file is
`MiMo-V2.6-Flash-MOPD-Q2_K` from ggml-org: 48 layers, 256 routed experts per MoE layer with 8 active per token,
Q2_K gates and ups and MXFP4 downs, 117.7 GB in two files (the first holds the metadata and the tokenizer, the
second every tensor). Vision and audio are not run. The model's MTP head ships as a small separate file
(`mtp-MiMo-V2.6-Flash-MOPD-Q4_0.gguf`, 1.2 GB) and runs as the draft block's guesser through `--draft-model`.

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
engine args: `--no-disk --no-disk-expert-exchange --no-disk-exchange-cache --compact-scratch --coalesce-expert-dma
--rows-vram --max-context 131072 --read-chunk 32768 --prompt-batch 4096 --stage-mib 2048 --expert-swaps 64
--decode-attn-warps auto --window 2`, plus `--model` and `--draft-model` pointing at the two files. The weights go
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

## Decode, measured on 4090 + 128 GB (2026-11-08)

Protocol: one 50 000-token prompt read, 1024 tokens generated at temperature=0 with the draft block on; the same
protocol measured DeepSeek V4 Flash at 19.4-21.3 tokens/s on this machine. The run-to-run drift of the same config
across sessions is about +-9%, so every number below compares phases measured back to back in one session.
Decode went from 9.30 to about 19.5 tokens/s in one day:

- **Decode-attention warps** (`--decode-attn-warps auto`): +90%. The global attend at 50 K read at ~37 GB/s
  useful because each warp walks its keys through a serial online-softmax chain; spreading keys across up to 32
  warps per block raises what the card does usefully to ~70 GB/s.
- **Split-K for single-row calls** (`auto`): rows-1 attends x1.76 (1998 to 1136 us a global layer at 50 K);
  measured no gain at 2-4 rows, so the auto rule applies it to rows 1 only.
- **Draft depth `--window 2`**: +20% (16.27 to 19.53 tokens/s at 50 K, +22% at short context). The shipped nextn
  head accepts 99.4% of first guesses (510/513) and almost nothing deeper - a 4-guess block accepted 0.91 tokens
  a verify and re-read the rejected rows' experts, about 1.9 expert-read rows a token where 1.0 of two-token
  forwards needs 1.0. One guess per forward stops paying for rejects. The answer's token ids were identical
  1024/1024 with and without the deep block: the draft only proposes and the verify enforces the greedy answer,
  so the flag cannot change the text.
- **The hot-expert exchange cache** (2026-11-06, already live on every `--no-disk` model): +7%.

The wall that remains is two pieces, both with a proven nature, and four levers died measuring at them:

- **~32 ms a token of CPU experts** streaming Q2_K from locked RAM at 62-70 GB/s, the machine's DDR5 roofline:
  ~230 picks x 9.50 MiB a token at the 50 K hit rate. Rejected by measurement: rotating hot experts into VRAM
  through the disk-exchange path (the hit rate rose 39% to 47% and CPU time fell 2.5 ms, but the exchange's stream
  fence cost more, net -0.9%, and short contexts lost 15%); leaving low-weight experts out at 3% (`--skip-weight`,
  -7.3 points of draft acceptance for no speed); any deeper draft block, per above.
- **~14 ms a token of attention at 50 K**, the softmax-chain ceiling itself: four launch shapes (warps, split-K,
  compiled batches, one fetch grouped over the 16 GQA rows) were measured and the useful bytes already equal what
  goes through the card. The prefill's tensor-core path reads 16 rows in the same ~28 ms a verify forward spends
  on 2, so it needs acceptance depth the draft does not have, not another launch shape.
- **The queue's wait is the attention, not the router**: a GPU-phase probe (events inside the captured graph)
  puts the router GEMV and the score copies at 0.017 ms a layer, 0.18 ms a token, 0.3% of the wall - which
  measures device-side top-k dead.

## Open

- Expert-level output parity against llama.cpp on long generated text: the kernel, routing and mxfp4 tests are
  green and agents use the model daily, but nobody has diffed a long generation against llama.cpp token-for-token.
- A deeper draft needs a drafter whose acceptance survives position 2; the shipped nextn head's does not
  (measured above), and the engine loads no other draft format.
