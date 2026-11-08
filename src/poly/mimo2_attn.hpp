// src/poly/mimo2_attn.hpp - the attention kernels MiMo needs and kernels::attend() cannot give it.
//
// MiMo-V2.6-Flash ("mimo2") reads heads of 192 key values and writes heads of 128 value values, and its queries,
// keys and values come out of one fused projection.  kernels::attend() indexes both caches with one dimension and
// head_norm_rope() norms what it turns, so neither fits: these kernels do the same jobs with separate key and
// value widths, no norm, and a learned sink per query head in the softmax's sum.
//
// The math follows llama.cpp's mimo2 graph (src/models/mimo2.cpp): a token at position p attends the rows
// 0..p of a global layer or the last `window` of a window layer (a key is hidden when p - key_pos >= window),
// the scores are q.k / sqrt(dim_k), and a window layer's softmax carries its head's sink logit - the sink weighs
// no value, it only takes weight from the rows.
#pragma once

#include <cstdint>

namespace strata::poly {

/// One launch over a call's fused-qkv rows: every token's row of `row` floats splits into its q (n_head heads
/// of dim_k), its k (kv_heads heads of dim_k) and its v (kv_heads heads of dim_v); q and k leave with ggml's
/// rotary embedding on their first n_rot values in the NeoX pairing (value i with value i + n_rot / 2, pair i
/// turning by position * theta_scale^i), v leaves as it came - no norm anywhere.  src: rows * row floats;
/// q: rows * n_head * dim_k; k and v: rows * kv_heads * dim_k and rows * kv_heads * dim_v.  The first token's
/// position is `pos`, or what the card holds at `at` when a recorded launch replays.
void mimo2_rope_split(const float* src, int row, int rows, float* q, float* k, float* v, int dim_k, int dim_v,
                      int n_head, int kv_heads, int n_rot, float theta_scale, int64_t pos, void* stream,
                      const long long* at = nullptr);

/// Causal attention over the two caches, the tokens from position `pos`: token t attends its own rows and the
/// ones before, with `window` only the last `window` of them, and with `ring` the sequence's row p lies at cache
/// row p % ring.  q: n_head * dim_k a token; kcache: kv_heads * dim_k halves a row; vcache: kv_heads * dim_v
/// halves a row; out: n_head * dim_v a token.  `scale` multiplies the products.  `sinks`, when given, is
/// n_head floats: each head's softmax sum carries its sink logit.  With `at` the first token's position is on
/// the card and the launch is sized for `room` rows, as kernels::attend()'s recorded launches are.
/// dim_k and dim_v are multiples of 32, dim_k of no more than 256/32 values a lane.
void mimo2_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* out, int dim_k, int dim_v, int n_head, int kv_heads, float scale, void* stream,
                  int window = 0, int ring = 0, const long long* at = nullptr, int room = 0,
                  const float* sinks = nullptr);

/// How many warps a block of the decode attend gets: "" for the kernel's own four, "auto" for the count that
/// fills the card (about 4096 warps divided by the call's blocks), or 4, 8, 16, 32.  Set once from the command
/// line, before any thread launches; STRATA_MIMO_ATTN_WARPS, when the environment carries it, answers instead,
/// so a run can measure both ways without a rebuild.  Anything unrecognised leaves the four in place.
void mimo2_attend_warps(const char* setting);

}  // namespace strata::poly
