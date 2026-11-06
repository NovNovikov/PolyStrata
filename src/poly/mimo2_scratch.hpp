// src/poly/mimo2_scratch.hpp - MiMo scratch offsets in floats. Kernels and their arithmetic are unchanged;
// a phase reuses an address only after the previous phase's last read on st_.
//
// The layer's whole attention phase fits over the batched experts' rows the engine placed at
// [base, moe_end): lin(wqkv) fills qkv_, rope_split reads every qkv_ row while it writes q_/k_/v_ (one
// kernel, so the rows keep their distance), the two store()s drain k_/v_ into the caches, mimo2_attend
// reads q_ and the caches and writes att_ (att_ may therefore not overlap q_; it overlaps qkv_, whose
// last read is a kernel earlier), and lin(wo) has att_'s last read. Nothing here outlives the experts:
// the norms and the tail work on the engine's r_/ao_/fo_, and the small-token MoE uses g_/u_/h_/d_,
// all outside this region. MiMo has no hyper-connection and no per-layer state in these rows.
#pragma once

#include "deepseek4_scratch.hpp"  // ScratchSpan

#include <algorithm>
#include <cstddef>

namespace strata::poly {

struct MimoScratch {
    ScratchSpan qkv, q, k, v, att;
    size_t end = 0;
};

/// The layout for n rows of a call in an arena that starts at `base`, all counts in floats: moe_end is
/// the first float past the batched experts' rows, which the engine allocated at the same base;
/// qkv_wide is the fused projection's row (dim_k * (n_head + n_kv_max) + dim_v * n_kv_max, the widest
/// the plain layout allocates), hq, hk and hv the split row widths at that same n_kv_max, and att_row
/// the attention's output row (n_head * dim_v).
inline MimoScratch mimo_scratch(size_t base, size_t moe_end, size_t n, size_t qkv_wide, size_t hq, size_t hk,
                                size_t hv, size_t att_row) {
    MimoScratch s;
    size_t cursor = base;
    const auto take = [&](size_t count) {
        ScratchSpan p{cursor, count};
        cursor = p.end();
        return p;
    };
    s.qkv = take(n * qkv_wide);        // rope_split is still reading it while it writes the three below
    s.q = take(n * hq);
    s.k = take(n * hk);
    s.v = take(n * hv);
    const size_t rows_end = cursor;
    // att_ arrives after every qkv_ read and after the stores drained k_/v_: it rides those rows from
    // their head. attend writes att_ while it reads q_, so an att_ wider than the qkv_ row (dim_v over
    // dim_k, not this model) stands behind the rows instead.
    if (ScratchSpan{base, n * att_row}.end() <= s.q.at) {
        cursor = base;
        s.att = take(n * att_row);
        cursor = std::max(cursor, rows_end);
    } else {
        s.att = take(n * att_row);
    }
    s.end = std::max(cursor, moe_end);   // the experts' gu/hx/dx rows work over [base, moe_end)
    return s;
}

}  // namespace strata::poly
