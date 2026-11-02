// src/poly/glm5next_scratch.hpp - GLM5Next scratch offsets in floats. Kernels and their arithmetic are
// unchanged; a phase reuses an address only after the previous phase's last read on st_.
//
// A layer's attention is one of two kinds, KDA or DSA, and the batched experts work over the rows the
// engine put at [base, moe_end): a KDA layer's attention rows serve the experts and back, and only one
// layer's own phases constrain the addresses. mix_'s and hcw_'s rows are the one set that outlives the
// attention phase: hc_post() reads hcw_ after the experts ran, so they sit behind moe_end.
#pragma once

#include "deepseek4_scratch.hpp"  // ScratchSpan

#include <algorithm>
#include <cstddef>

namespace strata::poly {

struct GlmScratch {
    ScratchSpan mix, hcw;
    ScratchSpan proj, qkv, t1, t2, ikr, ig, iw, kqv;
    ScratchSpan q, kv, ik, iq, gated, vout;
    ScratchSpan qabs;
    ScratchSpan score, ascore, g1;
    ScratchSpan akeys, g2;
    ScratchSpan qr, beta, fa, ga;
    size_t end = 0;
};

/// The layout for n rows of a call in an arena that starts at `base`, all counts in floats: moe_end is
/// the first float past the batched experts' rows, which the engine allocated at the same base, d is the
/// KDA layer's products (n * d_inner), gate_rank its widest low-rank gate, scores the indexer buffer
/// (score_rows * score_stride), akeys_n and ascore_n the sparse attention's key and score buffers,
/// streams the residual streams (kHc).
inline GlmScratch glm_scratch(size_t base, size_t moe_end, size_t n, size_t d, size_t gate_rank, size_t q_rank,
                              size_t kv_rank, size_t mla_k, size_t mla_v, size_t idx_dim,
                              size_t idx_heads, size_t heads, size_t scores, size_t akeys_n,
                              size_t ascore_n, size_t streams) {
    GlmScratch s;
    size_t cursor = base;
    const auto take = [&](size_t count) {
        ScratchSpan p{cursor, count};
        cursor = p.end();
        return p;
    };
    // The KDA layer's products: proj_ holds its q, k and v, and step() copies a call's rows right past
    // it, which are qkv_'s rows - kda_conv() takes them in place, so the two keep their adjacency.
    s.proj = take(3 * d);
    s.qkv = take(3 * d);
    size_t proj_end = cursor;
    // The DSA layer's temporaries ride those rows, which a KDA layer never launches: rms_norm reads
    // t1_ before q_b writes q_, kv_a's rms_norm reads t2_ before its store, the indexer key's
    // LayerNorm reads ikr_ before ik_ and ig_ fill the ring, and dsa_attend writes kqv_ once the
    // score loop's last read is done.  An occupant that is larger than the KDA rows extends them for
    // the regions behind, so the end moves with the largest one.
    const auto on_proj = [&](size_t count) {
        cursor = s.proj.at;
        const ScratchSpan p = take(count);
        cursor = std::max(cursor, proj_end);
        proj_end = cursor;
        return p;
    };
    s.t1 = on_proj(n * q_rank);
    s.t2 = on_proj(n * kv_rank);
    s.ikr = on_proj(n * idx_dim);
    s.ig = on_proj(n * idx_dim);
    s.iw = on_proj(n * idx_heads);
    s.kqv = on_proj(n * heads * kv_rank);
    // The DSA layer's projections: q_ dies at the absorbed key, kv_ at the latent's store, ik_ at the
    // ring's fill and iq_ at the score loop. gated_ is the KDA step's output, which only lin(wo) reads,
    // and vout_ what the attention leaves for v_b; both kinds never share a layer.
    s.q = take(n * heads * mla_k);
    size_t q_end = cursor;
    const auto on_q = [&](size_t count) {
        cursor = s.q.at;
        const ScratchSpan p = take(count);
        cursor = std::max(cursor, q_end);
        q_end = cursor;
        return p;
    };
    s.kv = on_q(n * kv_rank);
    s.ik = on_q(n * idx_dim);
    s.iq = on_q(n * idx_heads * idx_dim);
    s.gated = on_q(d);
    s.vout = on_q(n * heads * mla_v);
    // The absorbed keys are read by dsa_attend itself: their rows stay alone.
    s.qabs = take(n * heads * kv_rank);
    // The score loop's buffer, the attention's own scores written after its last read, and the KDA
    // forget gate's rows, which its step reads while gated_ is written over on q_.
    s.score = take(scores);
    size_t score_end = cursor;
    const auto on_score = [&](size_t count) {
        cursor = s.score.at;
        const ScratchSpan p = take(count);
        cursor = std::max(cursor, score_end);
        score_end = cursor;
        return p;
    };
    s.ascore = on_score(ascore_n);
    s.g1 = on_score(d);
    // The selected keys live through dsa_attend and the KDA output gate's rows through its step, which
    // writes gated_ over on q_: their kinds never share a layer either.
    s.akeys = take(akeys_n);
    size_t akeys_end = cursor;
    const auto on_akeys = [&](size_t count) {
        cursor = s.akeys.at;
        const ScratchSpan p = take(count);
        cursor = std::max(cursor, akeys_end);
        akeys_end = cursor;
        return p;
    };
    s.g2 = on_akeys(d);
    // qr_ is read again by the score loop's queries, so its rows stay alone; beta_ is read by the step,
    // which is where gated_ is written. lin(f_b) reads fa_ before lin(g_a) writes ga_ over it.
    s.qr = take(n * q_rank);
    s.beta = take(n * heads);
    s.fa = take(n * gate_rank);
    s.ga = s.fa;
    cursor = std::max(cursor, moe_end);    // the experts' gu/hx/dx rows work over [base, moe_end)
    s.mix = take(n * (streams * (2 + streams) + 1));
    s.hcw = take(n * (streams + streams * streams));   // hc_post() reads it after the experts, so it sits behind them
    s.end = cursor;
    return s;
}

}  // namespace strata::poly
