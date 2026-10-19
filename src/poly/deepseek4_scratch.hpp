// DeepSeek4 scratch offsets in floats. Kernels and their arithmetic are unchanged;
// phases reuse addresses only after the previous phase's last read on st_.
#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::poly {

struct ScratchSpan {
    size_t at = 0, count = 0;
    size_t end() const { return at + ((count + 63) & ~size_t(63)); }
};

struct DeepSeekScratch {
    ScratchSpan q, ob, att, t1, qr, kv, ckv, cscore, iq, iw, score, oa, mix, hcw;
    size_t end = 0;
};

inline DeepSeekScratch deepseek_scratch(size_t base, size_t shared_end, size_t n, size_t heads,
                                      size_t rank, size_t dim, size_t output, size_t index,
                                      size_t index_heads, size_t scores, size_t streams) {
    DeepSeekScratch s;
    size_t cursor = base;
    const auto take = [&](size_t count) {
        ScratchSpan p{cursor, count};
        cursor = p.end();
        return p;
    };
    s.q = take(heads);
    cursor = base;
    s.ob = take(output);                 // all q reads finish before oa -> ob
    cursor = std::max(cursor, s.q.end());
    const size_t prepare = cursor;
    s.att = take(heads);
    cursor = prepare;
    s.t1 = take(n * rank);
    s.qr = take(n * rank);
    s.kv = take(n * dim);
    s.ckv = take(n * 2 * dim);
    s.cscore = take(n * 2 * dim);
    s.iq = take(index);
    s.iw = take(n * index_heads);
    s.score = take(scores);             // preparation/indexer finish before attend
    cursor = std::max(cursor, s.att.end());
    s.oa = take(output);                // q/att/oa must be distinct for out_proj
    cursor = std::max(cursor, shared_end); // MoE gu/hx/dx use the same base later
    s.mix = take(n * (streams * (2 + streams) + 1));
    s.hcw = take(n * (streams + streams * streams)); // survives through MoE/tail
    s.end = cursor;
    return s;
}

} // namespace strata::poly
