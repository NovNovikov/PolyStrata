// Checks the live sets of Mimo2::attn()/attn_rows()/block_qkv()/block_attend_to()/layer_head() against the
// production layout: rope_split reads qkv_ while it writes q_/k_/v_, the stores drain k_/v_, attend reads q_
// while it writes att_ over the qkv_ rows, and nothing inside the region outlives the experts' rows.
#include "mimo2_scratch.hpp"

#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace strata::poly;

static void require(bool v) { if (!v) throw std::runtime_error("scratch lifetime/bounds check failed"); }

static void live(std::initializer_list<ScratchSpan> spans, size_t base, size_t end) {
    for (auto a = spans.begin(); a != spans.end(); ++a) {
        require(a->at >= base && a->end() <= end && a->at % 64 == 0);
        for (auto b = a + 1; b != spans.end(); ++b)
            require(a->end() <= b->at || b->end() <= a->at);
    }
}

static void check(size_t n) {
    // MiMo V2.6 Flash's geometry: 64 heads, 8 KV heads at the widest layer, dim_k 192, dim_v 128,
    // 8 experts of 2048 hidden over a 4096-wide model - the MOPD/RL Q2_K GGUF's metadata.
    const size_t heads = 64, kv = 8, dk = 192, dv = 128, n_used = 8, n_ff_exp = 2048, n_embd = 4096;
    const size_t qkv_wide = dk * (heads + kv) + dv * kv, hq = heads * dk, hk = kv * dk, hv = kv * dv,
                 att_row = heads * dv;
    const size_t base = 128;
    // The batched experts' rows, as shared_scratch() lays them: rr rows of gu (2 * n_ff_exp), hx
    // (n_ff_exp), dx (n_embd) and hf, with n_used_ expert rows per token.
    const ScratchSpan moe{base, n * n_used * (2 * n_ff_exp + n_ff_exp + n_embd + 1)};
    const auto s = mimo_scratch(base, moe.end(), n, qkv_wide, hq, hk, hv, att_row);

    // Every kernel's live set, attn()'s order; nothing in the region survives the experts' write.
    live({s.qkv}, base, s.end);                                   // lin(wqkv)
    live({s.qkv, s.q, s.k, s.v}, base, s.end);                    // mimo2_rope_split()
    live({s.k}, base, s.end);                                     // store(k)
    live({s.v}, base, s.end);                                     // store(v)
    live({s.q, s.att}, base, s.end);                              // mimo2_attend()
    live({s.att}, base, s.end);                                   // lin(wo)
    live({moe}, base, s.end);                                     // moe_rows() after the attention is done

    // The reuse the layout is for, and the one distance the kernels demand.
    require(s.q.at == s.qkv.end() && s.k.at == s.q.end() && s.v.at == s.k.end());
    require(s.att.at == base);                                    // attend reads q_ while it writes att_
    require(s.att.end() <= s.q.at);
    require(s.end == moe.end());   // MiMo's rows are always narrower than the experts' - the region is free

    if (n > 33) return;
    constexpr uint32_t guard = 0xa53ce19b, expert = 0x1deab0d1;
    std::vector<uint32_t> arena(std::max(s.end, moe.end()) + 64, guard);
    const auto fill = [&](ScratchSpan p, uint32_t value) {
        for (size_t i = p.at; i < p.end(); ++i) arena.at(i) = value;
    };
    const auto phase = [&](std::initializer_list<ScratchSpan> spans) {
        uint32_t value = 1;
        for (auto p : spans) fill(p, value++);
        value = 1;
        for (auto p : spans) {
            for (size_t i = p.at; i < p.end(); ++i) require(arena.at(i) == value);
            ++value;
        }
        fill(moe, expert);   // the layer's experts write their rows; the next layer rewrites its own rows after
        for (size_t i = 0; i < base; ++i) require(arena.at(i) == guard);
        for (size_t i = s.end; i < arena.size(); ++i) require(arena.at(i) == guard);
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        phase({s.qkv});
        phase({s.qkv, s.q, s.k, s.v});   // one kernel: the reads of qkv_ and the writes of the rows behind it
        phase({s.k});
        phase({s.v});
        phase({s.q, s.att});             // one kernel: q_ read, att_ written over the qkv_ rows
        phase({s.att});
    }
    // att_ rode the qkv_ rows: past its own end the qkv_ fill stands, proof the alias is the narrow one.
    for (size_t i = s.att.end(); i < s.qkv.end(); ++i) require(arena.at(i) != guard);
}

int main() {
    try {
        size_t cases = 0;
        for (size_t rows : {1, 8, 31, 32, 33, 256, 1024, 4096}) { check(rows); ++cases; }
        std::printf("PASS: %zu MiMo scratch layouts; disjoint live sets, phase reuse and the attend's distance\n",
                    cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
