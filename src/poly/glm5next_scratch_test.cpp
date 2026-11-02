// Checks the live sets of Glm5Next::kda()/dsa()/dsa_store()/dsa_select()/layer_head() against the
// production layout, including the rows_of_ call where kda_conv() takes proj_'s copied rows in place.
#include "glm5next_scratch.hpp"

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

static void check(size_t n, size_t ctx) {
    // GLM 5.3 Flash's geometry; gate_rank stands for the KDA layers' low-rank gates.
    const size_t base = 128, streams = 4, heads = 64, kda_dim = 128, q_rank = 1536, kv_rank = 512,
                 mla_k = 256, mla_v = 256, idx_dim = 128, idx_heads = 32, gate_rank = 1024,
                 kAttnRows = 32, kpool = 4, idx_top_k = 2048;
    const size_t d = n * heads * kda_dim;
    const size_t stride = ctx / kpool + 1;
    const size_t scores = std::clamp<size_t>((128u << 20) / (stride * 4), 1, n) * stride;
    const size_t keys = std::min(ctx, idx_top_k + kpool - 1);
    // The batched experts' rows, as shared_scratch() lays them out: rr rows of gu (2 * n_ff_exp), hx
    // (n_ff_exp), dx (n_embd) and hf, with n_used_ expert rows per token.
    const ScratchSpan moe{base, n * 8 * (2 * 2048 + 2048 + 4096 + 1)};
    const auto s = glm_scratch(base, moe.end(), n, d, gate_rank, q_rank, kv_rank, mla_k, mla_v, idx_dim, idx_heads,
                               heads, scores, std::min(n, kAttnRows) * keys * kv_rank,
                               std::min(n, kAttnRows) * keys * heads, streams);

    // hc_pre() writes mix_/hcw_, and hc_post() of the layer's tail reads hcw_ after the experts ran.
    live({moe, s.mix, s.hcw}, base, s.end);                                     // the MoE and the hyper-connections

    // kda(): the products, the convolved rows, the gates, the step, lin(wo).
    live({s.proj, s.qkv, s.hcw}, base, s.end);                                // kda_conv()
    live({s.fa, s.g1, s.qkv, s.hcw}, base, s.end);                            // lin(f_b)
    live({s.ga, s.g2, s.qkv, s.g1, s.beta, s.hcw}, base, s.end);              // lin(g_b)
    live({s.qkv, s.g1, s.g2, s.beta, s.gated, s.hcw}, base, s.end);           // kda_step()
    live({s.gated, s.hcw}, base, s.end);                                      // lin(wo)
    // dsa(): the projections, the stores, the indexer, the attention, the output.
    live({s.t1, s.qr, s.hcw}, base, s.end);                                   // rms_norm(q_a)
    live({s.qr, s.q, s.hcw}, base, s.end);                                    // lin(q_b)
    live({s.q, s.qabs, s.qr, s.hcw}, base, s.end);                            // lin(k_b)
    live({s.t2, s.kv, s.qabs, s.qr, s.hcw}, base, s.end);                     // rms_norm(kv_a)
    live({s.ikr, s.ik, s.qabs, s.qr, s.hcw}, base, s.end);              // layer_norm(idx_k)
    live({s.ik, s.ig, s.qabs, s.qr, s.hcw}, base, s.end);                     // idx_store()
    live({s.iq, s.iw, s.score, s.akeys, s.qabs, s.qr, s.hcw}, base, s.end);   // idx_score_batch()
    live({s.score, s.akeys, s.qabs, s.hcw}, base, s.end);                     // idx_select()
    live({s.qabs, s.akeys, s.kqv, s.ascore, s.hcw}, base, s.end);             // dsa_attend()
    live({s.kqv, s.vout, s.hcw}, base, s.end);                                // lin(v_b)
    live({s.vout, s.hcw}, base, s.end);                                       // lin(wo)

    // The adjacency step()'s copied rows rely on, and the reuse the layout is for.
    require(s.qkv.at == s.proj.end());
    require(s.t1.at == s.proj.at && s.kqv.at == s.proj.at);
    require(s.gated.at == s.q.at && s.vout.at == s.q.at);
    require(s.g1.at == s.score.at && s.g2.at == s.akeys.at && s.ga.at == s.fa.at);
    if (n == 4096 && ctx == 200000)
        require((s.end - base) * 4 == 1922973696);  // 1834 MiB against the 3128 MiB the rows take apart

    if (n > 33) return;
    constexpr uint32_t guard = 0xa53ce19b, persistent = 0x789afed1, expert = 0x1deab0d1;
    std::vector<uint32_t> arena(std::max(s.end, moe.end()) + 64, guard);
    const auto fill = [&](ScratchSpan p, uint32_t value) {
        for (size_t i = p.at; i < p.end(); ++i) arena.at(i) = value;
    };
    fill(s.mix, persistent); fill(s.hcw, persistent);
    const auto phase = [&](std::initializer_list<ScratchSpan> spans) {
        uint32_t value = 1;
        for (auto p : spans) fill(p, value++);
        value = 1;
        for (auto p : spans) {
            for (size_t i = p.at; i < p.end(); ++i) require(arena.at(i) == value);
            ++value;
        }
        fill(moe, expert);   // the layer's experts write their rows, then hc_post() still reads hcw_
        for (auto p : {s.mix, s.hcw})
            for (size_t i = p.at; i < p.end(); ++i) require(arena.at(i) == persistent);
        for (size_t i = 0; i < base; ++i) require(arena.at(i) == guard);
        for (size_t i = s.end; i < arena.size(); ++i) require(arena.at(i) == guard);
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        phase({s.proj, s.qkv});
        phase({s.fa, s.g1, s.qkv});
        phase({s.ga, s.g2, s.qkv, s.g1, s.beta});
        phase({s.qkv, s.g1, s.g2, s.beta, s.gated});
        phase({s.gated});
        phase({s.t1, s.qr});
        phase({s.qr, s.q});
        phase({s.q, s.qabs, s.qr});
        phase({s.t2, s.kv, s.qabs, s.qr});
        phase({s.ikr, s.ik, s.qabs, s.qr});
        phase({s.ik, s.ig, s.qabs, s.qr});
        phase({s.iq, s.iw, s.score, s.akeys, s.qabs, s.qr});
        phase({s.score, s.akeys, s.qabs});
        phase({s.qabs, s.akeys, s.kqv, s.ascore});
        phase({s.kqv, s.vout});
        phase({s.vout});
    }
}

int main() {
    try {
        size_t cases = 0;
        for (size_t rows : {1, 8, 31, 32, 33, 256, 1024, 4096})
            for (size_t ctx : {16, 16384, 131072, 200000}) {
                check(rows, ctx);
                ++cases;
            }
        std::printf("PASS: %zu GLM scratch layouts; disjoint live sets, phase reuse and guarded persistent state\n", cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
