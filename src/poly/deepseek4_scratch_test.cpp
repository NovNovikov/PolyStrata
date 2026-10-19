// Checks the live sets of DeepSeek4::attention/out_proj/moe_rows against the
// production layout, including sizes where preparation is larger than att_.
#include "deepseek4_scratch.hpp"

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

static void check(size_t n, size_t ctx, bool mmq, size_t output_scale = 1) {
    const size_t base = 128, stride = ctx / 4 + 1;
    const size_t scores = std::clamp<size_t>((128u << 20) / (stride * 4), 1, n) * stride;
    const ScratchSpan gu{base, mmq ? n * 6 * 4096 : 0};
    const ScratchSpan hx{gu.end(), mmq ? n * 6 * 2048 : 0};
    const ScratchSpan dx{hx.end(), mmq ? n * 6 * 4096 : 0};
    const auto s = deepseek_scratch(base, dx.end(), n, n * 64 * 512, 1024, 512,
                                   n * 8 * 1024 * output_scale, n * 64 * 128, 64, scores, 4);
    live({s.q, s.t1, s.qr, s.kv, s.ckv, s.cscore, s.iq, s.iw, s.score, s.mix, s.hcw}, base, s.end);
    live({s.q, s.att, s.oa, s.mix, s.hcw}, base, s.end);
    live({s.oa, s.ob, s.mix, s.hcw}, base, s.end);
    if (mmq) live({gu, hx, dx, s.mix, s.hcw}, base, s.end);
    require(s.q.at == s.ob.at && s.att.at == s.t1.at);
    if (n == 4096 && ctx == 131072 && mmq && output_scale == 1)
        require((s.end - base) * 4 == 1208696832); // 1152 MiB workspace + 720 KiB hyper-connections

    if (n > 33) return;
    constexpr uint32_t guard = 0xa53ce19b, persistent = 0x789afed1;
    std::vector<uint32_t> arena(s.end + 64, guard);
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
        for (auto p : {s.mix, s.hcw})
            for (size_t i = p.at; i < p.end(); ++i) require(arena.at(i) == persistent);
        for (size_t i = 0; i < base; ++i) require(arena.at(i) == guard);
        for (size_t i = s.end; i < arena.size(); ++i) require(arena.at(i) == guard);
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        phase({s.q, s.t1, s.qr, s.kv, s.ckv, s.cscore, s.iq, s.iw, s.score});
        phase({s.q, s.att, s.oa});
        phase({s.oa, s.ob});
        if (mmq) phase({gu, hx, dx});
    }
}

int main() {
    try {
        size_t cases = 0;
        for (size_t rows : {1, 8, 31, 32, 33, 256, 1024, 4096})
            for (size_t ctx : {16, 16384, 32768, 131072})
                for (bool mmq : {false, true})
                    for (size_t output : {1, 8}) {
                        check(rows, ctx, mmq, output);
                        ++cases;
                    }
        std::printf("PASS: %zu scratch layouts; disjoint live sets, phase reuse and guarded persistent state\n", cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
