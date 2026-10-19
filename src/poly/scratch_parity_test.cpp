// Full-logit comparison of the original and compact layouts on synthetic input.
// Manual GPU test: poly_scratch_parity_test MODEL.gguf [BATCH=4096] [graphs|no-graphs] [CONTEXT=16384]
// No permanent GPU experts or adaptive cache: placement and sum order are fixed.
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace strata::poly;
using Trace = std::vector<std::vector<float>>;

static void expect(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

static Trace exercise(Options o) {
    auto e = Engine::open(o);
    expect(e->cached_experts() == 0, "parity needs zero cached experts in both layouts");
    std::vector<int32_t> text(2 * o.prompt_batch + 400);
    for (size_t i = 0; i < text.size(); ++i) text[i] = 1000 + (53 * i + i / 9) % 100000;
    Trace trace;
    const auto take = [&](const float* p) {
        for (int j = 0; j < e->n_vocab(); ++j) expect(std::isfinite(p[j]), "NaN/Inf logits");
        trace.emplace_back(p, p + e->n_vocab());
    };
    const auto read = [&](int n) {
        int done = 0;
        while (done < n) {
            const int m = std::min(e->max_read(), n - done);
            expect(e->read(text.data() + done, m), "unexpected cancelled read");
            done += m;
        }
    };
    for (int n : {1, 8, 31, 32, 33}) {
        e->reset();
        read(n);
        e->eval(text.data() + n, 1, true);
        take(e->logits());
        for (int k = 0; k < 3; ++k) {
            e->eval(text.data() + n + 1 + k, 1, true);
            take(e->logits());
        }
        const int guesses = std::min(4, e->max_window());
        e->window(text.data() + n + 4, guesses);
        for (int k = 0; k < guesses; ++k) take(e->logits_of(k));
        e->keep(std::min(2, guesses));
        e->eval(text.data() + n + 6, 1, true);
        take(e->logits());
        std::fprintf(stderr, "parity compact=%d: boundary %d and window/keep complete\n", o.compact_scratch, n);
    }

    e->reset();
    const int prefix = o.prompt_batch + 33;
    read(prefix); // full batch + partial final batch; indexer scores active at 4096
    e->eval(text.data() + prefix, 1, true);
    take(e->logits());
    // State/attention rows survive a round trip with overwritten future rows.
    const int64_t at = e->pos();
    std::vector<uint8_t> state(e->state_bytes()), rows(at * e->row_bytes());
    e->save_state(state.data());
    e->save_rows(0, at, rows.data());
    e->eval(text.data(), 1, true);
    take(e->logits());
    e->load_rows(rows.data(), 0, at, 0, at);
    e->load_state(state.data(), at);
    e->eval(text.data(), 1, true);
    take(e->logits());
    expect(trace[trace.size()-1] == trace[trace.size()-2], "state/row restore changed logits");

    e->copy_seq(0, 1);
    const int seqs[] = {0, 1};
    e->step(seqs, text.data() + 1, 2);
    take(e->logits_of(0)); take(e->logits_of(1));
    e->bind(0);
    // Cancellation followed by a new prefill must not retain reused scratch.
    const std::function<bool(int, int)> cancel = [](int, int) { return false; };
    const int cancel_n = std::min(prefix, e->max_read());
    if (cancel_n > e->max_rows())
        expect(!e->read(text.data(), cancel_n, nullptr, 0, &cancel), "read did not exercise cancellation");
    e->reset(); read(33);
    e->eval(text.data() + 33, 1, true);
    take(e->logits());
    std::fprintf(stderr, "parity compact=%d: partial batch, restore, step and cancellation complete\n", o.compact_scratch);
    return trace;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [BATCH=4096] [graphs|no-graphs] [CONTEXT=16384]\n", argv[0]);
        return 2;
    }
    try {
#ifdef _WIN32
        _putenv_s("STRATA_FREEZE_EXPERTS", "1");
#else
        setenv("STRATA_FREEZE_EXPERTS", "1", 1);
#endif
        Options o;
        o.model = argv[1];
        o.prompt_batch = argc > 2 ? std::atoi(argv[2]) : 4096;
        o.graphs = argc < 4 || std::strcmp(argv[3], "no-graphs") != 0;
        o.max_context = argc > 4 ? std::atoi(argv[4]) : 16384;
        o.read_chunk_tokens = std::max(128, 2 * o.prompt_batch);
        o.threads = 12; o.stage_mib = 2048; o.stage_ahead = 32;
        o.rows_vram = true; o.batch = 1; o.draft = false;
        o.window = std::min(8, o.prompt_batch);
        o.expert_swaps = 0; o.cpu_share = 0;
        o.vram_reserve_mib = 32768; // mmap mode: no permanent experts in either layout
        const auto reference = exercise(o);
        o.compact_scratch = true;
        const auto compact = exercise(o);
        expect(reference.size() == compact.size(), "trace length changed");
        size_t count = 0, different = 0;
        double max_abs = 0, sum_abs = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            expect(reference[i].size() == compact[i].size(), "vocabulary changed");
            for (size_t j = 0; j < reference[i].size(); ++j) {
                const double delta = std::abs(double(reference[i][j]) - compact[i][j]);
                max_abs = std::max(max_abs, delta); sum_abs += delta; ++count;
                different += std::memcmp(&reference[i][j], &compact[i][j], sizeof(float)) != 0;
            }
        }
        std::printf("%s: batch=%d graphs=%d frames=%zu logits=%zu bit_differences=%zu max_abs=%.9g mean_abs=%.9g\n",
                    different ? "FAIL" : "PASS", o.prompt_batch, o.graphs, reference.size(), count, different,
                    max_abs, sum_abs / count);
        return different ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
