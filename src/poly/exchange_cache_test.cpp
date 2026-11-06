// src/poly/exchange_cache_test.cpp - Do the expert exchange and its staging copy cache survive in one breath?
// The engine opens --no-disk with both the exchange and --no-disk-exchange-cache, payload verification on for
// both paths (STRATA_VERIFY_EXCHANGE, STRATA_VERIFY_STAGE_CACHE), reads a short prompt and then decodes, which
// is where follow_resident() fills the staging slots and follow_exchange() trades permanent ones. After every
// 32 tokens the test asks where every expert's weights stand: a permanent slot must have no host twin, a
// staging copy must have one, stand on the slot grid above the swap's scratch slot and be named by its owner
// entry, and an owner entry must not outlive its copy (Engine::check_resident_ownership_for_test). At the end
// both mechanisms must have done something - exchanges completed, copies standing - and nothing failed.
//
//   poly_exchange_cache_test MODEL.gguf [STEPS]      (STEPS >= 64, default 400)
//
// Needs the model file, a card and the RAM to lock the experts, so it is run by hand. Verification reads every
// exchanged and copied payload twice: the run is slower than an untuned decode, which is the point.
// Exit 0: pass; 1: an invariant broke or a mechanism stayed idle; 2: an error or a bad argument; 3: SKIP.
#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using strata::poly::Engine;
using strata::poly::Options;

namespace {

void verify_env(const char* name, const char* value) {
#ifdef _WIN32
    const std::string kv = std::string(name) + "=" + value;
    _putenv(kv.c_str());
#else
    setenv(name, value, 1);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [STEPS]\n", argv[0]);
        return 2;
    }
    int steps = 400;
    if (argc > 2) {
        char* end = nullptr;
        const long v = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || v < 64 || v > 100000) {
            std::fprintf(stderr, "STEPS is a whole number from 64 to 100000, not '%s'\n", argv[2]);
            return 2;
        }
        steps = (int) v;
    }
    verify_env("STRATA_VERIFY_EXCHANGE", "1");      // both payloads of every exchange, hashed on card and host
    verify_env("STRATA_VERIFY_STAGE_CACHE", "1");   // every staging copy against its RAM twin
    Options o;
    o.model = argv[1];
    o.no_disk = true;
    o.no_disk_expert_exchange = true;
    o.no_disk_exchange_cache = true;
    o.expert_swaps = 8;                 // the uploads in flight and the gate follow_exchange() shares
    o.max_context = 200 + steps + 64;   // the read plus the decoded tail
    o.prompt_batch = 256;
    const int n = 200;
    std::vector<int32_t> text((size_t) n);
    for (int i = 0; i < n; ++i) text[(size_t) i] = 1000 + (53 * i + i / 9) % 100000;
    try {
        auto e = Engine::open(o);
        Engine& engine = *e;
        if (engine.cached_experts() == 0) {
            std::fprintf(stderr, "SKIP: no expert got a permanent slot (VRAM too tight for the hot arena?)\n");
            return 3;
        }
        engine.bind(0);
        for (int i = 0; i < n; i += engine.max_rows()) {
            const int m = std::min(n - i, engine.max_rows());
            engine.eval(text.data() + i, m, i + m == n);
        }
        int32_t t = (int32_t) strata::poly::most_likely(engine.logits(), engine.n_vocab());
        const strata::poly::ExchangeStats before = engine.exchange_stats();
        for (int i = 0; i < steps; ++i) {
            engine.eval(&t, 1, true);
            t = (int32_t) strata::poly::most_likely(engine.logits(), engine.n_vocab());
            if ((i & 31) == 31) {
                const auto r = engine.check_resident_ownership_for_test();
                std::fprintf(stderr, "token %d: hot %d, ram %d, staging copies %d, free slots %d\n",
                             i + 1, r.hot, r.ram, r.staging, r.staging_free);
            }
        }
        const auto r = engine.check_resident_ownership_for_test();
        const strata::poly::ExchangeStats& now = engine.exchange_stats();
        std::fprintf(stderr, "final: hot %d, ram %d, staging copies %d, free slots %d; exchanges completed %llu "
                             "of %llu, failed %llu, restored %llu\n", r.hot, r.ram, r.staging, r.staging_free,
                     (unsigned long long) (now.completed - before.completed), (unsigned long long) now.completed,
                     (unsigned long long) now.failed, (unsigned long long) now.restored);
        engine.print_exchange_stats(before);
        int bad = 0;
        if (now.failed != 0) { std::fprintf(stderr, "FAIL: %llu exchange(s) failed\n", (unsigned long long) now.failed); ++bad; }
        if (now.completed == before.completed) {
            std::fprintf(stderr, "FAIL: the exchange completed nothing in %d decoded tokens\n", steps); ++bad;
        }
        if (r.staging == 0) { std::fprintf(stderr, "FAIL: no copy stood on staging at the end\n"); ++bad; }
        std::fprintf(stderr, "%s: %d decoded tokens, ownership %s\n", bad ? "FAIL" : "PASS", steps,
                     bad ? "or activity check failed" : "and activity checks held");
        return bad ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 2;
    }
}
