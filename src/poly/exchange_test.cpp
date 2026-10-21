// The exchange planner: the pairs it picks and the ones it refuses, without a model and without a card.
#include "exchange.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::poly;
namespace {
int checks = 0;
void check(bool b) { ++checks; if (!b) { std::fprintf(stderr, "exchange_test: failure at check %d\n", checks); std::exit(1); } }

// a layer whose places take one expert of 100 bytes
ExchangeLayer layer_of(uint64_t ram_place = 100, uint64_t vram_place = 100, uint64_t payload = 100) {
    ExchangeLayer l;
    l.ram_place_bytes = ram_place;
    l.vram_place_bytes = vram_place;
    l.payload_bytes = payload;
    return l;
}
ExchangeExpert expert(int layer, int expert, ExchangeHome home, float recent, int64_t prior = 0) {
    ExchangeExpert e;
    e.layer = layer;
    e.expert = expert;
    e.home = home;
    e.payload_bytes = 100;
    e.slot_bytes = 100;
    e.recent = recent;
    e.prior = prior;
    return e;
}
}

int main() {
    // an empty plan is a plan
    {
        const auto plan = plan_exchanges({}, {});
        check(plan.pairs.empty());
        check(plan.considered == 0);
    }

    // one hot expert in the RAM and one cold expert in the VRAM of the same layer: a pair
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 10.f), expert(0, 1, ExchangeHome::vram, 1.f)};
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.size() == 1);
        check(plan.pairs[0].hot == 0 && plan.pairs[0].cold == 1);
        check(plan.considered == 1);
    }

    // the threshold: the hot expert has to be used twice as much, plus the margin
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 5.9f), expert(0, 1, ExchangeHome::vram, 2.f)};
        check(plan_exchanges(e, {layer_of()}).pairs.empty());          // 5.9 < 2*2 + 2
        e[0].recent = 6.01f;
        check(plan_exchanges(e, {layer_of()}).pairs.size() == 1);
    }

    // equal frequencies: no pair
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 5.f), expert(0, 1, ExchangeHome::vram, 5.f)};
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.empty());
        check(plan.skipped[0] == ExchangeSkip::below_threshold);
        check(plan.refused == 1);
    }

    // the layers differ: no partner
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 10.f), expert(1, 1, ExchangeHome::vram, 1.f)};
        const auto plan = plan_exchanges(e, {layer_of(), layer_of()});
        check(plan.pairs.empty());
        check(plan.skipped[0] == ExchangeSkip::no_cold);
    }

    // the place of the cold expert cannot take it: no room
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 10.f), expert(0, 1, ExchangeHome::vram, 1.f)};
        // the RAM place of the layer takes 50 bytes, the cold expert needs 100
        const auto plan = plan_exchanges(e, {layer_of(50, 100, 100)});
        check(plan.pairs.empty());
        check(plan.skipped[0] == ExchangeSkip::no_room);
    }

    // a locked expert is not a candidate
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 10.f), expert(0, 1, ExchangeHome::vram, 1.f)};
        e[0].locked = true;
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.empty());
        check(plan.skipped[0] == ExchangeSkip::locked);
    }

    // two hot experts, one cold expert: the pairs do not overlap
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 20.f), expert(0, 1, ExchangeHome::ram, 19.f),
                                     expert(0, 2, ExchangeHome::vram, 1.f)};
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.size() == 1);
        check(plan.pairs[0].hot == 0 && plan.pairs[0].cold == 2);
    }

    // the hottest hot expert takes the coldest cold one
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 20.f), expert(0, 1, ExchangeHome::ram, 19.f),
                                     expert(0, 2, ExchangeHome::vram, 5.f), expert(0, 3, ExchangeHome::vram, 1.f)};
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.size() == 2);                    // two hot experts, two cold ones: two pairs
        check(plan.pairs[0].hot == 0 && plan.pairs[0].cold == 3);   // the hottest takes the coldest
        check(plan.pairs[1].hot == 1 && plan.pairs[1].cold == 2);   // and the pairs do not overlap
    }

    // the batch is bounded
    {
        std::vector<ExchangeExpert> e;
        for (int i = 0; i < 10; ++i) e.push_back(expert(0, i, ExchangeHome::ram, 100.f - i));
        for (int i = 0; i < 10; ++i) e.push_back(expert(0, 100 + i, ExchangeHome::vram, 1.f));
        const auto plan = plan_exchanges(e, {layer_of()}, 4);
        check(plan.pairs.size() == 4);
    }

    // the pairs do not overlap over a thousand exchanges, and the threshold holds in each of them
    {
        std::vector<ExchangeExpert> e;
        std::vector<ExchangeLayer> layers{layer_of()};
        const int n = 8;
        for (int i = 0; i < n; ++i) e.push_back(expert(0, i, ExchangeHome::ram, 0.f));
        for (int i = 0; i < n; ++i) e.push_back(expert(0, n + i, ExchangeHome::vram, 0.f));
        std::vector<char> in_ram(n, 1), in_vram(n, 1);
        int exchanges = 0;
        for (int round = 0; round < 1000; ++round) {
            // the recent use of every expert, the way the engine counts it
            for (int i = 0; i < n; ++i) {
                e[(size_t) i].recent = 20.f - (float) (round % 7) * (float) i;
                e[(size_t)(n + i)].recent = 1.f + (float) (round % 5) * (float) i;
            }
            const auto plan = plan_exchanges(e, layers);
            std::vector<char> used(e.size(), 0);
            for (const auto& p : plan.pairs) {
                check(p.hot >= 0 && p.cold >= 0);
                check(!used[(size_t) p.hot] && !used[(size_t) p.cold]);
                used[(size_t) p.hot] = 1;
                used[(size_t) p.cold] = 1;
                // the hot expert is in the RAM and the cold one in the VRAM, and the threshold holds
                check(e[(size_t) p.hot].home == ExchangeHome::ram);
                check(e[(size_t) p.cold].home == ExchangeHome::vram);
                check(e[(size_t) p.hot].layer == e[(size_t) p.cold].layer);
                check(e[(size_t) p.hot].recent >= kExchangeRatio * e[(size_t) p.cold].recent + kExchangeMargin);
                // the exchange: the hot expert takes the VRAM place, the cold one the RAM place
                e[(size_t) p.hot].home = ExchangeHome::vram;
                e[(size_t) p.cold].home = ExchangeHome::ram;
                ++exchanges;
            }
        }
        check(exchanges > 0);
    }

    // the protection against the swinging: the experts that were exchanged are not candidates at once
    {
        std::vector<ExchangeExpert> e{expert(0, 0, ExchangeHome::ram, 10.f), expert(0, 1, ExchangeHome::vram, 1.f)};
        e[0].locked = true;
        e[1].locked = true;
        const auto plan = plan_exchanges(e, {layer_of()});
        check(plan.pairs.empty());
        check(plan.skipped[0] == ExchangeSkip::locked);
    }

    std::printf("exchange_test: %d checks OK\n", checks);
    return 0;
}
