// The placement arithmetic for the expert exchange: no GPU, files or allocation side effects.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::poly {

/// Where an expert's main source is.  The exchange moves the main source; the staging copy of an expert is a
/// separate thing and does not take part.
enum class ExchangeHome : uint8_t { ram, vram };

/// A layer's places: how big a RAM place and a VRAM place of it are, and what one expert of it takes.
struct ExchangeLayer {
    uint64_t ram_place_bytes = 0;   ///< the capacity of a RAM place of this layer
    uint64_t vram_place_bytes = 0;  ///< the capacity of a VRAM place of this layer
    uint64_t payload_bytes = 0;     ///< what one expert of this layer takes in RAM: gate + up + down
};

/// One expert as the planner sees it: which place it lives in, how big it is, and how often it was used.
struct ExchangeExpert {
    int layer = 0;
    int expert = 0;
    ExchangeHome home = ExchangeHome::ram;
    uint64_t payload_bytes = 0;    ///< what this expert takes in RAM
    uint64_t slot_bytes = 0;       ///< what this expert takes in a VRAM place, with the tail
    float recent = 0.f;            ///< how often it was used lately, already with the decay
    int64_t prior = 0;             ///< the profile's count: the tie-breaker
    bool locked = false;           ///< exchanged too recently: not a candidate until the lock passes
};

/// A pair to exchange: the hot expert in RAM and the cold expert in VRAM.
struct ExchangePair {
    int hot = -1;    ///< the index of the expert in the RAM, the one that is used more
    int cold = -1;   ///< the index of the expert in the VRAM, the one that is used less
};

/// Why an expert was not exchanged.
enum class ExchangeSkip : uint8_t {
    none = 0,
    no_cold,           ///< no expert in the VRAM of its layer that could take the place
    below_threshold,   ///< not used enough against the coldest one
    locked,            ///< exchanged too recently
    taken,             ///< already in a pair of this batch
    no_room,           ///< the place it would go to cannot take its payload
};

/// What the planner decided.
struct ExchangePlan {
    std::vector<ExchangePair> pairs;
    std::vector<ExchangeSkip> skipped;  ///< per expert, why it was not in a pair
    size_t considered = 0;              ///< the experts that could have been exchanged
    size_t refused = 0;                 ///< the ones that were not, with a reason
};

/// The threshold of the exchange: the hot expert has to be used this many times more than the cold one.
/// The margin is the constant the count has to beat over the cold one, the same one the staging cache uses.
inline constexpr float kExchangeRatio = 2.f;
inline constexpr float kExchangeMargin = 2.f;

/// The pairs to exchange now: the hot experts of the RAM and the cold experts of the VRAM, one place each.
///
/// The experts of a layer share their sizes, so an exchange inside a layer needs no size class of its own; the
/// check of the room in the places it goes to is made anyway, because the same sizes are not the same room.
/// The hottest expert takes the coldest partner it can have, and neither of them is offered again, so the pairs
/// do not overlap.  `max_pairs` bounds the batch: the link to the card is one, and a batch that fills it serves
/// nobody.
inline ExchangePlan plan_exchanges(const std::vector<ExchangeExpert>& experts,
                                   const std::vector<ExchangeLayer>& layers,
                                   size_t max_pairs = 8) {
    ExchangePlan plan;
    plan.skipped.assign(experts.size(), ExchangeSkip::none);
    std::vector<char> taken(experts.size(), 0);
    // the hot experts of the RAM, the most used first; the profile breaks a tie
    std::vector<int> hot;
    hot.reserve(experts.size());
    for (size_t i = 0; i < experts.size(); ++i)
        if (experts[i].home == ExchangeHome::ram) hot.push_back((int) i);
    std::stable_sort(hot.begin(), hot.end(), [&](int a, int b) {
        const auto& x = experts[(size_t) a];
        const auto& y = experts[(size_t) b];
        return x.recent != y.recent ? x.recent > y.recent : x.prior > y.prior;
    });
    for (int h : hot) {
        if (plan.pairs.size() >= max_pairs) break;
        const auto& he = experts[(size_t) h];
        if (he.locked) { plan.skipped[(size_t) h] = ExchangeSkip::locked; ++plan.refused; continue; }
        if (taken[(size_t) h]) { plan.skipped[(size_t) h] = ExchangeSkip::taken; ++plan.refused; continue; }
        ++plan.considered;
        // the cold experts of the VRAM of the same layer, the least used first
        std::vector<int> cold;
        for (size_t i = 0; i < experts.size(); ++i) {
            const auto& c = experts[i];
            if (c.home != ExchangeHome::vram || c.layer != he.layer) continue;
            if (taken[i] || c.locked) continue;
            cold.push_back((int) i);
        }
        if (cold.empty()) {
            // either there is no partner at all, or the ones there are locked or taken: say which
            bool any = false;
            for (size_t i = 0; i < experts.size(); ++i) {
                const auto& c = experts[i];
                if (c.home == ExchangeHome::vram && c.layer == he.layer) { any = true; break; }
            }
            plan.skipped[(size_t) h] = any ? ExchangeSkip::taken : ExchangeSkip::no_cold;
            ++plan.refused;
            continue;
        }
        std::stable_sort(cold.begin(), cold.end(), [&](int a, int b) {
            const auto& x = experts[(size_t) a];
            const auto& y = experts[(size_t) b];
            return x.recent != y.recent ? x.recent < y.recent : x.prior < y.prior;
        });
        // the coldest partner that the places can take
        int chosen = -1;
        bool no_room = false;
        for (int c : cold) {
            const auto& ce = experts[(size_t) c];
            const auto& layer = layers[(size_t) he.layer];
            const bool fits_ram = layer.ram_place_bytes >= ce.payload_bytes;
            const bool fits_vram = layer.vram_place_bytes >= he.slot_bytes;
            if (!fits_ram || !fits_vram) { no_room = true; continue; }
            if (he.recent < kExchangeRatio * ce.recent + kExchangeMargin) continue;   // the threshold
            chosen = c;
            break;
        }
        if (chosen < 0) {
            plan.skipped[(size_t) h] = no_room ? ExchangeSkip::no_room : ExchangeSkip::below_threshold;
            ++plan.refused;
            continue;
        }
        plan.pairs.push_back({h, chosen});
        taken[(size_t) h] = 1;
        taken[(size_t) chosen] = 1;
    }
    for (size_t i = 0; i < experts.size(); ++i) {
        if (plan.skipped[i] == ExchangeSkip::none && taken[i] && experts[i].home == ExchangeHome::vram)
            plan.skipped[i] = ExchangeSkip::taken;
    }
    return plan;
}

/// The recent use of an expert, with the decay the staging cache uses: halved every `half` tokens.
inline constexpr int kExchangeHalf = 128;

} // namespace strata::poly
