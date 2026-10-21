// The placement arithmetic for the expert exchange: no GPU, files or allocation side effects.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
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
    batch_limit,       ///< not selected because the bounded batch is full
    selected,          ///< participates in a selected pair (not a refusal)
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
    struct Group { std::vector<int> cold; size_t next = 0, total = 0, locked = 0; };
    std::vector<Group> groups(layers.size());
    std::vector<int> hot;
    hot.reserve(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        const auto& e = experts[i];
        if (e.layer < 0 || (size_t) e.layer >= layers.size() || !std::isfinite(e.recent) || e.recent < 0)
            throw std::invalid_argument("exchange: invalid layer or recent count");
        if (e.home == ExchangeHome::ram) hot.push_back((int) i);
        else {
            auto& g = groups[(size_t) e.layer];
            ++g.total;
            if (e.locked) ++g.locked;
            else g.cold.push_back((int) i);
        }
    }
    const auto more = [&](int a, int b) {
        const auto& x = experts[(size_t) a]; const auto& y = experts[(size_t) b];
        return x.recent != y.recent ? x.recent > y.recent : x.prior > y.prior;
    };
    std::stable_sort(hot.begin(), hot.end(), more);
    for (auto& g : groups)
        std::stable_sort(g.cold.begin(), g.cold.end(), [&](int a, int b) { return more(b, a); });
    std::vector<char> taken(experts.size(), 0);
    for (int h : hot) {
        ++plan.considered;
        const auto& he = experts[(size_t) h];
        auto& g = groups[(size_t) he.layer];
        const auto& l = layers[(size_t) he.layer];
        ExchangeSkip why = ExchangeSkip::none;
        if (he.locked) why = ExchangeSkip::locked;
        else if (plan.pairs.size() >= max_pairs) why = ExchangeSkip::batch_limit;
        else if (!g.total) why = ExchangeSkip::no_cold;
        else if (g.cold.empty()) why = ExchangeSkip::locked;
        else if (g.next == g.cold.size()) why = ExchangeSkip::taken;
        else if (!he.payload_bytes || !he.slot_bytes || he.slot_bytes > l.vram_place_bytes)
            why = ExchangeSkip::no_room;
        else {
            int chosen = -1;
            bool fits = false;
            // Each layer's sorted list is made once. With the engine's uniform layer geometry this
            // examines one available entry; heterogeneous test inputs still check each destination.
            for (size_t k = g.next; k < g.cold.size(); ++k) {
                const int c = g.cold[k];
                if (taken[(size_t) c]) continue;
                const auto& ce = experts[(size_t) c];
                if (!ce.payload_bytes || ce.payload_bytes > l.ram_place_bytes) continue;
                fits = true;
                if (he.recent < kExchangeRatio * ce.recent + kExchangeMargin) break;
                chosen = c;
                break;
            }
            if (chosen < 0) why = fits ? ExchangeSkip::below_threshold : ExchangeSkip::no_room;
            else {
                plan.pairs.push_back({h, chosen});
                taken[(size_t) h] = taken[(size_t) chosen] = 1;
                plan.skipped[(size_t) h] = plan.skipped[(size_t) chosen] = ExchangeSkip::selected;
                while (g.next < g.cold.size() && taken[(size_t) g.cold[g.next]]) ++g.next;
            }
        }
        if (why != ExchangeSkip::none) { plan.skipped[(size_t) h] = why; ++plan.refused; }
    }
    return plan;
}

/// The recent use of an expert, with the decay the staging cache uses: halved every `half` tokens.
inline constexpr int kExchangeHalf = 128;

} // namespace strata::poly
