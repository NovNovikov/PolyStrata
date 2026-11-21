// Strata. src/dsv41/ds4_vram_plan.hpp - how the VRAM expert slots of one layer group are planned, split out of
// the loader so tests/dsv41/ds4_vram_plan_test.cpp checks the arithmetic without a device, a model or a single
// allocation. The engine's expert loader (ds4_model.cu) fills a VramPlanLayer per loaded layer and calls
// plan_group_vram once per group; the numbers it writes into n_vram are then the slots it allocates. Pure
// arithmetic, no CUDA, no environment: the caller reads DS4_MTP_VRAM_GB and passes mtp_cap (kNoCap when unset).
//
// The shape (unchanged from the loader, donor 19cbf656ab ported): the drafter's (mtp) layers take their slots
// first, each capped at mtp_cap bytes so one block's experts cannot eat the card and starve the prompt path;
// the trunk layers then split what the drafter left proportionally to their needs, floored so no layer gets
// past the group budget; finally whole experts are added layer by layer while they fit. With a cap set the
// drafter never grows past it in that last stage - the freed room goes to the trunk.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace strata::ds4 {

constexpr double kNoCap = 1e300;   // no DS4_MTP_VRAM_GB: the drafter takes what it wants, as before

struct VramPlanLayer {
    bool mtp = false;        // the drafter's block, not a trunk layer
    bool cpu_only = false;   // its type has no device kernel: the CPU lane reads every expert of it
    int n_exp = 0;           // routed experts of the layer
    std::size_t unit = 1;    // bytes of one expert slot
    int n_vram = 0;          // out: the experts that get VRAM slots
};

// One group's plan: `layers` is every layer of the group, `budget` the group's VRAM byte budget for experts,
// `mtp_cap` the per-mtp-layer ceiling (kNoCap: uncapped). Writes n_vram for every layer; sums never pass budget.
inline void plan_group_vram(std::vector<VramPlanLayer>& layers, double budget, double mtp_cap) {
    budget = std::max(0.0, budget);
    double budget_t = budget;   // the drafter's experts first, all of them (every draft uses its blocks)
    for (VramPlanLayer& l : layers)
        if (l.mtp) {
            l.n_vram = std::min(l.n_exp, (int) std::floor(std::min(budget_t, mtp_cap) / (double) l.unit));
            budget_t -= (double) l.n_vram * l.unit;
        }
    double want = 0;
    for (const VramPlanLayer& l : layers)
        if (!l.mtp && !l.cpu_only) want += (double) l.unit * l.n_exp;
    const double frac = want > 0 ? std::min(1.0, std::max(0.0, budget_t) / want) : 0.0;
    for (VramPlanLayer& l : layers) {
        if (l.mtp) continue;
        if (l.cpu_only) {   // (its type has no device kernel: the CPU lane reads every expert of it)
            l.n_vram = 0;
            continue;
        }
        l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
        // never more than fits: the floor of each layer's share can still round up the total
        while (l.n_vram > 0 && (double) l.unit * l.n_vram > budget) --l.n_vram;
    }
    double used = 0;
    for (const VramPlanLayer& l : layers)
        used += (double) l.unit * l.n_vram;
    // what the floors left: whole experts, layer by layer
    for (VramPlanLayer& l : layers) {
        if (l.cpu_only || (l.mtp && mtp_cap < kNoCap)) continue;
        while (l.n_vram < l.n_exp && used + (double) l.unit <= budget) {
            ++l.n_vram;
            used += (double) l.unit;
        }
    }
}

}   // namespace strata::ds4
