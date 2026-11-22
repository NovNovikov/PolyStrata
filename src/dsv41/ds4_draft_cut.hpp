// Strata. src/dsv41/ds4_draft_cut.hpp - the DSpark confidence cut, one rule shared by the two places that apply it:
// the drafter's lazy row loop (ds4_model.cu, STRATA_DS4_DSPARK_LAZY) and main.cpp's verify loop that decides how
// many drafts a round accepts. They must agree exactly - the drafter only skips computing rows the verify would
// reject - so it lives here once, and tests/dsv41/ds4_draft_cut_test.cpp checks the rule (the task spec's 8.5)
// without a device or a model: same function, same production code path.
#pragma once

#include <algorithm>
#include <vector>

namespace strata {
namespace ds4 {

// Advance the kept count over w[keep .. n): a draft stays kept while its own confidence reaches conf_min and the
// running product of the kept confidences stays at or above cum_min. Boundaries are inclusive (>= both times):
// a weight exactly at the cutoff keeps. `cum` carries the product of the already-kept weights (1.0: none yet)
// and leaves carrying the product including the newly kept ones; a weight that fails has still multiplied it,
// as the loop's short-circuit order gives (the drafter's draft t-1 was not accepted either way).
inline int draft_cut_keep(const float* w, int keep, int n, float& cum, float conf_min, float cum_min) {
    while (keep < n && w[keep] >= conf_min && (cum *= w[keep]) >= cum_min) ++keep;
    return keep;
}

// The lazy drafter's batch plan (STRATA_DS4_DSPARK_LAZY): the row counts the head computes, in order, each
// endpoint cumulative - rows [previous, this) go through the head, markov and sampler, and confidence
// min(endpoint + 1, lim) is read after them (confidence t is whether draft t survives). Confidence 0 first:
// it needs no head row at all, and if it cuts, the block ends with zero rows made. Then rows 0 and 1 with
// their confidences - the common cut falls inside them - then everything left up to lim, the bound
// min(T, max_drafts, lz_max; <= 0: no bound of its own). Nondecreasing, deduplicated, inside [0, lim].
inline std::vector<int> dspark_lazy_batches(int T, int max_drafts, int lz_max) {
    int lim = T;
    if (max_drafts > 0) lim = std::min(lim, max_drafts);
    if (lz_max > 0) lim = std::min(lim, lz_max);
    std::vector<int> plan;
    for (const int b0 : {0, 2, lim}) {
        const int b = std::min(b0, lim);
        if (plan.empty() || b > plan.back()) plan.push_back(b);
    }
    return plan;
}

}   // namespace ds4
}   // namespace strata
