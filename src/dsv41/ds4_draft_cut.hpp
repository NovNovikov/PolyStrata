// Strata. src/dsv41/ds4_draft_cut.hpp - the DSpark confidence cut, one rule shared by the two places that apply it:
// the drafter's lazy row loop (ds4_model.cu, STRATA_DS4_DSPARK_LAZY) and main.cpp's verify loop that decides how
// many drafts a round accepts. They must agree exactly - the drafter only skips computing rows the verify would
// reject - so it lives here once, and tests/dsv41/ds4_draft_cut_test.cpp checks the rule (the task spec's 8.5)
// without a device or a model: same function, same production code path.
#pragma once

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

}   // namespace ds4
}   // namespace strata
