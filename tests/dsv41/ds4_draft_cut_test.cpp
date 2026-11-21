// Strata. The DSpark confidence cut (ds4_draft_cut.hpp) on synthetic confidences - the rule the lazy drafter's
// row loop (STRATA_DS4_DSPARK_LAZY, the e6a49369fb port) and main.cpp's verify share, checked without a model
// for the task spec's 8.5 cases: everything passes, the first or a middle draft is rejected, the cumulative
// cutoff stops a round whose drafts pass individually, the speculative window caps the round, a confidence
// exactly on either cutoff, and several rounds in a row. Token identity between eager and lazy is the shared
// rule plus the row-at-a-time head (reviewed at the call site); sampling RNG needs a device and is noted
// untested in the port report.
#include "ds4_draft_cut.hpp"

#include <cstdio>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

// one round: the product starts at 1.0f like both callers' `float cum = 1.0f`
static int cut(const std::vector<float>& w, int n, float conf_min, float cum_min) {
    float cum = 1.0f;
    return draft_cut_keep(w.data(), 0, n, cum, conf_min, cum_min);
}

int main() {
    const float cm = 0.5f, cc = 0.1f;

    // 1. every draft passes
    check(cut({0.9f, 0.8f, 0.7f}, 3, cm, cc) == 3, "all confidences over the cutoffs: the whole window is kept");

    // 2. the first draft fails
    check(cut({0.4f, 0.9f, 0.9f}, 3, cm, cc) == 0, "a first draft under conf_min: none is kept");
    // a middle draft fails: rows past it are never taken, whatever they hold
    check(cut({0.9f, 0.9f, 0.49f, 0.9f}, 4, cm, cc) == 2, "a middle draft under conf_min: the round stops before it");

    // 3. the cumulative cutoff: individually strong drafts, product below cum_min
    check(cut({0.6f, 0.6f, 0.6f, 0.6f}, 4, cm, 0.2f) == 3, "cumulative cutoff: 0.6*0.6*0.6=0.216 vs 0.2 keeps 3, not 4");
    {
        // exactly at the cumulative cutoff: 0.5*0.5 = 0.25 = cum_min keeps (inclusive boundary)
        float cum = 1.0f;
        const float w[3] = {0.5f, 0.5f, 0.5f};
        check(draft_cut_keep(w, 0, 3, cum, 0.4f, 0.25f) == 2, "the product exactly on cum_min: kept");
    }

    // 4. confidence exactly on the cutoff: kept (>=), one ULP under: cut
    check(cut({0.5f, 0.5f}, 2, cm, cc) == 2, "confidences exactly at conf_min: kept");
    check(cut({0.5f, 0.49999997f}, 2, cm, cc) == 1, "the float under conf_min: cut");

    // 5. the window's end caps the round: n is already the min with room-1 and the spec depth, as in main.cpp
    check(cut({0.9f, 0.9f, 0.9f, 0.9f, 0.9f}, 3, cm, cc) == 3, "a short window over a long row list: n ends the round");

    // 6. several rounds in a row: each round restarts the product (both callers' fresh `cum`), keep per round right
    {
        const std::vector<float> r1{0.9f, 0.9f}, r2{0.9f, 0.3f}, r3{0.5f, 0.5f, 0.9f};
        check(cut(r1, 2, cm, cc) == 2 && cut(r2, 2, cm, cc) == 1 && cut(r3, 3, cm, cc) == 3,
              "three rounds back to back: per-round product, per-round cut");
    }

    // 7. the carried product and partial keep: a caller that already kept two rows continues correctly
    {
        float cum = 1.0f;
        const float w[4] = {0.8f, 0.8f, 0.8f, 0.8f};
        int keep = draft_cut_keep(w, 0, 2, cum, cm, 0.0f);   // first two through the first block of the lazy loop
        check(keep == 2, "first block keeps its two rows");
        keep = draft_cut_keep(w, keep, 4, cum, cm, 0.4f);    // the second block with the product carried
        check(keep == 4 && cum > 0.400f && cum < 0.42f, "the second block continues with the carried product");
    }

    // 8. a weight failing conf_min does not touch the product (short-circuit order, same on both callers)
    {
        float cum = 1.0f;
        const float w[1] = {0.2f};
        check(draft_cut_keep(w, 0, 1, cum, cm, cc) == 0 && cum == 1.0f, "a conf_min failure leaves the product alone");
    }

    std::printf("ds4_draft_cut_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
