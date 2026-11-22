// Strata. The DSpark confidence cut (ds4_draft_cut.hpp) on synthetic confidences - the rule the lazy drafter's
// row loop (STRATA_DS4_DSPARK_LAZY, the e6a49369fb port) and main.cpp's verify share, checked without a model
// for the task spec's 8.5 cases: everything passes, the first or a middle draft is rejected, the cumulative
// cutoff stops a round whose drafts pass individually, the speculative window caps the round, a confidence
// exactly on either cutoff, and several rounds in a row. Token identity between eager and lazy is the shared
// rule plus the row-at-a-time head (reviewed at the call site); sampling RNG needs a device and is noted
// untested in the port report.
#include "ds4_draft_cut.hpp"

#include <algorithm>
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

    // 9. the lazy batch plan (hostile review 7): endpoints strictly increasing, starting at 0 (confidence 0
    // costs no head row), ending at lim = min(T, max_drafts>0, lz_max>0), all inside [0, lim]
    {
        check(dspark_lazy_batches(8, 0, 0) == std::vector<int>({0, 2, 8}), "the full block: conf 0, rows 0-1, the rest");
        check(dspark_lazy_batches(1, 0, 0) == std::vector<int>({0, 1}), "one draft: no phantom row-2 batch");
        check(dspark_lazy_batches(2, 0, 0) == std::vector<int>({0, 2}), "two drafts: the row batch is the last");
        check(dspark_lazy_batches(16, 5, 0) == std::vector<int>({0, 2, 5}), "max_drafts bounds the plan");
        check(dspark_lazy_batches(16, 1, 0) == std::vector<int>({0, 1}), "a bound of 1: conf 0 and one row");
        check(dspark_lazy_batches(16, 16, 2) == std::vector<int>({0, 2}), "lz_max bounds it too");
        check(dspark_lazy_batches(16, -3, 0) == std::vector<int>({0, 2, 16}), "a negative bound of its own: ignored");
        bool inv = true;
        for (int T = 1; T <= 16; ++T)
            for (int md : {0, 1, 3, 17, -1})
                for (int lm : {0, 2, 4, 17}) {
                    const auto p = dspark_lazy_batches(T, md, lm);
                    const int lim = std::min({T, md > 0 ? md : T, lm > 0 ? lm : T});
                    if (p.empty() || p.front() != 0 || p.back() != lim) inv = false;
                    for (size_t i = 1; i < p.size(); ++i)
                        if (p[i] <= p[i - 1] || p[i] > lim) inv = false;
                }
        check(inv, "over the grid: monotone, first 0, last lim=min(T,max_drafts,lz_max), inside [0,lim]");
    }

    // 10. the drafter's loop, simulated with the production plan and the production cut: where many rows get
    // computed for a block that cuts at a given draft (made never exceeds its batch's endpoint; a block whose
    // very first confidence cuts pays zero head rows - the lazy point)
    {
        const float cmin = 0.5f, cummin = 0.0f;
        auto made_of = [&](int T, const std::vector<float>& w, int* kept) {
            const auto plan = dspark_lazy_batches(T, 0, 0);
            const int lim = T;
            int nc = 0, keep = 0, made = 0;
            float cum = 1.0f;
            for (int b : plan) {
                if (b > made) made = b;
                const int c1 = std::min(b + 1, lim);
                const bool open = keep == nc;
                nc = c1;
                if (open) keep = draft_cut_keep(w.data(), keep, nc, cum, cmin, cummin);
                if (keep < nc || made >= lim) break;
            }
            *kept = keep;
            return made;
        };
        int kept = 0;
        std::vector<float> hi(8, 0.9f);
        check(made_of(8, hi, &kept) == 8 && kept == 8, "no cut: every row of the block is made and kept");
        std::vector<float> c0{0.2f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        check(made_of(8, c0, &kept) == 0 && kept == 0, "draft 0 cut by confidence 0: zero head rows computed at all");
        std::vector<float> c1{0.9f, 0.2f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        check(made_of(8, c1, &kept) == 2 && kept == 1, "cut at draft 1: only rows 0-1 paid for");
        std::vector<float> c2{0.9f, 0.9f, 0.2f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        check(made_of(8, c2, &kept) == 2 && kept == 2, "cut at draft 2: paid for rows 0-1, the confidence caught it");
        std::vector<float> c5(8, 0.9f);
        c5[5] = 0.2f;
        check(made_of(8, c5, &kept) == 8 && kept == 5, "a deep cut pays the last batch whole (it is one batch)");
        bool bound = true;
        for (int T = 1; T <= 16; ++T) {
            for (int cut = 0; cut <= T; ++cut) {
                std::vector<float> w((size_t) T, 0.9f);   // (fresh every pair: a stale 0.2 would move the cut)
                if (cut < T) w[(size_t) cut] = 0.2f;
                const int made = made_of(T, w, &kept);
                // confidence t needs only x_t and draft t-1, so a cut at draft t needs rows 0..t-1: made
                // must cover min(cut, T) rows and can never exceed lim; kept is exactly the survivors
                if (made > T || kept != std::min(cut, T) || made < std::min(cut, T)) bound = false;
            }
        }
        check(bound, "over every cut position: made <= lim, every row the cut needs was made, kept == cut");
    }

    std::printf("ds4_draft_cut_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
