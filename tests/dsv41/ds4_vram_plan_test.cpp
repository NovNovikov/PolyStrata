// Strata. ds4_vram_plan.hpp on made-up layer shapes: the DS4_MTP_VRAM_GB split (the 19cbf656ab port) checked
// without a device or a model, as the task spec's 8.2 asks - the arithmetic never allocates anything. The
// production function plan_group_vram is what the expert loader calls; the cases cover the drafter's budget
// distribution, cap compliance through every stage, the freed room reaching the trunk, zero / minimal / middle /
// unlimited budgets, no-parameter behaviour, cpu_only layers, and full fits.
#include "ds4_vram_plan.hpp"

#include <cstdio>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

static double used_bytes(const std::vector<VramPlanLayer>& p) {
    double u = 0;
    for (const VramPlanLayer& l : p) u += (double) l.n_vram * l.unit;
    return u;
}
static int mtp_slots(const std::vector<VramPlanLayer>& p) {
    int n = 0;
    for (const VramPlanLayer& l : p)
        if (l.mtp) n += l.n_vram;
    return n;
}
static int trunk_slots(const std::vector<VramPlanLayer>& p) {
    int n = 0;
    for (const VramPlanLayer& l : p)
        if (!l.mtp && !l.cpu_only) n += l.n_vram;
    return n;
}

// a group: 3 trunk layers of 8 experts at 100 MB, 2 drafter layers of 8 experts at 200 MB
static std::vector<VramPlanLayer> group() {
    std::vector<VramPlanLayer> p;
    for (int i = 0; i < 3; ++i) p.push_back({false, false, 8, 100u << 20, 0});
    for (int i = 0; i < 2; ++i) p.push_back({true, false, 8, 200u << 20, 0});
    return p;
}
static const double G100 = 100.0 * (1 << 20);

int main() {
    // 1. no cap: the drafter takes its experts first, greedily, and the trunk splits what is left
    {
        auto p = group();
        plan_group_vram(p, 3.0 * G100, kNoCap);   // 300 MB: the first drafter layer eats all it can (one 200 MB slot)
        check(p[3].n_vram == 1, "uncapped: the first drafter layer takes what fits first (1 of 8 experts)");
        check(p[4].n_vram == 0 && trunk_slots(p) == 1, "uncapped: the drafter is served first, the trunk only gets the 100 MB left");
        check(used_bytes(p) <= 3.0 * G100, "uncapped: the budget holds");
    }
    // 2. a cap splits evenly and the freed room goes to the trunk
    {
        auto p = group();
        plan_group_vram(p, 3.0 * G100, 100.0 * (1 << 20));   // 100 MB per drafter layer
        check(p[3].n_vram == 0 && p[4].n_vram == 0, "cap below one slot: the drafter keeps none (streams like the trunk)");
        check(trunk_slots(p) == 3, "the freed 300 MB went to the trunk (3 of 24 trunk experts)");
        check(used_bytes(p) <= 3.0 * G100, "the budget holds with the cap");
        auto q = group();
        plan_group_vram(q, 20.0 * G100, 400.0 * (1 << 20));   // roomy: both drafter layers take exactly the cap
        check(q[3].n_vram == 2 && q[4].n_vram == 2, "roomy budget with a cap: each drafter layer stops at its cap (2 experts x 200 MB)");
        check(used_bytes(q) <= 20.0 * G100 && trunk_slots(q) > trunk_slots(p), "trunk gets everything the capped drafter left");
    }
    // 3. the cap is never passed by a later stage: a tiny per-layer cap under a huge budget
    {
        auto p = group();
        plan_group_vram(p, 50.0 * G100, 50.0 * (1 << 20));   // a slot is 200 MB, the cap is 50 MB: drafter stays at 0
        check(mtp_slots(p) == 0, "the fill stage never grows a capped drafter layer");
        check(trunk_slots(p) == 24, "with the drafter capped out the trunk fills the whole group budget");
    }
    // 4. budgets: zero, one slot, everything
    {
        auto z = group();
        plan_group_vram(z, 0.0, kNoCap);
        check(used_bytes(z) == 0, "zero budget: nothing is planned");
        auto one = group();
        plan_group_vram(one, 200.0 * (1 << 20), kNoCap);
        check(mtp_slots(one) == 1 && used_bytes(one) <= 200.0 * (1 << 20), "one-slot budget: the drafter, being first, takes it");
        auto all = group();
        plan_group_vram(all, 1000.0 * G100, kNoCap);
        bool full = true;
        for (const VramPlanLayer& l : all)
            if (l.n_vram != l.n_exp) full = false;
        check(full, "an unlimited budget seats every expert in VRAM");
        auto ac = group();
        plan_group_vram(ac, 1000.0 * G100, 400.0 * (1 << 20));
        check(ac[3].n_vram == 2 && ac[4].n_vram == 2, "an unlimited budget with a cap: the drafter still stops at the cap");
        int trunk_full = true;
        for (int i = 0; i < 3; ++i)
            if (ac[i].n_vram != 8) trunk_full = false;
        check(trunk_full && used_bytes(ac) <= 1000.0 * G100, "the capped drafter's room went to the trunk, fully");
    }
    // 5. cpu_only layers get nothing in any shape; a group with no drafter ignores the cap
    {
        std::vector<VramPlanLayer> p;
        p.push_back({false, true, 8, 100u << 20, 0});
        p.push_back({false, false, 8, 100u << 20, 0});
        plan_group_vram(p, 16.0 * G100, 800.0 * (1 << 20));
        check(p[0].n_vram == 0, "a cpu_only layer gets no slots");
        check(p[1].n_vram == 8, "the rest takes the budget as usual");
        std::vector<VramPlanLayer> q;
        q.push_back({false, false, 4, 100u << 20, 0});
        plan_group_vram(q, 2.0 * G100, 1.0 * (1 << 20));
        check(q[0].n_vram == 2, "a cap with no drafter layers in the group changes nothing for the trunk");
    }
    // 6. rounding: fractional shares never pass the budget (odd units, odd budgets)
    {
        for (int mb = 1; mb <= 40; ++mb) {
            auto p = group();
            const double b = mb * 33.0 * (1 << 20);
            plan_group_vram(p, b, mb == 0 ? kNoCap : 330.0 * (1 << 20));
            if (used_bytes(p) > b + 1e-9) { check(false, "some budget overfull"); return 1; }
            for (const VramPlanLayer& l : p)
                if (l.n_vram < 0 || l.n_vram > l.n_exp) { check(false, "a planned count outside [0, n_exp]"); return 1; }
        }
        check(true, "20 odd budgets: never over budget, every count inside [0, n_exp]");
    }
    std::printf("ds4_vram_plan_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
