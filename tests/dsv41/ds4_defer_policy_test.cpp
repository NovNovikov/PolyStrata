// Strata. tests/dsv41/ds4_defer_policy_test.cpp - the CPU/GPU deferral policy agreement (hostile review 7), on
// src/dsv41/ds4_defer_policy.hpp - the same functions the engine calls (ds4_model.cu: the publisher's verdict
// stamped into Req::defer, the lane's obedience to the stamp, the light/heavy split). The defect this pins
// shut: the lane used to decide with its own reading of the layer, so on a window the publisher had refused
// to defer (KV rows pending, T past kWin, a member device) the light part was still split off, sat unread
// in the late ring, and vanished from the expert sum. No device, no model, no GGUF.
#include "ds4_defer_policy.hpp"

#include <cstdio>
#include <initializer_list>   // (the range-for over {1, kWin, kWin + 1} below needs it; cstdio alone does not bring it)

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

int main() {
    constexpr int kWin = 16;   // the engine's kWin

    // 1. The publisher's verdict: every input that makes a window unable to land a part refuses deferral,
    //    and the fully-able window takes it.
    check(defer_window(true, true, false, 1, kWin, true), "decode on a lead device, plain layer: defers");
    check(!defer_window(false, true, false, 1, kWin, true), "member device: never defers (nobody reads its mailbox)");
    check(!defer_window(true, false, false, 1, kWin, true), "split model (nm > 1): never defers");
    check(!defer_window(true, true, true, 1, kWin, true), "step writing KV rows: never defers");
    check(!defer_window(true, true, false, kWin + 1, kWin, true), "window past kWin: never defers");
    check(defer_window(true, true, false, kWin, kWin, true), "window exactly kWin: defers (the mailbox holds it)");
    check(!defer_window(true, true, false, 1, kWin, false), "layer whose landing layers are not plain: never defers");

    // 2. The lane obeys the stamp: for every combination of window state and mode, the lane splits only
    //    where the publisher deferred AND the mode splits. This is the invariant the lost-light-part bug
    //    broke: the stamp is the single source, the lane has no second opinion.
    for (int lead = 0; lead <= 1; ++lead)
        for (int lane = 0; lane <= 1; ++lane)
            for (int kv = 0; kv <= 1; ++kv)
                for (int T : {1, kWin, kWin + 1})
                    for (int ok = 0; ok <= 1; ++ok)
                        for (int mode = 0; mode <= 4; ++mode) {
                            const bool win = defer_window(lead != 0, lane != 0, kv != 0, T, kWin, ok != 0);
                            const int stamp = win ? 1 : 0;   // what route_publish puts into Req::defer
                            const bool split = lane_splits(stamp, mode);
                            if (split && (!win || mode < 2)) { check(false, "lane split where the publisher did not defer"); return 1; }
                            if (!split && win && mode >= 2) { check(false, "publisher deferred but the lane kept one answer"); return 1; }
                            if (mode < 2 && split) { check(false, "mode 0/1 split (mode 1 defers the whole answer)"); return 1; }
                        }
    check(true, "lane splits exactly on (stamp set) AND (mode >= 2), over all window states x modes");

    // 3. The mode-1 whole-answer rule: the publisher's stamp does not make the lane split; the whole answer
    //    goes into the Resp the next layer waits on.
    check(!lane_splits(1, 1), "mode 1: whole answer, no split, even with the stamp set");
    check(!lane_splits(0, 4), "no stamp: mode 4 keeps one whole answer too");

    // 4. The light classifier: the boundary is strict, an entry lands whole, cold ignores weights.
    check(defer_light_slot(0.19f, 1.0f, 0.2f), "weight under dw x the route sum: light");
    check(!defer_light_slot(0.2f, 1.0f, 0.2f), "weight exactly at dw x the route sum: heavy (strict <)");
    check(!defer_light_slot(0.21f, 1.0f, 0.2f), "weight over: heavy");
    {
        const float w[3] = {0.05f, 0.06f, 0.5f};
        const int tok[3] = {0, 0, 1};
        const float wsum[2] = {1.0f, 1.0f};
        check(defer_light_entry(w, tok, 2, wsum, 0.2f), "entry of two light slots: light");
        check(!defer_light_entry(w, tok, 3, wsum, 0.2f), "one heavy slot: the whole entry stays heavy (no half-read expert)");
        check(defer_light_entry(w, tok, 0, wsum, 0.2f), "an entry with no slots reads as light (nothing to land)");
    }
    check(defer_light_entry_cold(false), "cold: not in RAM = light whatever the weights");
    check(!defer_light_entry_cold(true), "cold: in RAM = heavy whatever the weights");

    // 5. The forward boundary: a part of an earlier forward is dropped, never added to a later one's residual.
    check(defer_part_still_ours(7, 7), "same forward: the part lands");
    check(!defer_part_still_ours(6, 7), "earlier forward's part: dropped at the new forward's entry");
    check(!defer_part_still_ours(8, 7), "a part of a future epoch (a bug elsewhere): not ours either");

    std::printf("ds4_defer_policy_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
