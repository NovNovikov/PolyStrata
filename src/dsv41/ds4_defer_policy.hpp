// Strata. src/dsv41/ds4_defer_policy.hpp - the pure arithmetic of the STRATA_DS4_DEFER decision, shared by the
// engine (ds4_model.cu: the publisher stamps its verdict into every request, the CPU lane obeys that stamp) and
// its synthetic test (tests/dsv41/ds4_defer_policy_test.cpp). No device, no model, no environment: the state that
// feeds these (mode, lead device, lane on, KV rows pending, window size, layer shape, per-entry weights) is the
// caller's, so the invariants can be checked over its whole input space.
#pragma once

namespace strata::ds4 {

// The publisher's verdict for one window, one layer: deferral needs a lead device (a member device's group
// reduces partials; its next layer's forward never looks at this device's mailbox), the layer served by the
// CPU lane (nm == 1: no split model), no KV rows this step writes (wp.nrow: a multi-row step's landing layers
// run interleaved), a window within the mailbox's kWin tokens, and a layer whose landing layers are plain
// trunk layers of the same device and group (the engine's defer_ok: it checks the layer table, passed here
// as layer_ok). False: the window must not split and its lane answers the whole sum, waited for as before.
inline bool defer_window(bool lead, bool cpu_lane, bool kv_rows_pending, int T, int kWin, bool layer_ok) {
    return lead && cpu_lane && !kv_rows_pending && T <= kWin && layer_ok;
}

// What the published verdict obliges the lane to do: split its answer into the heavy part (the Resp the
// publisher waits for) and the light one (the late ring) exactly when the stamp says so AND the mode splits.
// Mode 0 defers nothing, mode 1 defers the lane's whole answer (a split would lose the heavy part's wait);
// only modes 2, 3 and 4 land a light part. The lane must NOT re-read the layer's state for this: the stamp
// is the publisher's word on what its forward can land, and a split nobody consumes loses the light part.
inline bool lane_splits(int defer_flag, int mode) { return defer_flag != 0 && mode >= 2; }

// One entry-slot is light: its route weight strictly under DEFER_W x its token's route sum (a weight exactly
// at the boundary stays heavy - the comparison the engine does).
inline bool defer_light_slot(float w, float tok_route_sum, float dw) { return w < dw * tok_route_sum; }

// An entry (one expert, its slots across the window) is light when every slot is: an entry lands whole, and
// a half-split entry would double-read the expert. w/tok address the entry's first slot; n is its slots.
inline bool defer_light_entry(const float* w, const int* tok, int n, const float* tok_route_sum, float dw) {
    for (int q = 0; q < n; ++q)
        if (!defer_light_slot(w[q], tok_route_sum[tok[q]], dw)) return false;
    return true;
}

// With STRATA_DS4_DEFER_COLD=1, light means cold in every splitting mode (2, 3, 4): the expert the RAM does
// not hold, whatever its weights. (The fingerprint carries cold for every mode >= 2 for the same reason.)
inline bool defer_light_entry_cold(bool in_ram) { return !in_ram; }

// The forward boundary rule for deferred parts: a part belongs to the forward that made it. A part stamped
// with an older forward's epoch is dropped at that forward's entry, never added to a later one's residual -
// it belongs to a stream that a finished, aborted or rolled-back request already gave up on.
inline bool defer_part_still_ours(unsigned part_epoch, unsigned forward_epoch) {
    return part_epoch == forward_epoch;
}

}  // namespace strata::ds4
