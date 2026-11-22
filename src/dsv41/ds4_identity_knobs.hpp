// Strata. src/dsv41/ds4_identity_knobs.hpp - the approximate and deferral knobs a session state was computed
// under, and their emission into the session identity string (ds4_model.cu's slot_identity). Shared with the
// synthetic identity test (tests/dsv41/ds4_identity_knobs_test.cpp); the engine passes the values it already
// resolved - this header reads no environment: the caller owns the precedence (DS4_YOCO over
// STRATA_DS4_PROMPT_SKIP), the clamps (yoco_keep, the auto policy's minimums) and the parsing ("0.20" and
// "0.2" must name one state, so a knob arrives parsed, never as a raw string).
//
// The rule each knob follows: whatever changes what a layer adds to the state a session file holds (the KV
// rows, the marks) belongs in the identity, because a file mounted under a different math would feed the
// conversation rows computed another way; a knob at its default, or inert for the mode in use, contributes
// nothing, so a plain run's string stays byte-identical across builds that add knobs.
#pragma once

#include <charconv>
#include <cstdio>
#include <string>

namespace strata::ds4 {

/// Locale-independent, round-trip-exact float text (hostile review, second round 6.1). The stream `<<` on a
/// float prints six significant digits in the PROCESS locale: 0.20000003f and 0.2f both print "0.2" - two
/// different weights share one session identity, and a session file mounted under one answers under the
/// other - and a machine whose C locale prints commas puts "0,2" into a string the loader compares bytewise.
/// std::to_chars gives the shortest decimal that reads back as the same float, always with a dot, whatever
/// the locale: distinct floats get distinct text, one float the same text everywhere.
inline std::string fp_txt(float v) {
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof buf, v);
    if (r.ec == std::errc()) return std::string(buf, r.ptr);
    std::snprintf(buf, sizeof buf, "%.9g", v);   // (to_chars has no failure for finite floats; belt for the odd one)
    return std::string(buf);
}

struct SessionKnobs {
    // Expert Deferral (STRATA_DS4_DEFER family)
    int defer = 0;
    float defer_w = 0.2f;        // modes 2-4: the light share of a token's route sum
    int defer_k = 1;             // mode 4: layers late
    bool defer_cold = false;     // modes 2-4: light means not-in-RAM, weights aside
    bool defer_early = false;    // the part lands at the next layer's entry
    bool defer_max_set = false;  // DEFER_MAX was given at all
    int defer_max = 0;
    // weight-based expert skipping
    float disk_skip_w = 0.0f;    // 0: off
    int disk_skip_from = 0;      // first layer it applies to
    bool disk_skip_auto = false; // only while cold; then its state machine decides per request
    int disk_skip_window = 32;   // the auto policy's knobs, parsed as the engine parses them
    int disk_skip_settle = 64;
    int disk_skip_budget = 512;
    float ram_skip_w = 0.0f;     // 0: off
    // prompt skip / bounded replay: yoco_keep 0 means the skip is not active in this run (off, or the
    // geometry does not allow it), so no row routing changed and nothing is emitted
    int yoco = 0;                // effective mode: 1 exact skip, 2 serving-style bounded replay
    int yoco_keep = 0;           // rows through every layer, after the engine's clamps
    // experts per word: fewer than the model's k_exp drops the smallest picks (approximate);
    // equal (or unknown: either 0) is the model's own math and adds nothing
    int experts_used = 0;
    int k_exp = 0;
};

/// Append the identity section for these knobs. An ostringstream (or anything streamable).
template <class OS>
void emit_session_knobs(OS& id, const SessionKnobs& k) {
    if (k.defer != 0) {
        id << " defer=" << k.defer;
        if (k.defer >= 2) id << " w=" << fp_txt(k.defer_w);   // mode 1 defers the whole lane: W cannot change it
        if (k.defer == 4) id << " k=" << k.defer_k;           // only mode 4 lands the part late
        if (k.defer >= 2 && k.defer_cold) id << " cold";      // every splitting mode, not just 4
        if (k.defer_early) id << " early";
        if (k.defer_max_set) id << " max=" << k.defer_max;
    }
    if (k.disk_skip_w > 0.0f) {
        id << " dsk=" << fp_txt(k.disk_skip_w) << " dsk_from=" << k.disk_skip_from;
        if (k.disk_skip_auto)
            id << " dsk_auto=" << k.disk_skip_window << "," << k.disk_skip_settle << "," << k.disk_skip_budget;
    }
    if (k.ram_skip_w > 0.0f) id << " rsk=" << fp_txt(k.ram_skip_w);
    if (k.yoco != 0 && k.yoco_keep > 0) id << " yoco=" << k.yoco << " keep=" << k.yoco_keep;
    if (k.experts_used > 0 && k.k_exp > 0 && k.experts_used < k.k_exp) id << " kexp=" << k.experts_used;
}

}  // namespace strata::ds4
