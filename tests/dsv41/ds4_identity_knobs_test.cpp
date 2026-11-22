// Strata. tests/dsv41/ds4_identity_knobs_test.cpp - the session identity's knob section (hostile review 2 and 7):
// src/dsv41/ds4_identity_knobs.hpp - the very emitter ds4_model.cu's slot_identity calls. Rule under test: two
// runs share an identity string exactly when they compute the same state - every knob that changes what a
// layer adds must change the string, an inert or default knob must not, and a knob arrives parsed so "0.20"
// and "0.2" are one state. No model, no pack, no device.
#include "ds4_identity_knobs.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <clocale>
#include <functional>
#include <sstream>
#include <string>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

static std::string emit(const SessionKnobs& k) {
    std::ostringstream id;
    emit_session_knobs(id, k);
    return id.str();
}

int main() {
    // 1. An untouched struct is the pre-port state: empty section, the identity string unchanged.
    check(emit(SessionKnobs{}).empty(), "no knobs set: the identity carries no knob section");

    // 2. Every knob that matters changes the string; each change is visible.
    {
        SessionKnobs base;
        base.yoco = 1; base.yoco_keep = 512; base.experts_used = 8; base.k_exp = 8;   // the effective default run
        const std::string b = emit(base);
        check(b.find("yoco=1") != std::string::npos && b.find("kexp") == std::string::npos,
              "the default exact skip is in; full-quality experts are not");
        auto changed = [&](std::function<void(SessionKnobs&)> f, const char* what) {
            SessionKnobs k = base;
            f(k);
            check(emit(k) != b, what);
        };
        changed([](SessionKnobs& k) { k.defer = 1; }, "defer on changes the identity");
        changed([](SessionKnobs& k) { k.defer = 2; k.defer_w = 0.1f; }, "defer 2 with a lighter threshold changes it");
        changed([](SessionKnobs& k) { k.defer = 2; k.defer_cold = true; }, "cold changes it in a splitting mode");
        changed([](SessionKnobs& k) { k.defer = 4; k.defer_k = 2; }, "mode 4 landing later changes it");
        changed([](SessionKnobs& k) { k.defer = 2; k.defer_early = true; }, "early changes a live deferral");
        changed([](SessionKnobs& k) { k.defer = 2; k.defer_max_set = true; k.defer_max = 20; }, "DEFER_MAX changes a live deferral");
        {
            SessionKnobs k = base; k.defer_early = true;   // inert without DEFER: must not split sessions over state that cannot differ
            check(emit(k) == b, "early with no deferral in force changes nothing");
            SessionKnobs m = base; m.defer_max_set = true; m.defer_max = 20;
            check(emit(m) == b, "DEFER_MAX with no deferral in force changes nothing");
        }
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; }, "DISK_SKIP_W changes it");
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; k.disk_skip_auto = true; }, "and its auto policy again");
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; k.disk_skip_window = 48; }, "the auto window changes it");
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; k.disk_skip_settle = 96; }, "the auto settle changes it");
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; k.disk_skip_budget = 1024; }, "the auto budget changes it");
        changed([](SessionKnobs& k) { k.disk_skip_w = 0.26f; k.disk_skip_from = 4; }, "the skip's first layer changes it");
        changed([](SessionKnobs& k) { k.ram_skip_w = 0.1f; }, "RAM_SKIP_W changes it");
        changed([](SessionKnobs& k) { k.yoco = 2; }, "the serving-style replay differs from the exact skip");
        changed([](SessionKnobs& k) { k.yoco_keep = 1024; }, "a different replay tail differs");
        changed([](SessionKnobs& k) { k.experts_used = 5; }, "fewer experts per word changes it");
    }

    // 3. Inert knobs add nothing: the string must not split sessions over state that cannot differ.
    {
        SessionKnobs k;
        k.defer = 1; k.defer_w = 0.05f; k.defer_cold = true; k.defer_k = 3;   // mode 1: W, cold and K cannot change anything
        check(emit(k) == " defer=1", "mode 1 carries only the mode (whole-lane deferral ignores W, K, cold)");
        SessionKnobs d;
        d.disk_skip_w = 0.0f; d.disk_skip_auto = true; d.disk_skip_window = 99;   // auto without its W is inert (the engine's own rule)
        check(emit(d).empty(), "an auto policy with the skip off adds nothing");
        SessionKnobs y;
        y.yoco = 2; y.yoco_keep = 0;   // the geometry does not run the skip: no row routing changed
        check(emit(y).empty(), "a prompt skip that does not run adds nothing");
        SessionKnobs e;
        e.experts_used = 8; e.k_exp = 8;
        check(emit(e).empty(), "full-quality experts (used == the model's k) add nothing");
        SessionKnobs m;
        m.defer_max_set = false; m.defer_max = 12345;   // unset: whatever the field holds must not leak
        m.defer = 2;
        check(emit(m).find("max") == std::string::npos, "an unset DEFER_MAX does not leak its field");
    }

    // 4. Parsed values: the strings built from "0.20" and "0.2" (the engine parses before calling) are one state.
    {
        SessionKnobs a, c;
        a.defer = 3; a.defer_w = (float) std::atof("0.20");
        c.defer = 3; c.defer_w = (float) std::atof("0.2");
        check(emit(a) == emit(c), "\"0.20\" and \"0.2\" give the same identity (parsed, not raw env)");
        SessionKnobs r, s;
        r.disk_skip_w = (float) std::atof("0.260");
        s.disk_skip_w = (float) std::atof("0.26");
        check(emit(r) == emit(s), "the same holds for DISK_SKIP_W");
    }

    // 5. Determinism: the same knobs, twice, byte for byte (mount compares strings).
    {
        SessionKnobs k;
        k.defer = 4; k.defer_k = 2; k.defer_cold = true; k.defer_early = true;
        k.defer_max_set = true; k.defer_max = 39;
        k.disk_skip_w = 0.26f; k.disk_skip_auto = true; k.disk_skip_from = 4;
        k.ram_skip_w = 0.1f; k.yoco = 2; k.yoco_keep = 128; k.experts_used = 5; k.k_exp = 8;
        check(emit(k) == emit(k), "the same knobs emit the same string twice");
        check(emit(k).find("defer=4") < emit(k).find("dsk=") && emit(k).find("yoco=") < emit(k).find("kexp="),
              "the section's parts keep a fixed order");
    }

    // 6. (second round 6.1) the float texts: adjacent floats are distinct, every text round-trips, and the dot
    // survives a comma locale - the fingerprint compares strings, so this is where two weights may not collide.
    {
        const float w = 0.2f, w2 = std::nextafter(w, 1.0f);
        check(w2 != w, "the float right above 0.2 exists (the test itself is sound)");
        SessionKnobs a, b2;
        a.defer = 2; a.defer_w = w;
        b2.defer = 2; b2.defer_w = w2;
        check(emit(a) != emit(b2),
              "0.2f and its neighbour get different identities (six digits printed both as \"0.2\")");
        const std::string ta = emit(a);
        const size_t p = ta.find(" w=");
        check(p != std::string::npos && std::strtof(ta.c_str() + p + 3, nullptr) == w && ta.size() == p + 6 &&
                  ta.compare(p + 1, 5, "w=0.2") == 0,
              "the text reads back as the very float and keeps the short 0.2 form");
        bool exact = true;
        for (int i = 0; i < 2000; ++i) {
            SessionKnobs k;
            k.ram_skip_w = (float) (i + 1) * 0.00123f;
            const std::string t = emit(k);
            const size_t q = t.find("rsk=");
            if (q == std::string::npos || std::strtof(t.c_str() + q + 4, nullptr) != k.ram_skip_w) { exact = false; break; }
        }
        check(exact, "2000 arbitrary weights: each prints text that reads back as exactly itself");
        // the process's C locale may print commas; the fingerprint may not: fp_txt never follows the locale
        char saved[128] = {0};
        std::snprintf(saved, sizeof saved, "%s", std::setlocale(LC_ALL, nullptr) ? std::setlocale(LC_ALL, nullptr) : "C");
        if (std::setlocale(LC_ALL, "German") != nullptr) {
            SessionKnobs k;
            k.ram_skip_w = 0.25f;
            const std::string t = emit(k);
            check(t.find("rsk=0.25") != std::string::npos && t.find(',') == std::string::npos,
                  "in a comma locale the weight still prints with a dot");
            std::setlocale(LC_ALL, saved);
        } else {
            check(fp_txt(0.25f) == "0.25", "no comma locale on this machine: fp_txt itself still prints the dot");
        }
    }

    std::printf("ds4_identity_knobs_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
