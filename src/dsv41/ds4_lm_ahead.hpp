// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// src/dsv41/ds4_lm_ahead.hpp - the STRATA_DS4_LM_AHEAD read-ahead's slot book, split from the engine so the reuse
// rules the read-ahead must obey are testable without a model (hostile review 6).
//
// LM_AHEAD keeps a second layer-cache-sized buffer (la) beside the layer cache (lc): while a layer's later chunks
// compute, the NEXT layer's likely experts are read into la at low priority; the next layer's first item adopts la
// as its cache and those reads are hits. That next item may never come: the layer can be skipped (a YOCO cut) or
// served entirely from RAM/VRAM. The buffer must then be reused for another target - and reusing it while the old
// target's reads are still in flight mixes two experts' bytes in one slot and breaks the read counts. Hence the
// rule below: a side is drained before a pass begins on it, always.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace strata::ds4 {

/// One buffer's speculative-read slots: each slot counts its read's parts still in flight (3: gate, up, down) and
/// says whether any part failed.
struct PfSide {
    void init(size_t n) {
        n_ = n;
        left.reset(new std::atomic<int>[n]);
        bad.reset(new std::atomic<bool>[n]);
        for (size_t i = 0; i < n; ++i) {
            left[i].store(0, std::memory_order_relaxed);
            bad[i].store(false, std::memory_order_relaxed);
        }
    }
    size_t slots() const { return n_; }
    std::atomic<int>* left_of(size_t s) { return &left[s]; }
    std::atomic<bool>* bad_of(size_t s) { return &bad[s]; }
    bool landed(size_t s) const { return left[s].load(std::memory_order_acquire) == 0; }
    bool failed(size_t s) const { return bad[s].load(std::memory_order_acquire); }
    /// A slot a pass is about to write: fresh counters, then post its reads with left_of/bad_of.
    void begin(size_t s) {
        left[s].store(3, std::memory_order_relaxed);
        bad[s].store(false, std::memory_order_relaxed);
    }
    /// The reuse boundary: return only when every read this side ever posted has landed. wait() does the
    /// waiting's sleeping (the engine waits on its IO queue); it must return periodically, not spin forever.
    /// A failed read counts as landed - its slot is dead weight the new pass overwrites, which is the point.
    template <class WaitFn> void drain(WaitFn&& wait) {
        for (size_t s = 0; s < n_; ++s)
            while (left[s].load(std::memory_order_acquire) != 0) wait();
    }

private:
    std::unique_ptr<std::atomic<int>[]> left;
    std::unique_ptr<std::atomic<bool>[]> bad;
    size_t n_ = 0;
};

/// A speculative pass's book: the layer it read ahead for, the expert -> slot map, the slots taken. The layer's
/// first item adopts the pass (is_for(layer)); when that item never comes, the pass simply gets begin() again -
/// after the side's drain, which is the caller's (PfSide::drain).
struct LmAheadPass {
    std::vector<int> of;   // expert -> its slot in the buffer, -1: none
    int il = -1;
    int n = 0;

    void init(size_t n_exp) {
        of.assign(n_exp, -1);
        il = -1;
        n = 0;
    }
    void begin(int layer) {
        il = layer;
        n = 0;
        for (int& x : of) x = -1;
    }
    void assign(int expert, int slot) {
        of[(size_t) expert] = slot;
        ++n;
    }
    bool is_for(int layer) const { return il == layer; }
    int taken() const { return n; }
};

}  // namespace strata::ds4
