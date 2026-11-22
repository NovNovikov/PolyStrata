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
        fixing.reset(new std::atomic<bool>[n]);
        cancel.reset(new std::atomic<bool>[n]);
        for (size_t i = 0; i < n; ++i) {
            left[i].store(0, std::memory_order_relaxed);
            bad[i].store(false, std::memory_order_relaxed);
            fixing[i].store(false, std::memory_order_relaxed);
            cancel[i].store(false, std::memory_order_relaxed);
        }
    }
    size_t slots() const { return n_; }
    std::atomic<int>* left_of(size_t s) { return &left[s]; }
    std::atomic<bool>* bad_of(size_t s) { return &bad[s]; }
    /// (hostile review, third round 9) the pass's queued speculative reads carry this flag: when the target's
    /// bytes are no longer wanted by anyone (the pass is being replaced, or the run has ended), setting it makes
    /// every job NOT YET STARTED drop unread - it would read bytes nobody adopts, over the link, while a real
    /// demand waits. A read already running completes: the reuse boundary's drain counts it either way. Safe
    /// to reset at begin(): the engine's invariant is drain-before-reuse (PfSide::drain ran first), so no job
    /// of the previous pass is still in the queue to see a reset flag and run against the new pass's slot.
    std::atomic<bool>* cancel_of(size_t s) { return &cancel[s]; }
    void discard_queued(size_t s) { cancel[s].store(true, std::memory_order_release); }
    void discard_all() {
        for (size_t i = 0; i < n_; ++i) cancel[i].store(true, std::memory_order_release);
    }
    bool landed(size_t s) const { return left[s].load(std::memory_order_acquire) == 0; }
    bool failed(size_t s) const { return bad[s].load(std::memory_order_acquire); }
    /// A slot a pass is about to write: fresh counters, then post its reads with left_of/bad_of.
    void begin(size_t s) {
        left[s].store(3, std::memory_order_relaxed);
        bad[s].store(false, std::memory_order_relaxed);
        fixing[s].store(false, std::memory_order_relaxed);
        cancel[s].store(false, std::memory_order_relaxed);
    }
    /// The reuse boundary: return only when every read this side ever posted has landed. wait() does the
    /// waiting's sleeping (the engine waits on its IO queue); it must return periodically, not spin forever.
    /// A failed read counts as landed - its slot is dead weight the new pass overwrites, which is the point.
    template <class WaitFn> void drain(WaitFn&& wait) {
        for (size_t s = 0; s < n_; ++s)
            while (left[s].load(std::memory_order_acquire) != 0) wait();
    }

    /// Demand recovery of a slot whose speculative read failed (hostile review, second round 5.1; third round 6
    /// closes the winner's stale check). The failed guess's bytes may be partial, so the slot must be re-read
    /// before anything runs on it - but by ONE re-reader only: several items (chunks) pick the same guessed
    /// expert, and each re-reading the slot on its own means several writers over bytes another item's copy is
    /// already reading. exchange() picks exactly one: that caller runs read_slot() (which writes the whole slot
    /// and answers whether the bytes arrived) and RESTORES the slot's state - on success the bad flag goes clear,
    /// so every other and every later reader of this slot sees a plain cache hit and nobody re-reads again; on
    /// failure the bad flag stays, the slot never reads as good, and all callers report false. The ordering:
    /// finish clears bad BEFORE clearing fixing (both release); a waiter whose acquire-load of fixing sees false
    /// has synchronised with that store and reads the matching bad - never a half-restored "good".
    ///
    /// The third round's hole: a caller that saw failed() at the top and then paused - A recovered the slot
    /// SUCCESSFULLY meanwhile - would win the exchange next (A had already cleared fixing) and start a second
    /// whole-slot write under the readers of A's bytes. Winning the fix says nothing about the state being unfixed
    /// anymore, so the winner re-checks it before writing (recover_owned's first line): a slot someone restored is
    /// taken as the hit it now is. The synchronisation for that re-check is the exchange itself: reading fixing's
    /// false after A set and cleared it is a read-from of A's release store, so A's slot bytes are visible here.
    template <class ReadSlot>
    static bool recover_owned(PfSide& side, size_t s, ReadSlot&& read_slot) {   // (the slot is OURS to fix: this is
        if (!side.failed(s)) {                                         //   the only writer's path; the stale winner
            side.fixing[s].store(false, std::memory_order_release);    //   walks in with a slot someone else already
            return true;                                               //   restored - take it as the hit it is)
        }
        const bool got = read_slot();
        if (got) side.bad[s].store(false, std::memory_order_release);
        side.fixing[s].store(false, std::memory_order_release);
        return got;
    }
    template <class ReadSlot, class WaitFn>
    static bool recover(PfSide& side, size_t s, ReadSlot&& read_slot, WaitFn&& wait) {
        if (!side.failed(s)) return true;
        if (!side.fixing[s].exchange(true, std::memory_order_acq_rel))   // the one re-reader (may not need one:
            return recover_owned(side, s, read_slot);                    //   see recover_owned)
        while (side.fixing[s].load(std::memory_order_acquire)) wait();
        return !side.failed(s);   // the recovery that ran restored the slot: this item takes it as a hit
    }

private:
    std::unique_ptr<std::atomic<int>[]> left;
    std::unique_ptr<std::atomic<bool>[]> bad;
    std::unique_ptr<std::atomic<bool>[]> fixing;   // a demand re-read owns the slot while set
    std::unique_ptr<std::atomic<bool>[]> cancel;   // the pass gave up on this slot: queued jobs drop unread
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
