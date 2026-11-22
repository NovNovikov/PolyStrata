// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// src/dsv41/ds4_lane_gate.hpp - the CPU lane's failure state, apart from the model (hostile review, third round item 4).
//
// The lane's demand reads answer the device's mailbox. When one fails for real (the store had no slot AND reading
// the files failed too), the service thread used to note the error and leave its loop - permanently. The API failed
// THAT request, but the next run cleared the error flag and walked in: the GPU published a mailbox request that
// nobody would ever answer, and its wait never ended. An infinite wait on a dead thread, dressed as a fresh run.
//
// So the failure state is two things, and they were one:
//   - a REPORT (note()): which read failed, for the current run's error text. Reports clear at every run's start -
//     a copy-thread note belongs to the run that saw it, and the lane answers on.
//   - a DEATH (die()): the service thread left the mailbox with a report outstanding. Nothing can answer requests
//     after it: the state is sticky past every clear, an entry guard refuses later inference BEFORE any GPU work
//     (publishing into a mailbox nobody reads is the hang), and every outcome check still names the reason.
//
// The machine lives here, small and testable, because serve_lane itself needs a model to run:
// tests/dsv41/ds4_lane_gate_test.cpp drives these transitions - including clear/entry racing the death itself, with
// the invariant that a dead lane never reopens - with latches and a watchdog, never sleep-timed guesses.
#pragma once

#include <atomic>
#include <mutex>
#include <string>

namespace strata::ds4 {

class LaneFail {
public:
    void note(const std::string& msg) {   // (the lane's thread, or the store's copy thread) a real read failure;
        std::lock_guard<std::mutex> lk(mu_);   //   the first one's text is the run's error
        if (why_.empty()) why_ = msg;
        set_.store(true, std::memory_order_relaxed);   // (the hot paths check only this relaxed flag)
    }
    void die() {   // (the service thread, on its way out) terminal: sticky past every clear_run, forever
        set_.store(true, std::memory_order_relaxed);
        dead_.store(true, std::memory_order_release);
    }
    bool noted() const { return set_.load(std::memory_order_relaxed); }
    bool dead() const { return dead_.load(std::memory_order_acquire); }
    bool api_ok(std::string& err) const {   // (the API's outcome check, given the call's own success) a run whose
        if (!set_.load(std::memory_order_relaxed)) return true;   //  lane reported must not be handed back as an
        err = "CPU lane: " + why();                                 //  answer, and a dead lane's never can be
        return false;
    }
    bool entry_refused(std::string& err) const {   // (before any work of a new run) the mailbox has nobody left:
        if (!dead_.load(std::memory_order_acquire)) return false;   //  publishing a request would wait on it forever
        err = "CPU lane: permanently stopped (" + why() + ") - the reader thread is gone; reload the model";
        return true;
    }
    void clear_run() {   // (every run's start) a fresh report; a dead lane stays dead and keeps its reason readable
        std::lock_guard<std::mutex> lk(mu_);
        if (dead_.load(std::memory_order_relaxed)) {
            set_.store(true, std::memory_order_relaxed);   // (a racing clear during the death cannot reopen it:
            return;                                        //   api_ok keeps failing every run, dead last)
        }
        why_.clear();
        set_.store(false, std::memory_order_relaxed);
    }
    std::string why() const {
        std::lock_guard<std::mutex> lk(mu_);
        return why_;
    }

private:
    std::atomic<bool> set_{false};
    std::atomic<bool> dead_{false};
    mutable std::mutex mu_;
    std::string why_;
};

}  // namespace strata::ds4
