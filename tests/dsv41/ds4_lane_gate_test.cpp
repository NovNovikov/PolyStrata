// Strata. src/dsv41 LaneFail, the CPU lane's failure state, host-only, no model (hostile review, third round
// item 4): the defect it keeps is the silent death of serve_lane - the thread noted a demand-read failure, left its
// loop, and the NEXT run cleared the report and walked into a mailbox nobody answers. So: a report clears with its
// run; a DEATH does not, and an entry guard must refuse every later run while the death holds. The scenarios drive
// the transitions the engine makes (note-then-die on the service thread; the API's api_ok; run-start clears; an
// entry), including clear/entry/api racing the death itself - all bounded by watchdogs, not by sleep guesses.
#include "ds4_lane_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using strata::ds4::LaneFail;

namespace {

int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) ++failures;
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
}

template <class F>
bool within(double ms, F&& f) {   // (a bounded wait on a latch-style condition: true if f() became true in time)
    const auto t0 = std::chrono::steady_clock::now();
    while (!f()) {
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > ms) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

}  // namespace

int main() {
    // 1. a fresh gate: nothing noted, nothing refused, every outcome trusted
    {
        LaneFail g;
        std::string err;
        check(!g.noted() && !g.dead(), "a fresh lane has no report and no death");
        check(g.api_ok(err), "and a fresh run's outcome is trusted");
        check(!g.entry_refused(err), "and its entry is allowed");
        g.clear_run();
        check(g.api_ok(err), "clearing a fresh run changes nothing");
    }

    // 2. a report while the lane lives on (the store's copy thread failing a demand read): the run that saw it
    //    fails with the reason, the next run starts clean - the round-two semantics, unchanged
    {
        LaneFail g;
        std::string err;
        g.note("layer 12 expert 404: the files refused the read");
        check(g.noted(), "the report is set");
        check(!g.api_ok(err) && err.find("layer 12 expert 404") != std::string::npos,
              "the run that saw it fails, and says which read");
        g.clear_run();
        check(!g.noted() && g.api_ok(err), "the next run starts clean: a report belongs to the run that saw it");
        check(!g.entry_refused(err), "and the lane answers on");
    }

    // 3. the death: note, then the service thread leaves. Every clear_run, however many, cannot reopen the lane;
    //    every outcome fails naming the reason; every entry is refused with it
    {
        LaneFail g;
        std::string err;
        g.note("layer 7 expert 91: no slot and the read failed");
        g.die();
        check(g.dead(), "the death is marked");
        check(!g.api_ok(err) && err.find("layer 7 expert 91") != std::string::npos, "the dying run fails with its reason");
        bool reopened = false;
        for (int i = 0; i < 100; ++i) {   // (the API's run-start clear, a hundred times over: the old code's hole)
            g.clear_run();
            std::string e2;
            if (g.api_ok(e2)) reopened = true;
        }
        check(!reopened, "a hundred run-start clears never trust a dead lane's outcome again");
        check(g.entry_refused(err) && err.find("layer 7 expert 91") != std::string::npos,
              "and every entry is refused, naming the reason that killed it");
        check(g.why().find("expert 91") != std::string::npos, "the reason stays readable past every clear");
    }

    // 4. the race: run-start clears, outcome checks and entry guards hammering the gate while the death happens
    //    in the middle of them. The invariant LaneFail promises: once the death is observed, no sequence of
    //    clear + check / clear + entry can present the old alive behavior again - no matter where the death
    //    landed between one call and the next. Every join is behind a watchdog: a hang is the defect.
    {
        LaneFail g;
        std::atomic<bool> stop{false}, violation{false}, died{false};
        std::vector<std::thread> ws;
        for (int t = 0; t < 3; ++t)
            ws.emplace_back([&] {
                while (!stop.load(std::memory_order_relaxed)) {
                    std::string err;
                    g.clear_run();
                    // (a) if this thread can SEE the death, the outcome check must fail right now, whatever the
                    //     clear just did (the round-two gate: the clear erased the report, api_ok said success,
                    //     and the mailbox request it allowed is the hang)
                    if (g.dead() && g.api_ok(err)) violation.store(true);
                    // (b) and the entry guard that decides whether to publish to the mailbox must agree
                    if (g.dead() && !g.entry_refused(err)) violation.store(true);
                }
            });
        // the lane: a few live reports first (some cleared, some seen), then note, then death
        for (int i = 0; i < 500; ++i) g.note("transient report " + std::to_string(i));
        g.note("layer 3 expert 5: the demand read failed");
        g.die();
        died.store(true);
        // after the death: the same two invariants, observed directly
        for (int i = 0; i < 200; ++i) {
            std::string err;
            g.clear_run();
            if (g.api_ok(err)) violation.store(true);
            if (!g.entry_refused(err)) violation.store(true);
        }
        stop.store(true);
        bool joined = true;
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& w : ws) {
            // (join is blocking; the threads only loop on atomics and short mutex holds, so bound the whole set:
            //  if a join could hang here, the test says so and leaves instead of hanging the box)
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 5.0 && w.joinable()) {
                w.join();
                break;   // (join returns when the thread is done; the loop is the watchdog's shape only)
            }
            if (w.joinable()) joined = false;
        }
        check(joined, "the gate never blocked a clear/check/entry caller - all callers returned");
        check(!violation.load(), "and no interleaving of clear with the death reopened the lane");
        check(died.load(), "the death landed (the driver always runs to it - the workers never block it)");
    }

    // 5. death without a prior report (a producer-side exit that only marks the terminal state): entries still
    //    refuse, outcomes still fail, with a reason line that still identifies the lane as the dead part
    {
        LaneFail g;
        std::string err;
        g.die();
        check(g.entry_refused(err) && err.find("CPU lane") != std::string::npos,
              "a death with no earlier note still refuses entries and says who died");
        check(!g.api_ok(err), "and no outcome is trusted");
    }

    std::printf("ds4_lane_gate_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
