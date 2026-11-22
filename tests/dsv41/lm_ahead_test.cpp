// Strata. src/dsv41 the LM_AHEAD read-ahead's reuse rules, no model (hostile review 6). PfSide and LmAheadPass
// (ds4_lm_ahead.hpp) drive real IoQueue reads over a synthetic expert file: a buffer reused for a NEW read-ahead
// target must first see the old target's reads land - the old target may be a layer that never ran (a YOCO skip,
// a RAM-served layer: its adoption never came), and without the drain the old reads land in slots that now
// promise another expert's bytes. The failed slot's story too: a failed speculative read is a dead slot, the
// engine's answer is to read it again on demand (its flow_pump fallback), and the slot must then carry its file's
// bytes. No GGUF, no device, no model.
#include "ds4_lm_ahead.hpp"
#include "ds4_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "  FAIL: %s\n", what); }
}

static uint8_t want_at(uint64_t i) { return (uint8_t) (i * 37u + 11u); }

static void post_expert(IoQueue& io, PfSide& side, size_t slot, os::fd_t fd, uint64_t off, uint64_t len, uint8_t* dst) {
    side.begin(slot);
    IoQueue::Job jobs[3];   // (the engine reads gate|up|down; one part per job is the whole bookkeeping story)
    for (int r = 0; r < 3; ++r) {
        jobs[r] = IoQueue::Job{};
        jobs[r].fd = fd;
        jobs[r].off = off + r * (len / 3);
        jobs[r].len = len / 3;
        jobs[r].dst = dst + r * (len / 3);
        jobs[r].exact = true;   // (dst and the reads land exactly; no slack around them)
        jobs[r].left = side.left_of(slot);
        jobs[r].bad = side.bad_of(slot);
    }
    io.push(jobs, 3, true);   // (low: a guess, behind every demand read - as the engine pushes them)
}

int main() {
    const char* tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
    const std::string path = std::string(tmp) + "\\ds4_lm_ahead.bin";
    const uint64_t expert = 16u << 20;   // big enough to still be in flight at the reuse point
    const uint64_t total = expert * 4;
    {
        std::vector<uint8_t> data((size_t) total);
        for (uint64_t i = 0; i < total; ++i) data[(size_t) i] = want_at(i);
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { std::printf("lm_ahead_test: cannot write %s\n", path.c_str()); return 2; }
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }
    std::vector<uint8_t> slotbuf((size_t) expert, 0);

    IoQueue io(2);
    const os::fd_t fd = os::open_par(path);
    check(fd != os::kBadFd, "the expert file opens");

    // 1. the pass book: a pass targets a layer; a skipped layer's pass is never adopted; a new pass starts clean
    {
        LmAheadPass p;
        p.init(64);
        check(!p.is_for(5) && p.taken() == 0, "an empty pass belongs to no layer");
        p.begin(5);
        p.assign(7, 0);
        p.assign(9, 1);
        check(p.is_for(5) && !p.is_for(6), "the pass knows its layer (the adoption asks is_for(layer))");
        check(p.of[7] == 0 && p.of[9] == 1 && p.of[8] == -1 && p.taken() == 2, "the expert -> slot map");
        p.begin(8);   // (the layer-5 pass was skipped: its buffer is retargeted - after PfSide::drain, section 2)
        check(!p.is_for(5) && p.of[7] == -1 && p.taken() == 0, "the retarget cleared the skipped pass's book");
        p.begin(-1);
        check(!p.is_for(0) && !p.is_for(7), "the end-of-run reset leaves no adoptable pass");
    }

    // 2. the drain is a real boundary: spec reads in flight; a new pass on the side may only begin when they land
    {
        PfSide side;
        side.init(4);
        post_expert(io, side, 0, fd, 2 * expert, expert, slotbuf.data());   // expert 2, three jobs in flight
        side.drain([&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
        check(side.landed(0), "the drain returned only after the speculative reads landed");
        bool bytes = true;
        for (uint64_t i = 0; i < expert; i += 4099)
            if (slotbuf[(size_t) i] != want_at(2 * expert + i)) bytes = false;
        check(bytes, "and the slot really holds the read-ahead's expert when the drain returns");
        // reuse for another expert (the skipped-target shape): the drain ran, so the slot may be rewritten
        post_expert(io, side, 0, fd, 1 * expert, expert, slotbuf.data());
        side.drain([&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
        bool bytes2 = true;
        for (uint64_t i = 0; i < expert; i += 4099)
            if (slotbuf[(size_t) i] != want_at(1 * expert + i)) bytes2 = false;
        check(bytes2, "the reuse's expert owns the slot whole - the previous pass could no longer land in it");
    }

    // 3. a failed speculative read: the slot reports landed AND failed (a dead slot, not a hang), and the
    // engine's answer - PfSide::recover, the production demand re-read - restores the slot's state and bytes
    {
        PfSide side;
        side.init(2);
        post_expert(io, side, 0, os::kBadFd, 3 * expert, expert, slotbuf.data());   // both handles dead: the read fails
        side.drain([&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
        check(side.landed(0), "a failed speculative read counts as landed: no waiter hangs on it");
        check(side.failed(0), "and says so: the engine's pump sees pf_bad and calls for a demand re-read");
        int rereads = 0;
        auto good_read = [&](uint64_t off) {
            return [&] {
                ++rereads;
                const os::fd_t sfd = os::open_ro(path);   // (a plain handle, the shape the engine's read_at uses)
                bool demand = sfd != os::kBadFd;
                for (int r = 0; r < 3 && demand; ++r)
                    demand = os::pread_full(sfd, slotbuf.data() + r * (expert / 3), expert / 3, off + r * (expert / 3));
                if (sfd != os::kBadFd) os::close_fd(sfd);
                return demand;
            };
        };
        const bool got = PfSide::recover(side, 0, good_read(3 * expert), [] { std::this_thread::sleep_for(std::chrono::microseconds(100)); });
        check(got && rereads == 1, "the production recovery re-reads the slot once and reports it landed");
        check(!side.failed(0), "and restores the slot's state: the bad flag is clear, the slot is a plain hit now");
        bool bytes = true;
        for (uint64_t i = 0; i < expert; i += 4099)
            if (slotbuf[(size_t) i] != want_at(3 * expert + i)) bytes = false;
        check(bytes, "and the item runs on the slot's expert, not a failed guess - the request is not killed");
        const bool again = PfSide::recover(side, 0, good_read(3 * expert), [] {});
        check(again && rereads == 1, "a second call on the restored slot reads nothing: later items take the hit");
    }

    // 4. (second round 5.1) the single-writer promise, driven for real: eight items pick the same guessed slot,
    // all find it failed at once. One re-reads; the rest wait and take the restored bytes. The bytes must never
    // have more than one writer - another item's copy reads them while the recovery runs.
    {
        PfSide side;
        side.init(2);
        post_expert(io, side, 0, os::kBadFd, 2 * expert, expert, slotbuf.data());
        side.drain([&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
        check(side.failed(0), "the slot is failed before the eight ask");
        std::atomic<int> readers{0}, slow_reads{0};
        auto recover_one = [&](bool works) {
            return PfSide::recover(
                side, 0,
                [&] {
                    ++slow_reads;
                    (void) works;   // (the successful shape: the read below is the real one)
                    const os::fd_t sfd = os::open_ro(path);
                    bool ok = sfd != os::kBadFd && works;
                    for (int r = 0; r < 3 && ok; ++r)
                        ok = os::pread_full(sfd, slotbuf.data() + r * (expert / 3), expert / 3, 2 * expert + r * (expert / 3));
                    if (sfd != os::kBadFd) os::close_fd(sfd);
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));   // (the window the others must wait out)
                    return ok;
                },
                [&] {
                    ++readers;   // (a waiter's sleep tick, not a re-read: it must not write the slot)
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                });
        };
        std::vector<std::thread> th;
        std::atomic<int> oks{0}, nos{0};
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&] { (recover_one(true) ? oks : nos).fetch_add(1); });
        for (auto& t : th) t.join();
        check(slow_reads.load() == 1, "exactly ONE item re-read the slot (exchange picks the single writer)");
        check(oks.load() == 8 && nos.load() == 0, "all eight get the slot: the re-reader and the seven restored hits");
        check(!side.failed(0), "the slot ends restored, not failed-again");
        bool bytes = true;
        for (uint64_t i = 0; i < expert; i += 4099)
            if (slotbuf[(size_t) i] != want_at(2 * expert + i)) bytes = false;
        check(bytes, "and it carries its expert's bytes whole");
        // the refused slot: one attempt, everyone told no, and the failed state STAYS - nobody may run on it
        side.begin(1);
        *side.bad_of(1) = true;
        std::atomic<int> fails_reads{0};
        th.clear();
        oks.store(0); nos.store(0);
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&] {
                const bool ok = PfSide::recover(
                    side, 1,
                    [&] { ++fails_reads; std::this_thread::sleep_for(std::chrono::milliseconds(20)); return false; },
                    [&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
                (ok ? oks : nos).fetch_add(1);
            });
        for (auto& t : th) t.join();
        check(fails_reads.load() == 1 && nos.load() == 8,
              "a refused slot: one attempt, all eight told no - the request fails, no per-item re-read storm");
        check(side.failed(1), "and the slot stays failed: no item takes the half-written bytes as good");
        // the files came back: a later recovery still can fix it
        const bool later = PfSide::recover(
            side, 1,
            [&] {
                const os::fd_t sfd = os::open_ro(path);
                bool ok = sfd != os::kBadFd;
                for (int r = 0; r < 3 && ok; ++r)
                    ok = os::pread_full(sfd, slotbuf.data() + r * (expert / 3), expert / 3, 2 * expert + r * (expert / 3));
                if (sfd != os::kBadFd) os::close_fd(sfd);
                return ok;
            },
            [] { std::this_thread::sleep_for(std::chrono::microseconds(100)); });
        check(later && !side.failed(1), "and a recovery after the files came back still restores the slot");
    }

    os::close_fd(fd);
    remove(path.c_str());
    std::printf("lm_ahead_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
