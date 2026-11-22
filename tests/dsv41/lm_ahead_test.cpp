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
    // engine's answer - re-read on demand - restores its bytes
    {
        PfSide side;
        side.init(2);
        post_expert(io, side, 0, os::kBadFd, 3 * expert, expert, slotbuf.data());   // both handles dead: the read fails
        side.drain([&] { std::this_thread::sleep_for(std::chrono::microseconds(200)); });
        check(side.landed(0), "a failed speculative read counts as landed: no waiter hangs on it");
        check(side.failed(0), "and says so: the engine's pump sees pf_bad and falls back to a demand read");
        bool demand = true;
        const os::fd_t sfd = os::open_ro(path);   // (a plain synchronous handle, the shape the engine's read_at uses)
        for (int r = 0; r < 3 && demand; ++r)
            demand = os::pread_full(sfd, slotbuf.data() + r * (expert / 3), expert / 3, 3 * expert + r * (expert / 3));
        if (sfd != os::kBadFd) os::close_fd(sfd);
        check(demand, "the demand retry the engine now runs on it succeeds");
        bool bytes = true;
        for (uint64_t i = 0; i < expert; i += 4099)
            if (slotbuf[(size_t) i] != want_at(3 * expert + i)) bytes = false;
        check(bytes, "and the item runs on the slot's expert, not a failed guess - the request is not killed");
    }

    os::close_fd(fd);
    remove(path.c_str());
    std::printf("lm_ahead_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
