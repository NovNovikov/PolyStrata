// Strata. ds4_os.hpp's read paths on a small temporary file - the shapes STRATA_DS4_DROP_SHARDS (the f62520a2aa
// port, task spec 8.8) leaves the engine running with: unbuffered handles alive, buffered handles and mapped
// views closed. Checks the production functions: buffered, parallel-buffered, unbuffered and direct reads of
// the same bytes agree; unaligned requests (the lead) give exactly the asked-for bytes; after the buffered
// handle is closed the direct one still reads right; touching a closed handle reports failure rather than
// success; a read past the file's end fails and does not fake data; file metadata survives the drop.
// No GGUF, no device, no model. Windows runs all of it; the unbuffered/direct handles are Windows concepts
// and the functions report failure on other platforms, where the comparisons still run on the buffered pair.
#include "ds4_os.hpp"
#include "ds4_pool.hpp"   // (third round 7) IoQueue: the flow's CPU-side reads and the drain-before-release contract

#ifdef _WIN32
#include <malloc.h>   // _aligned_malloc: the direct reads need an aligned destination, as the pool's does
#endif
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

static uint8_t pat(uint64_t i) { return (uint8_t) (i * 131u + (i >> 8) * 7u + 11u); }

int main() {
    const uint64_t FSIZE = 200000;   // not a multiple of 4096: the tail edge matters
    const std::string path = (std::getenv("TEMP") ? std::getenv("TEMP") : ".") + std::string("\\ds4_io_test.bin");
    {
        std::vector<uint8_t> data(FSIZE);
        for (uint64_t i = 0; i < FSIZE; ++i) data[i] = pat(i);
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { std::printf("ds4_io_test: cannot write the temp file\n"); return 2; }
        fwrite(data.data(), 1, FSIZE, f);
        fclose(f);
    }
    const os::fd_t ro = os::open_ro(path), par = os::open_par(path), dir = os::open_direct(path), unbuf = os::open_unbuffered(path);
    check(ro != os::kBadFd, "buffered handle opens");
    check(par != os::kBadFd, "the pool's buffered handle opens");
    check(dir != os::kBadFd, "the unbuffered handle opens");
    check(unbuf != os::kBadFd, "the second unbuffered handle opens");
    check(os::file_size(ro) == FSIZE, "buffered metadata: the file size");
    check(os::file_size(dir) == FSIZE, "the unbuffered handle keeps the file metadata");

    // aligned reads through all four paths agree with the pattern
    {
        std::vector<uint8_t> want(4096);
        for (uint64_t i = 0; i < 4096; ++i) want[i] = pat(8192 + i);
        std::vector<uint8_t> a(4096), b(4096), c(4096, 0), d(4096, 0);
        check(os::pread_full(ro, a.data(), 4096, 8192) && a == want, "buffered aligned read");
        check(os::pread_par(par, b.data(), 4096, 8192) && b == want, "parallel-buffered aligned read");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(8192, 4096) + os::kDirectAlign, os::kDirectAlign);
            check(raw != nullptr && os::pread_direct(dir, raw, 4096, 8192) &&
                  std::memcmp(raw + 8192 % os::kDirectAlign, want.data(), 4096) == 0, "direct aligned read (lead 0)");
            _aligned_free(raw);
        }
        if (unbuf != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(8192, 4096) + os::kDirectAlign, os::kDirectAlign);
            check(raw != nullptr && os::pread_unbuffered(unbuf, raw, 4096, 8192) &&
                  std::memcmp(raw + 8192 % os::kDirectAlign, want.data(), 4096) == 0, "unbuffered aligned read (lead 0)");
            _aligned_free(raw);
        }
    }
    // unaligned requests: the lead's bytes are the pool's dst+lead contract - exact content at odd offsets
    for (uint64_t off : {3ull, 4095ull, 4097ull, 150001ull}) {
        const uint64_t n = 1000;
        std::vector<uint8_t> want(n), a(n), b(n);
        for (uint64_t i = 0; i < n; ++i) want[i] = pat(off + i);
        check(os::pread_full(ro, a.data(), n, off) && a == want, "buffered unaligned read");
        check(os::pread_par(par, b.data(), n, off) && b == want, "parallel-buffered unaligned read (its own lead)");
        if (dir != os::kBadFd) {
            const uint64_t lead = off % os::kDirectAlign;
            std::vector<uint8_t> raw((size_t) os::direct_span(off, n));
            uint8_t* al = (uint8_t*) _aligned_malloc(raw.size(), os::kDirectAlign);
            bool got = al != nullptr && os::pread_direct(dir, al, n, off) && std::memcmp(al + lead, want.data(), n) == 0;
            _aligned_free(al);
            check(got, "direct unaligned read: dst[lead .. lead+n) is exactly the asked-for bytes");
        }
    }
    // the DROP_SHARDS shape: close the buffered handles, then read through the direct one
    {
        os::close_fd(par);
        std::vector<uint8_t> want(4096), a(4096);
        for (uint64_t i = 0; i < 4096; ++i) want[i] = pat(20480 + i);
        check(os::pread_full(ro, a.data(), 4096, 20480) && a == want, "the kept buffered handle is untouched by another file-shape close");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(20480, 4096), os::kDirectAlign);
            check(raw != nullptr && os::pread_direct(dir, raw, 4096, 20480) && std::memcmp(raw, want.data(), 4096) == 0,
                  "direct read works with the pool's buffered handle closed (the post-drop state)");
            _aligned_free(raw);
        }
        os::close_fd(ro);
        std::vector<uint8_t> junk(4096, 0);
        check(!os::pread_full(ro, junk.data(), 4096, 20480), "a closed buffered handle reports failure, not success");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(20480, 4096), os::kDirectAlign);
            bool still = raw != nullptr && os::pread_direct(dir, raw, 4096, 20480);
            _aligned_free(raw);
            check(still, "the direct handle still reads after every buffered handle closed");
        }
    }
    // a read past the end fails rather than filling with anything: 4096 at FSIZE-1000 spans the EOF
    {
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(FSIZE - 1000, 4096), os::kDirectAlign);
            const bool bad = !os::pread_direct(dir, raw, 4096, FSIZE - 1000);
            _aligned_free(raw);
            check(bad, "a direct read past the end fails - the pool reports it, nothing computes on it");
        }
    }
    os::close_fd(dir);
    os::close_fd(unbuf);

    // (hostile review, third round 7) drain before release, the abandoned flow-run shape. The engine's chunk owns
    // its items: a job in flight counts the item's left down and writes bytes into the chunk's staging. Abandon
    // points - the store unpin, the item drop, the arena free - may pass only after every posted job landed; the
    // old flow_release_pins unpinned (and run_dev_chunks dropped items) with jobs still running, so a completion
    // wrote into freed counters and buffers. Production IoQueue, production Job contract, the engine's exact
    // procedure: post three parts an item, abandon mid-run, drain the posted items' counters the way
    // flow_release_pins now does, and only then free the items and the staging. 200 rounds; the invariants: after
    // the drain every counter is zero and every byte landed, and the abandons did catch jobs in flight.
    {
        struct Item {
            std::atomic<int> left{0};
            std::atomic<bool> bad{false};
        };
        const auto part_off = [](int i, int p, int round) { return (uint64_t) (10 + (i * 3 + p) + round % 4) * 4096ull; };
        const os::fd_t iofd = os::open_par(path);
        check(iofd != os::kBadFd, "the file reopens for the drain scenario");
        bool good = iofd != os::kBadFd;
        int inflight_rounds = 0;
        for (int round = 0; round < 200 && good; ++round) {
            IoQueue io(2);
            auto items = std::make_unique<std::vector<Item>>(8);
            auto staging = std::make_unique<std::vector<uint8_t>>(8 * 3 * 4096, 0);
            const int posted = 1 + (round * 7919) % 8;   // (the pump's cut: the first `posted` items got their jobs)
            for (int i = 0; i < posted; ++i) {
                Item& it = (*items)[(size_t) i];
                it.left.store(3);
                IoQueue::Job jobs[3];
                for (int p = 0; p < 3; ++p) {
                    IoQueue::Job& j = jobs[p];
                    j = IoQueue::Job{};
                    j.fd = iofd;
                    j.off = part_off(i, p, round);
                    j.len = 4096;
                    j.dst = staging->data() + (size_t) i * 3 * 4096 + (size_t) p * 4096;
                    j.left = &it.left;
                    j.bad = &it.bad;
                }
                io.push(jobs, 3);
            }
            bool flying = false;   // (the abandon moment: what the old code did was walk on FROM HERE)
            for (int i = 0; i < posted; ++i)
                flying = flying || (*items)[(size_t) i].left.load(std::memory_order_relaxed) != 0;
            if (flying) ++inflight_rounds;
            for (int i = 0; i < posted; ++i)   // (flow_release_pins' drain, the same keys: base..n_read, wait_any)
                while ((*items)[(size_t) i].left.load(std::memory_order_acquire) != 0) io.wait_any(500);
            for (int i = 0; i < posted && good; ++i) {
                if ((*items)[(size_t) i].left.load() != 0 || (*items)[(size_t) i].bad.load()) good = false;
                for (int p = 0; p < 3 && good; ++p) {
                    const uint64_t off = part_off(i, p, round);
                    for (size_t q = 0; q < 4096 && good; q += 61)
                        if ((*staging)[(size_t) i * 3 * 4096 + (size_t) p * 4096 + q] != pat(off + q)) good = false;
                }
            }
            items.reset();   // (the release step: this very free was the use-after-free when it came too early)
            staging.reset();
        }
        check(good, "200 abandoned runs drained then freed: counters zero, bytes right, no job outlived its item");
        check(inflight_rounds > 100, "and the abandons caught jobs in flight (the scenario really drives the hazard)");
        os::close_fd(iofd);
    }

    // (hostile review, third round item 8) promotion out of the low queue. LM_AHEAD's speculative reads queue at
    // LOW priority; the moment an item of that layer reaches the pump, the read is a DEMAND - the pump waits on
    // it - and it must overtake the guesses nobody needs. Deterministic shape: ONE worker; a big host-copy job
    // occupies it; five guesses (each a 16 MB copy, so the first completion can be OBSERVED, not guessed at)
    // queue low; a second big high job re-occupies it; the third guess is promoted. The worker's order is then
    // forced: big one, big two, the promoted one, the rest. So the FIRST guess to complete must be the promoted
    // one - without a promote the guesses run in queue order and g0 finishes first.
    {
        const size_t BUSY = 128u << 20;   // (~15-30 ms an occupancy window; the promote fires microseconds in)
        const size_t GUESS = 16u << 20;   // (~1-2 ms: the observation window for who finished first)
        std::vector<uint8_t> big_src(BUSY, 7), big_dst_a(BUSY, 0), big_dst_b(BUSY, 0);
        std::vector<uint8_t> guess_dst(5 * GUESS, 0);
        IoQueue io(1);
        std::atomic<int> g0{1}, g1{1}, g2{1}, g3{1}, g4{1};
        std::atomic<int>* guesses[5] = {&g0, &g1, &g2, &g3, &g4};
        IoQueue::Job busy = IoQueue::Job{};
        busy.mem = big_src.data();
        busy.dst = big_dst_a.data();
        busy.len = BUSY;
        io.push(&busy, 1);   // (the worker takes it at once: everything after this queues behind it)
        for (int i = 0; i < 5; ++i) {
            IoQueue::Job j = IoQueue::Job{};
            j.mem = big_src.data();
            j.dst = guess_dst.data() + (size_t) i * GUESS;
            j.len = GUESS;
            j.left = guesses[i];
            io.push(&j, 1, true);   // (low: the LM_AHEAD shape)
        }
        busy.dst = big_dst_b.data();
        io.push(&busy, 1);   // (a demand of its own in the high queue: it holds the worker past the promote)
        io.promote(guesses[2]);   // (the pump's moment: this guess is needed NOW)
        // the strict watch: the first guess whose counter lands - it must be the promoted one, and the others
        // must be whole at that instant (the worker is single: whatever lands first also STARTED first)
        const auto t0 = std::chrono::steady_clock::now();
        bool first_was_promoted = false;
        for (;;) {
            const bool g2_done = g2.load(std::memory_order_acquire) == 0;
            const bool other_done = g0.load(std::memory_order_acquire) == 0 || g1.load() == 0 ||
                                    g3.load() == 0 || g4.load() == 0;
            if (g2_done || other_done) { first_was_promoted = g2_done && !other_done; break; }
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 10.0) break;
        }
        check(first_was_promoted, "the promoted demand read was the FIRST guess to land, ahead of the whole low queue");
        for (int i = 0; i < 5; ++i)
            while (guesses[i]->load() != 0 &&
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 10.0)
                io.wait_any(1000);
        bool all_in = g0.load() == 0 && g1.load() == 0 && g2.load() == 0 && g3.load() == 0 && g4.load() == 0;
        check(all_in, "and every guess still ran after the demands - promotion is a reorder, not a loss");
    }

    remove(path.c_str());
    std::printf("ds4_io_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
