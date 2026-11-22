// Strata. src/dsv41 Engram ring, row cache and batch finish, no model (hostile review 5). The pieces the engine's
// Engram path runs on - EgRing (the reader thread's slot book: wraparound, the copy-before-refill gate, the reset
// when the chunk grows), EgRowCache (the clock's second chance and wrap), EgReader (real threads over real files,
// read errors, stop with a batch pending, the MXFP8 unpack) and eg_complete (the finish the engine now calls) -
// driven with synthetic files: a ring whose consumer proves no slot is refilled under it; a batch whose failed
// read must leave zeros in EVERY place its row lands (the ported order copied the duplicates before zeroing, so
// a repeated row of a failed read got the slot's stale bytes); the failed-row report is sticky and a retry hook
// can land a reader-failed row before it is zeroed (second review 4.1). No GGUF, no device, no model.
#include "ds4_engram_io.hpp"

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

static const uint32_t kHd = 64;   // codes a row's column; scales follow, one per 32

static uint8_t row_byte(int row, uint32_t i) { return (uint8_t) (row * 91u + i * 7u + 5u); }

static uint64_t keyfn(int row, int mode) { return ((uint64_t) row + 1) * 16 + (uint64_t) mode; }

int main() {
    // ---- 1. EgRing: wraparound and the copy-before-refill gate, off-device, with a consumer that would notice
    // a slot refilled under it
    {
        EgRing rg;
        rg.reset(3, 1024);
        check(rg.slots() == 3 && rg.bytes() == 1024, "the ring knows its shape");
        check(rg.slot_of(0) == 0 && rg.slot_of(2) == 2 && rg.slot_of(3) == 0 && rg.slot_of(7) == 1,
              "unit u lands in slot u %% n: the wraparound");
        check(rg.can_fill(0) && rg.can_fill(2), "the first round fills every slot at once");
        check(!rg.can_fill(3), "unit 3 waits for unit 0's copy - not filled while the slot holds it");
        rg.copied(0);
        check(rg.can_fill(3) && !rg.can_fill(4), "and only for that: unit 4 still waits for unit 1");
        rg.reset(4, 4096);   // (the resize shape: a bigger chunk remakes the ring; the book starts empty)
        check(rg.slots() == 4 && rg.bytes() == 4096 && !rg.ready(0) && rg.can_fill(0) && rg.can_fill(3),
              "the reset ring: fresh book, all first-round slots fillable");

        // threaded: a producer fills slots, a consumer copies them out; the slot memory carries the unit number,
        // so a producer running ahead of a unit - n's copy shows as a wrong number at the consumer
        std::atomic<bool> violated{false}, stop{false};
        const size_t kUnits = 400, kSlots = 3;
        std::vector<std::atomic<int64_t>> mem(kSlots);
        for (size_t i = 0; i < kSlots; ++i) mem[i].store(-1, std::memory_order_relaxed);
        rg.reset(kSlots, 64);
        std::thread prod([&] {
            for (size_t u = 0; u < kUnits && !stop.load(); ++u) {
                while (!rg.can_fill(u) && !stop.load())
                    std::this_thread::sleep_for(std::chrono::microseconds(20));   // (the engine sleeps 100 us here)
                if (stop.load()) break;
                mem[rg.slot_of(u)].store((int64_t) u, std::memory_order_seq_cst);   // (the fill; the engine reads rows in)
                rg.filled(u);
            }
        });
        size_t got = 0;
        for (size_t u = 0; u < kUnits; ++u) {
            while (!rg.ready(u)) std::this_thread::sleep_for(std::chrono::microseconds(20));
            if ((size_t) mem[rg.slot_of(u)].load(std::memory_order_seq_cst) != u) violated.store(true);
            std::this_thread::sleep_for(std::chrono::microseconds(10));   // (a slow H2D copy)
            rg.copied(u);   // (the engine records the copy, then syncs its event before the refill)
            ++got;
        }
        stop.store(true);
        prod.join();
        check(got == kUnits && !violated.load(), "400 units through 3 slots: nobody copied a refilled slot, no slot filled early");
    }

    // ---- 2. EgRowCache: the clock's second chance, its wrap, and the identity of a key's slot
    {
        EgRowCache c;
        c.init(4, kHd + kHd / 32);
        bool put_ok = true;
        for (int k = 0; k < 4; ++k) {
            uint8_t* p = c.put((uint64_t) k);
            if (!p) put_ok = false;
            else
                for (uint32_t i = 0; i < kHd + kHd / 32; ++i) p[i] = row_byte(k, i);
        }
        check(put_ok, "the cache fills to capacity");
        check(c.find(0) != nullptr, "a held row comes back");
        const uint8_t* p0 = c.put(0);   // (a hit marks it: the clock gives it a second chance)
        const uint8_t* q0 = c.find(0);
        check(p0 != nullptr && p0 == q0, "an existing key keeps its slot");
        check(c.put(4) != nullptr, "the fifth key evicts the clock's victim");
        check(c.find(0) != nullptr, "the touched row survived the clock's first pass");
        check(c.find(1) == nullptr, "and the untouched oldest one did not");
        check(c.find(2) != nullptr && c.find(3) != nullptr && c.find(4) != nullptr, "the rest is intact");
        bool walk = true;
        for (int k = 5; k < 30; ++k)
            if (c.put((uint64_t) k) == nullptr) walk = false;
        check(walk && c.size() == 4, "the clock keeps walking: the cache stays at capacity");
        bool live = true;
        for (int k = 26; k < 30; ++k)
            if (c.find((uint64_t) k) == nullptr) live = false;
        check(live, "the newest rows are all held");
    }

    // ---- 3. EgReader + eg_complete over synthetic files: MXFP8 rows buffered and through the unbuffered
    // small-row path, two-file scale rows, a failed read (its zeros must reach the duplicates too), the
    // cache keeping only whole rows, a second finish being a no-op, and stop() draining a pending batch
    {
        const char* tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
        const std::string good = std::string(tmp) + "\\ds4_eg_good.bin";
        const std::string bad = std::string(tmp) + "\\ds4_eg_missing_42.bin";
        const uint32_t nr = 100, mb = (kHd / 32) * 33;   // an MXFP8 column: [scale, 32 codes] blocks
        auto mxfp_byte = [](int row, uint32_t j) {   // the file's byte j of MXFP8 row `row`
            const uint32_t k = j / 33, o = j % 33;
            return o == 0 ? (uint8_t) (row * 13u + k + 1u) : row_byte(row, k * 32 + o - 1);
        };
        {
            std::vector<uint8_t> data((size_t) nr * mb);
            for (uint32_t r = 0; r < nr; ++r)
                for (uint32_t j = 0; j < mb; ++j) data[(size_t) r * mb + j] = mxfp_byte((int) r, j);
            FILE* f = fopen(good.c_str(), "wb");
            if (!f) { std::printf("engram_ring_test: cannot write %s\n", good.c_str()); return 2; }
            fwrite(data.data(), 1, data.size(), f);
            fclose(f);
        }
        for (int mode = 0; mode < 2; ++mode) {   // 0: buffered handles; 1: the unbuffered small-row path too
            EgReader io;
            const int got_th = io.start({good, bad}, 2, mode == 1 ? std::vector<char>{1, 0} : std::vector<char>{});
            check(got_th == 2, "start reports the reader threads it could create (second review 7.1)");
            check(io.threads() == 2, "the reader runs its own threads");
            std::vector<uint8_t> d0(kHd, 0xAB), d1(kHd, 0xAB), d2(kHd, 0xAB), d3(kHd, 0xAB);
            std::vector<uint8_t> s0(2, 0xCD), s1(2, 0xCD), s2(2, 0xCD), s3(2, 0xCD);
            std::vector<uint8_t> dup_ok(kHd, 0xAB), dup_bad(kHd, 0xAB), dup_ok_s(2, 0xCD), dup_bad_s(2, 0xCD);
            EgBatch b;
            auto mk = [&](int row, int file, std::vector<uint8_t>& d, std::vector<uint8_t>& s) {
                EgRead r;
                r.file = file;
                r.off = (uint64_t) row * mb;
                r.n = mb;
                r.dst = d.data();
                r.scales = s.data();
                r.mx8 = true;
                r.key = keyfn(row, mode);
                b.reads.push_back(r);
            };
            mk(3, 0, d0, s0);    // a good row
            mk(7, 0, d1, s1);    // another good row
            mk(11, 1, d2, s2);   // a missing file: the read must fail
            mk(900, 0, d3, s3);  // an offset past the end: the read must fail
            b.dups.push_back({0, {dup_ok.data(), dup_ok_s.data()}});    // a repeat of the good read
            b.dups.push_back({2, {dup_bad.data(), dup_bad_s.data()}});  // a repeat of the FAILED read
            io.submit(b);   // (the engine submits in eg_issue; eg_complete only waits and finishes)
            int reported = 0;
            EgRowCache cache;
            cache.init(16, kHd + kHd / 32);
            std::mutex cm;
            const size_t bad_n = eg_complete(b, io, cache, cm, kHd, [&](const EgRead&) { ++reported; });
            check(bad_n == 2 && reported == 2, "the missing file and the past-the-end row report as failed reads");
            bool row3 = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d0[i] != row_byte(3, i)) row3 = false;
            for (uint32_t k = 0; k < 2; ++k) {
                if (s0[k] != (uint8_t) (3 * 13u + k + 1u)) row3 = false;
                if (dup_ok_s[k] != s0[k]) row3 = false;
            }
            for (uint32_t i = 0; i < kHd; ++i)
                if (dup_ok[i] != row_byte(3, i)) row3 = false;
            check(row3, "an MXFP8 row unpacks to codes | scales, in its place and in its duplicate");
            bool row7 = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d1[i] != row_byte(7, i)) row7 = false;
            check(row7, "and the next row of the same batch lands whole, next door");
            bool zeroed = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d2[i] || dup_bad[i]) zeroed = false;
            for (uint32_t k = 0; k < 2; ++k)
                if (s2[k] || dup_bad_s[k]) zeroed = false;
            check(zeroed, "a failed read zeros its row AND its duplicate: never the slot's stale 0xAB (the review's point)");
            bool z3 = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d3[i]) z3 = false;
            check(z3, "the past-the-end row reads as zeros too");
            {
                std::lock_guard<std::mutex> lk(cm);
                check(cache.find(keyfn(3, mode)) != nullptr && cache.find(keyfn(7, mode)) != nullptr,
                      "whole rows are kept in the cache");
                check(cache.find(keyfn(11, mode)) == nullptr && cache.find(keyfn(900, mode)) == nullptr,
                      "failed rows are not cached: a later window retries the file");
            }
            // (second review 4.1) the second finish reports the SAME failed count (sticky): a caller that consumes
            // these rows later - the engine's per-layer eg_finish - still learns the rows are zeros, never a clean 0
            const size_t again = eg_complete(b, io, cache, cm, kHd, [&](const EgRead&) { ++reported; });
            check(again == 2 && reported == 2, "a second finish does no work but reports the batch's failed rows again");
            bool intact = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d0[i] != row_byte(3, i) || dup_ok[i] != row_byte(3, i)) intact = false;
            check(intact, "and leaves every byte where it was");

            // stop() with a batch pending: the reader drains what was submitted, a waiter is never stranded
            {
                EgBatch b2;
                std::vector<std::vector<uint8_t>> keep(64, std::vector<uint8_t>(kHd));
                std::vector<std::vector<uint8_t>> keep_s(64, std::vector<uint8_t>(2));
                for (int i = 0; i < 64; ++i) {
                    EgRead r;
                    r.file = 0;
                    r.off = (uint64_t) (i % 100) * mb;
                    r.n = mb;
                    r.dst = keep[(size_t) i].data();
                    r.scales = keep_s[(size_t) i].data();
                    r.mx8 = true;
                    r.key = (uint64_t) i + 1000;
                    b2.reads.push_back(r);
                }
                io.submit(b2);
                io.stop();   // (the engine stops the reader at unload, batches included)
                EgRowCache c2;
                c2.init(0, 0);   // (cap 0: put returns nullptr - the drain must still count reads done)
                std::mutex m2;
                const size_t bad2 = eg_complete(b2, io, c2, m2, kHd, nullptr);
                check(bad2 == 0 && b2.left.load() == 0,
                      "stop() drains a submitted batch: every read counted, a waiter is never stranded");
            }
        }
        // a plain row with the scales in a second place (the GGUF Q6_K-with-side-scales shape, buffered path)
        {
            EgReader io;
            io.start({good}, 1, {});
            std::vector<uint8_t> d(kHd, 0xAB), s(2, 0xCD), dupd(kHd, 0xAB), dups(2, 0xCD);
            EgBatch b;
            EgRead r;
            r.file = 0;
            r.off = 5 * mb;
            r.n = mb;
            r.dst = d.data();
            r.scales = s.data();
            r.mx8 = false;
            r.file2 = 0;
            r.off2 = 9 * mb;
            r.n2 = 2;
            r.key = 77;
            b.reads.push_back(r);
            b.dups.push_back({0, {dupd.data(), dups.data()}});
            io.submit(b);
            EgRowCache cache;
            cache.init(4, kHd + 2);
            std::mutex cm;
            const size_t bad_n = eg_complete(b, io, cache, cm, kHd, nullptr);
            bool ok = bad_n == 0;
            for (uint32_t i = 0; i < kHd; ++i)
                if (d[i] != mxfp_byte(5, i)) ok = false;   // (n bytes as they are: the file's own, no unpack)
            for (uint32_t k = 0; k < 2; ++k)
                if (s[k] != mxfp_byte(9, k)) ok = false;
            check(ok, "a two-part row reads its bytes and its second file's scales");
            bool dup_ok2 = true;
            for (uint32_t i = 0; i < kHd; ++i)
                if (dupd[i] != mxfp_byte(5, i)) dup_ok2 = false;
            check(dup_ok2, "and its duplicate gets the same bytes");
            io.stop();
        }
        remove(good.c_str());
    }

    // ---- 4. (second review 4.1) the retry hook: a failing reader row the engine's own read can still land, and
    // one it cannot. The repaired row is bytes (cached, not zeroed, not failed); the refused one is zeros, stays
    // out of the cache, and the batch keeps reporting it - so the engine's ring never publishes it as ready.
    {
        const char* tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
        const std::string good = std::string(tmp) + "\\ds4_eg_good.bin";
        const std::string bad = std::string(tmp) + "\\ds4_eg_missing_42.bin";
        const uint32_t mb = (kHd / 32) * 33;
        {
            std::vector<uint8_t> data((size_t) 100 * mb);
            for (uint32_t r = 0; r < 100; ++r)
                for (uint32_t j = 0; j < mb; ++j) {
                    const uint32_t k = j / 33, o = j % 33;
                    data[(size_t) r * mb + j] = o == 0 ? (uint8_t) (r * 13u + k + 1u) : row_byte((int) r, k * 32 + o - 1);
                }
            FILE* f = fopen(good.c_str(), "wb");
            if (!f) { std::printf("engram_ring_test: cannot write %s\n", good.c_str()); return 2; }
            fwrite(data.data(), 1, data.size(), f);
            fclose(f);
        }
        EgReader io;
        io.start({good, bad}, 2, {});
        std::vector<uint8_t> d_fixed(kHd, 0xAB), s_fixed(2, 0xCD), d_refused(kHd, 0xAB), s_refused(2, 0xCD);
        EgBatch b;
        auto mk = [&](int row, int file, std::vector<uint8_t>& d, std::vector<uint8_t>& s) {
            EgRead r;
            r.file = file;
            r.off = (uint64_t) row * mb;
            r.n = mb;
            r.dst = d.data();
            r.scales = s.data();
            r.mx8 = true;
            r.key = keyfn(row, 9);
            b.reads.push_back(r);
        };
        mk(21, 1, d_fixed, s_fixed);      // the missing file: the READER fails...
        mk(33, 1, d_refused, s_refused);  // ... and here even the engine's own read says no
        io.submit(b);
        EgRowCache cache;
        cache.init(8, kHd + kHd / 32);
        std::mutex cm;
        int retried = 0, reported = 0;
        // the hook stands in for the engine's eg_retry_row: it reads the row where the file really is (the good
        // file at the row's offset) and lays it out as EgReader would; it refuses the row the engine cannot get
        const size_t bad_n = eg_complete(
            b, io, cache, cm, kHd, [&](const EgRead&) { ++reported; },
            [&](const EgRead& r) {
                ++retried;
                if (r.key == keyfn(21, 9)) {   // the repairable row: read from `good` and de-interleave
                    FILE* f = fopen(good.c_str(), "rb");
                    if (!f) return false;
                    std::vector<uint8_t> tmp(mb);
                    const bool got = fseek(f, (long) r.off, SEEK_SET) == 0 && fread(tmp.data(), 1, mb, f) == mb;
                    fclose(f);
                    if (!got) return false;
                    for (uint32_t k = 0; k < mb / 33; ++k) {
                        r.scales[k] = tmp[(size_t) k * 33];
                        memcpy(r.dst + (size_t) k * 32, tmp.data() + (size_t) k * 33 + 1, 32);
                    }
                    return true;
                }
                return false;   // the refused row: the engine's own handles fail too
            });
        check(retried == 2, "the hook is asked for every failed read, before any zeroing");
        check(bad_n == 1 && reported == 1, "a repaired retry is not a failed row; only the refused one is");
        bool fixed_ok = true;
        for (uint32_t i = 0; i < kHd; ++i)
            if (d_fixed[i] != row_byte(21, i)) fixed_ok = false;
        for (uint32_t k = 0; k < 2; ++k)
            if (s_fixed[k] != (uint8_t) (21 * 13u + k + 1u)) fixed_ok = false;
        check(fixed_ok, "the repaired row carries the file's bytes, not zeros");
        bool refused_zero = true;
        for (uint32_t i = 0; i < kHd; ++i)
            if (d_refused[i]) refused_zero = false;
        for (uint32_t k = 0; k < 2; ++k)
            if (s_refused[k]) refused_zero = false;
        check(refused_zero, "the refused row is zeroed as before");
        {
            std::lock_guard<std::mutex> lk(cm);
            check(cache.find(keyfn(21, 9)) != nullptr, "the repaired row is cached like any whole one");
            check(cache.find(keyfn(33, 9)) == nullptr, "the refused row is not");
        }
        check(eg_complete(b, io, cache, cm, kHd, nullptr) == 1,
              "and the batch keeps reporting the refused row: the ring will not publish its unit as ready");
        io.stop();
        remove(good.c_str());
    }
    std::printf("engram_ring_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
