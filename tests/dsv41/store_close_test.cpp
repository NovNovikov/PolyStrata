// Strata. src/dsv41 store handle swap under live readers, no model: ds4_store.hpp + ds4_pool.hpp on synthetic
// shard files (hostile review 4). A model load may close or reopen a file's handles while the store's background
// fill (DS4_STORE_BG) still has reads queued through them; a job that carried a copy of the handle would then
// read through a dead value - or, since the OS recycles handle values, through another file's live one, landing
// its bytes in an expert slot silently. The store keeps its handles in FdRef slots and resolves them at dispatch;
// set_pfds closes a replaced handle only once the reads that acquired it have finished. This test drives the real
// store, the real reader queue and real file handles: a background preload thread beside handle swaps (the
// drop_views reopen shape and the drop_files close shape), and an FdRef torture loop that closes a handle and
// churns the OS's value pool right after every replace. No GGUF, no device, no model.
#include "ds4_store.hpp"

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
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

// the expected byte at absolute file offset i (the fixture writes exactly this)
static uint8_t want_at(uint64_t i) { return (uint8_t) (i * 131u + 7u); }

static const int kNExp = 12;
static const int kNLay = 3;

struct Fixture {
    std::vector<ExpertStore::LayerGeo> geo;
    std::vector<std::string> path;                 // one file per layer
    std::vector<os::fd_t> dfds;                    // the direct handles (the test's own; the store never swaps these)
    std::vector<uint8_t> first;                    // file 0's exact bytes (the FdRef torture reads check them)

    // Ownership rule, as in the engine after review 4: a parallel handle handed to a store belongs to the store
    // from then on - set_pfds closes whatever it replaces, and a section hands back a table of kBadFd to close
    // the rest. The test itself only closes what it opens for the FdRef loop, and the dfds.
    std::vector<os::fd_t> fresh_pfds() const {
        std::vector<os::fd_t> v;
        for (const std::string& p : path) v.push_back(os::open_par(p));
        return v;
    }
    std::vector<os::fd_t> no_pfds() const { return std::vector<os::fd_t>(path.size(), os::kBadFd); }

    bool build() {
        const uint64_t mb[kNLay][3] = {
            {64u << 10, 64u << 10, 128u << 10},
            {32u << 10, 32u << 10, 64u << 10},
            {16u << 10, 16u << 10, 32u << 10},
        };
        geo.resize(kNLay);
        const char* tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
        for (int l = 0; l < kNLay; ++l) {
            ExpertStore::LayerGeo& g = geo[(size_t) l];
            g.n_exp = kNExp;
            uint64_t at = 0;
            for (int r = 0; r < 3; ++r) {
                g.file[r] = l;   // one file per layer: a swap closes three handles a round
                g.off[r] = at;
                g.stride[r] = mb[l][r];
                g.bytes[r] = mb[l][r];
                at += mb[l][r] * (uint64_t) kNExp;
            }
            const uint64_t total = (at + 4095) & ~4095ull;
            g.tr[0] = 0;
            g.tr[1] = mb[l][0] + 8192;
            g.tr[2] = g.tr[1] + mb[l][1] + 8192;
            g.blob = g.tr[2] + mb[l][2];
            std::vector<uint8_t> data((size_t) total, 0);
            for (int e = 0; e < kNExp; ++e)
                for (int r = 0; r < 3; ++r) {
                    const uint64_t base = g.off[r] + (uint64_t) e * g.stride[r];
                    for (uint64_t i = 0; i < g.bytes[r]; ++i)
                        data[(size_t) (base + i)] = want_at(base + i);   // position-only: every file the same rule
                }
            path.push_back(std::string(tmp) + "\\ds4_close_" + std::to_string(l) + ".bin");
            FILE* f = fopen(path.back().c_str(), "wb");
            if (!f) return false;
            fwrite(data.data(), 1, data.size(), f);
            fclose(f);
            if (l == 0) first = std::move(data);
        }
        for (const std::string& p : path) {
            dfds.push_back(os::open_unbuffered(p));   // (Linux: kBadFd - the buffered path runs there anyway)
        }
        return !path.empty();
    }
    void close() {
        for (os::fd_t f : dfds)
            if (f != os::kBadFd) os::close_fd(f);
        for (const std::string& p : path) remove(p.c_str());
    }
};

// the held expert (layer, e) matches its file (the blob's slot layout follows geo's tr[])
static bool check_held(const ExpertStore& st, int layer, int e, const std::vector<ExpertStore::LayerGeo>& geo) {
    const int slot = st.find(layer, e);
    if (slot < 0) return false;
    const ExpertStore::LayerGeo& g = geo[(size_t) layer];
    for (int r = 0; r < 3; ++r) {
        const uint8_t* m = st.mat(slot, r);
        for (uint64_t i = 0; i < g.bytes[r]; i += (g.bytes[r] / 32 ? g.bytes[r] / 32 : 1))
            if (m[i] != want_at(g.off[r] + (uint64_t) e * g.stride[r] + i)) return false;
        const uint64_t last = g.bytes[r] - 1;
        if (m[last] != want_at(g.off[r] + (uint64_t) e * g.stride[r] + last)) return false;
    }
    return true;
}

// a demand load held while the caller checks it: pinned, so no eviction can race the content check.
// A transient "no slot" (the classes thrash a 3 MB budget while the fill churns) is answered the way the
// engine's StoreHold::rescue answers it: try again until the store can serve or the window runs out.
static bool load_and_check(ExpertStore& st, int layer, int e, const std::vector<ExpertStore::LayerGeo>& geo) {
    const int es[1] = {e};
    std::vector<char> ok;
    std::vector<int> sl;
    bool good = false;
    for (int t = 0; t < 2000 && !good; ++t) {
        st.begin(layer, es, 1, ok, sl);
        st.wait_slots(sl, ok, 0, 1);
        good = ok.size() && ok[0] && check_held(st, layer, e, geo);
        if (sl.size() && sl[0] >= 0) st.release(sl, ok);   // (a real slot: unpin before the retry; by slot, second review 1.1)
        else if (t < 1999) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (good) st.release(sl, ok);   // (checked while pinned; the pin goes only now)
    if (!good)
        std::printf("  dbg load l%d e%d: ok %d sl %d find %d fails %lld no_slot %lld\n", layer, e,
                    ok.empty() ? -1 : (int) ok[0], sl.empty() ? -9 : sl[0], st.find(layer, e),
                    (long long) st.stats().fails.load(), (long long) st.stats().no_slot.load());
    return good;
}

// the background fill's shape: preload piece after piece until told to stop (the engine's store_bg, in pieces
// of 128 there; one layer's set here), counting finished passes
struct Filler {
    ExpertStore* st;
    std::atomic<bool> stop{false};
    std::atomic<int> rounds{0};
    std::atomic<size_t> got{0};
    void run() {
        while (!stop.load(std::memory_order_relaxed)) {
            for (int l = 0; l < kNLay; ++l) {
                std::vector<std::pair<int, int>> piece;
                for (int e = 0; e < kNExp; ++e) piece.push_back({l, e});
                got += st->preload(piece, 0);
                if (stop.load(std::memory_order_relaxed)) break;
            }
            ++rounds;   // (one pass over every layer counts: the swap sections wait on this)
        }
    }
};

int main() {
    Fixture fx;
    if (!fx.build()) { std::printf("store_close_test: fixture failed\n"); return 2; }

    // 1. the drop_views shape under a live fill: fresh handles swapped in by set_pfds while a preload thread
    // queues reads through the old ones. Buffered-only store (no direct handles): the swapped handles are the
    // ONLY way bytes arrive, so a job stranded on a closed value fails at once. Nothing may fail, and every
    // expert a demand load brings in amid the swaps must carry its file's bytes.
    {
        ExpertStore::Cfg c;
        c.threads = 3;
        c.floor_bytes = 0;
        c.budget_bytes = 3u << 20;   // below the whole set: the fill churns slots while the swaps run
        ExpertStore st;
        check(st.init(c, fx.geo, fx.no_pfds() /*dfds*/ , fx.fresh_pfds()),
              "buffered store inits (the swap runs through the store's own table)");
        Filler fill{&st};
        std::thread th([&] { fill.run(); });
        bool pinned_right = true;
        for (int round = 0; round < 25; ++round) {
            st.set_pfds(fx.fresh_pfds());   // (production order: publish first; the store closes the old at drain)
            if (!load_and_check(st, round % kNLay, round % kNExp, fx.geo)) pinned_right = false;
        }
        fill.stop.store(true);
        th.join();
        check(fill.got.load() > 0, "the background fill really read experts while the swaps ran");
        check(fill.rounds.load() > 0, "the background fill ran through the swaps");
        check(pinned_right, "every demand load landing amid the handle swaps holds its file's bytes");
        check(st.stats().fails.load() == 0,
              "no read failed while handles were swapped under it (before the fix: dead or recycled values)");
        st.set_pfds(fx.no_pfds());   // (hands the table back: the store closes everything still live)
    }

    // 2. the drop_files shape mid-stream: STRATA_DS4_DROP_SHARDS closes a file's buffered handles while the
    // background fill runs - the production way (publish kBadFd through set_pfds, the store closes at drain).
    // Everything after must read through the direct handles with nothing failing.
    {
        if (fx.dfds[0] == os::kBadFd) {
            std::printf("  (no unbuffered handles on this OS: the drop section checks the buffered swap only)\n");
        } else {
            ExpertStore::Cfg c;
            c.threads = 3;
            c.floor_bytes = 0;
            c.budget_bytes = 3u << 20;
            ExpertStore st;
            check(st.init(c, fx.geo, fx.dfds, fx.fresh_pfds()), "direct store inits");
            Filler fill{&st};
            std::thread th([&] { fill.run(); });
            for (int round = 0; round < 5; ++round)
                while (fill.rounds.load() < round + 1) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st.set_pfds(fx.no_pfds());   // (the drop: later reads resolve kBadFd - the direct fd serves them)
            for (int round = 5; round < 10; ++round)
                while (fill.rounds.load() < round + 1) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            bool right = true;
            for (int l = 0; l < kNLay && right; ++l)
                for (int e = 0; e < kNExp; ++e)
                    if (!load_and_check(st, l, e, fx.geo)) right = false;
            fill.stop.store(true);
            th.join();
            check(right, "demand loads after the mid-stream drop read their files through the direct handles");
            check(st.stats().fails.load() == 0, "the mid-stream drop failed no read in flight or queued (drained close)");
            st.set_pfds(fx.fresh_pfds());   // the drop was not the end of the world: buffered serving resumes
            check(load_and_check(st, 1, 3, fx.geo), "a reopened handle serves again");
            st.set_pfds(fx.no_pfds());
        }
    }

    // 3. the FdRef contract itself, without the store: what acquire() hands out is never closed under the reader -
    // the handle and its reader reference are one refcounted block, taken together (third round item 1). A swap
    // returns immediately even mid-read and the loop recycles handle values through the OS as fast as it can, so
    // any use-after-close would show as a failed or wrong-content read; every read checks its file's bytes
    {
        FdRef r;
        os::fd_t cur = os::open_par(fx.path[0]);
        check(cur != os::kBadFd, "the torture handle opens");
        r.set(cur);
        std::atomic<int> bad_reads{0}, mismatches{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> rd;
        for (int t = 0; t < 2; ++t)
            rd.emplace_back([&] {
                std::vector<uint8_t> buf(4096);
                uint64_t i = (uint64_t) t * 977 + 13;
                while (!stop.load(std::memory_order_relaxed)) {
                    FdRef::Handle h;
                    r.acquire(h);
                    const os::fd_t f = h.fd();
                    if (f != os::kBadFd) {
                        i = i * 6364136223846793005ull + 1442695040888963407ull;
                        const uint64_t off = (i >> 13) % (fx.first.size() - 4096);
                        if (!os::pread_par(f, buf.data(), 4096, off)) ++bad_reads;
                        else if (std::memcmp(buf.data(), fx.first.data() + off, 4096) != 0) ++mismatches;
                    }
                    r.release(h);
                }
            });
        int replaced = 0;
        for (int it = 0; it < 300; ++it) {
            const os::fd_t fresh = os::open_par(fx.path[0]);
            if (fresh == os::kBadFd) break;
            r.replace(fresh);   // (returns at once: a read still in flight on the old value counts against the
            ++replaced;         //   retired block, and only its last reference - possibly a reader - closes it)
            const os::fd_t decoy = os::open_par(fx.path[1]);   // (grab the freed value for another file, now)
            if (decoy != os::kBadFd) os::close_fd(decoy);
        }
        stop.store(true);
        for (auto& t : rd) t.join();
        check(replaced > 0, "the torture loop ran its replace/recycle rounds");
        check(bad_reads.load() == 0, "no read through an acquired handle ever hit a closed value");
        check(mismatches.load() == 0, "and never another file's bytes: every read matched the file its handle names");
        r.replace(os::kBadFd);
        check(r.get() == os::kBadFd, "replace(kBadFd) takes the handle out of service");
    }

    // 4. (hostile review, third round item 1) the scenarios the lock-free pair could not hold, each driven with
    // latches and a bounded time: the swapper's halfway point between publishing the new generation and the new
    // handle no longer exists at all - and its mirror case, an old reader carrying the new value.
    {
        FdRef r;
        os::fd_t f0 = os::open_par(fx.path[0]);
        check(f0 != os::kBadFd, "the scenario test handle opens");
        r.set(f0);
        std::vector<uint8_t> want = fx.first;
        auto read_via = [&](os::fd_t f) {   // (one 4096-byte read, verified: is this handle alive and whose bytes?)
            std::vector<uint8_t> buf(4096);
            return os::pread_par(f, buf.data(), 4096, 8192) && std::memcmp(buf.data(), want.data() + 8192, 4096) == 0;
        };
        // (a) a reader acquires, the swap happens WHILE it reads, and its read must still deliver file 0's bytes:
        // the old value leaves service at once but closes only when this last reference lets go
        std::atomic<bool> acq{false}, swapped{false}, mid_ok{false}, rel{false};
        std::thread mid([&] {
            FdRef::Handle h;
            r.acquire(h);
            acq.store(true, std::memory_order_release);
            while (!swapped.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            mid_ok.store(read_via(h.fd()), std::memory_order_release);   // (the handle the swap replaced, mid-read)
            r.release(h);
            rel.store(true, std::memory_order_release);
        });
        while (!acq.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        const os::fd_t f1 = os::open_par(fx.path[0]);
        const auto ta = std::chrono::steady_clock::now();
        r.replace(f1);   // (the third-round defect point: the old protocol closed the value this thread was mid-read through)
        const double wa_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
        swapped.store(true, std::memory_order_release);
        const auto tb = std::chrono::steady_clock::now();
        while (!rel.load(std::memory_order_acquire) &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count() < 5.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        mid.join();
        check(wa_ms < 50.0, "a swap takes the handle out of service without waiting for its in-flight reads");
        check(mid_ok.load(std::memory_order_acquire), "and the read it overtook still got its own file's bytes");
        // (b) the stream of NEW readers: while an old reader sits on f1 (200 ms), new acquires must hand out f2
        // at once (never the old value, never blocked) and read through it fine - the swap that retires f1 must
        // not chase, and must not be chased by, readers that came after it
        std::atomic<bool> held{false}, stop_b{false};
        std::thread old_reader([&] {
            FdRef::Handle h;
            r.acquire(h);   // (f1: the value the next swap retires)
            held.store(true);
            for (int i = 0; i < 40 && !stop_b.load(std::memory_order_relaxed); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            r.release(h);
        });
        while (!held.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const os::fd_t f2 = os::open_par(fx.path[0]);
        r.replace(f2);
        const auto tc = std::chrono::steady_clock::now();
        bool new_fast = false;
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count() < 150.0) {
            FdRef::Handle h;
            r.acquire(h);
            if (h.fd() == f2) { new_fast = read_via(h.fd()); r.release(h); break; }
            r.release(h);
        }
        check(new_fast, "a new reader got the swapped-IN handle - and read through it - while an old reader was still up");
        stop_b.store(true);
        old_reader.join();
        // (c) several swaps in a row over one live reader: each read the holder completes must name file 0
        std::atomic<bool> acq3{false}, done3{false}, ok3{true};
        std::thread holder3([&] {
            FdRef::Handle h;
            r.acquire(h);
            acq3.store(true, std::memory_order_release);
            for (int i = 0; i < 8; ++i) {
                if (!read_via(h.fd())) ok3.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            r.release(h);
            done3.store(true, std::memory_order_release);
        });
        while (!acq3.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        for (int i = 0; i < 4; ++i) {
            r.replace(os::open_par(fx.path[0]));
            const os::fd_t decoy = os::open_par(fx.path[1]);   // (recycle a freed value onto file 1 between swaps)
            if (decoy != os::kBadFd) os::close_fd(decoy);
        }
        const auto td = std::chrono::steady_clock::now();
        while (!done3.load(std::memory_order_acquire) &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count() < 5.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        holder3.join();
        check(ok3.load(std::memory_order_acquire), "four swaps over one live reader: every read still named its file");
        // (d) swap to kBadFd while a reader holds the live one: the reader is unaffected, new acquires come up empty
        std::atomic<bool> acq4{false}, done4{false}, ok4{false};
        std::thread holder4([&] {
            FdRef::Handle h;
            r.acquire(h);
            acq4.store(true, std::memory_order_release);
            while (!done4.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            ok4.store(read_via(h.fd()));
            r.release(h);
        });
        while (!acq4.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        r.replace(os::kBadFd);
        {
            FdRef::Handle gone;
            r.acquire(gone);
            check(gone.fd() == os::kBadFd, "after the swap to kBadFd a fresh acquire hands out no handle at all");
            r.release(gone);
        }
        done4.store(true, std::memory_order_relaxed);
        holder4.join();
        check(ok4.load(), "while the held reader finished its read unharmed");
        // (e) 500 rounds of a last release racing a swap: no lost wakeup is possible with no wait - what these
        // rounds prove instead is that under exactly that churn no value closes early and none is read wrong.
        // Every swap runs behind a watchdog: a hung thread is the defect, so the test names it and leaves.
        std::atomic<int> bad5{0}, wrong5{0};
        std::atomic<bool> stop5{false};
        std::vector<std::thread> churn;
        for (int t = 0; t < 3; ++t)
            churn.emplace_back([&] {
                while (!stop5.load(std::memory_order_relaxed)) {
                    FdRef::Handle h;
                    r.acquire(h);
                    if (h.fd() != os::kBadFd && !read_via(h.fd())) ++wrong5;
                    r.release(h);
                }
            });
        r.replace(os::open_par(fx.path[0]));   // (back to a live handle for the rounds)
        for (int it = 0; it < 500; ++it) {
            std::atomic<bool> acq{false}, done{false};
            std::thread last([&] {
                FdRef::Handle h;
                r.acquire(h);
                acq.store(true, std::memory_order_release);
                if (h.fd() != os::kBadFd && !read_via(h.fd())) ++bad5;   // (the last reader's read: the swap may
                r.release(h);                                            //   retire the value only after this)
                done.store(true, std::memory_order_release);
            });
            while (!acq.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(1));
            r.replace(os::open_par(fx.path[0]));
            std::atomic<bool> decoy_closed{false};
            std::thread decoyer([&] {   // (recycle a value mid-round: an early close would poison THIS reader's next)
                const os::fd_t decoy = os::open_par(fx.path[1]);
                if (decoy != os::kBadFd) os::close_fd(decoy);
                decoy_closed.store(true);
            });
            const auto t0w = std::chrono::steady_clock::now();
            while ((!done.load(std::memory_order_acquire) || !decoy_closed.load()) &&
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0w).count() < 5.0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!done.load(std::memory_order_acquire) || !decoy_closed.load()) {   // (a hang is the defect itself:
                check(false, "500 rounds of last-release-against-swap: no round ever hung");   //  name it, leave -
                std::fflush(stdout);                                                 //   joining would hang the box)
                std::exit(3);
            }
            last.join();
            decoyer.join();
        }
        stop5.store(true);
        for (auto& t : churn) t.join();
        check(bad5.load() == 0 && wrong5.load() == 0, "500 rounds of last-release-against-swap: no early close, no wrong read");
        r.replace(os::kBadFd);
    }

    fx.close();
    std::printf("store_close_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
