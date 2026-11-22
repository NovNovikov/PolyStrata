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

    // 3. the FdRef contract itself, without the store: acquire hands out the live value only - replace()
    // returns after every read that had the old one, and the test closes the old value and churns the OS's
    // handle pool right there, so any use-after-close shows as a failed or wrong-content read
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
                    int gen = 0;
                    const os::fd_t f = r.acquire(gen);
                    if (f != os::kBadFd) {
                        i = i * 6364136223846793005ull + 1442695040888963407ull;
                        const uint64_t off = (i >> 13) % (fx.first.size() - 4096);
                        if (!os::pread_par(f, buf.data(), 4096, off)) ++bad_reads;
                        else if (std::memcmp(buf.data(), fx.first.data() + off, 4096) != 0) ++mismatches;
                    }
                    r.release(gen);
                }
            });
        int replaced = 0;
        for (int it = 0; it < 300; ++it) {
            const os::fd_t fresh = os::open_par(fx.path[0]);
            if (fresh == os::kBadFd) break;
            const os::fd_t old = r.replace(fresh);   // (waits for every read that had the old value)
            if (old != os::kBadFd) { os::close_fd(old); ++replaced; }
            const os::fd_t decoy = os::open_par(fx.path[1]);   // (grab the freed value for another file, now)
            if (decoy != os::kBadFd) os::close_fd(decoy);
        }
        stop.store(true);
        for (auto& t : rd) t.join();
        check(replaced > 0, "the torture loop ran its replace/close/recycle rounds");
        check(bad_reads.load() == 0, "no read through an acquired handle ever hit a closed value");
        check(mismatches.load() == 0, "and never another file's bytes: every read matched the file its handle names");
        const os::fd_t last = r.replace(os::kBadFd);
        check(last != os::kBadFd, "replace reports the handle it took over (the closer closes that one)");
        if (last != os::kBadFd) os::close_fd(last);
    }

    // 4. (hostile review, second round 3.1) what the generation counters promise, each with a bounded time:
    // a swap waits for the readers of the handle it replaced and NOT for readers of the one it swapped in,
    // and a last release racing a swap wakes it - every round, always, no sleeping forever.
    {
        FdRef r;
        os::fd_t f0 = os::open_par(fx.path[0]);
        check(f0 != os::kBadFd, "the generation test handle opens");
        r.set(f0);
        // (a) the old generation IS waited for: a holder that acquired f0 keeps the swap waiting (bounded: it lets go)
        std::atomic<bool> held{false};
        std::thread holder([&] {
            int gen = 0;
            const os::fd_t f = r.acquire(gen);
            held.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            r.release(gen);
            (void) f;
        });
        while (!held.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto ta = std::chrono::steady_clock::now();
        const os::fd_t f1v = os::open_par(fx.path[0]);   // (the handle the swap installs: (b) names it below)
        const os::fd_t old_a = r.replace(f1v);
        const double wa_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
        holder.join();
        check(old_a == f0, "the swap took over the handle the holder had");
        check(wa_ms > 50.0 && wa_ms < 5000.0, "a swap waits its own generation's readers out, and returns");
        // (b) readers that arrive after the swap - and so hold the handle it swapped IN - must not be waited for:
        // a slow reader of the OLD handle keeps the swap busy 150 ms (it must be waited out), while a stream of
        // readers spins for the new handle and one then HOLDS it 400 ms. The single old counter would have let
        // that 400 ms holder extend the swap past 300 ms; the generation-scoped wait ends when the slow one lets go.
        std::atomic<bool> held2{false}, held3{false}, stop3{false};
        std::thread old_slow([&] {
            int gen = 0;
            r.acquire(gen);   // (f1: the generation the next swap will replace)
            held2.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            r.release(gen);
        });
        while (!held2.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::thread new_long([&] {
            for (;;) {   // (spin until the acquire hands out the swapped-IN handle: then sit on it)
                int gen = 0;
                const os::fd_t f = r.acquire(gen);
                if (f != f1v) {
                    held3.store(true);
                    for (int i = 0; i < 160 && !stop3.load(std::memory_order_relaxed); ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    r.release(gen);
                    return;
                }
                r.release(gen);
            }
        });
        const auto tb = std::chrono::steady_clock::now();
        const os::fd_t old_b = r.replace(os::open_par(fx.path[1]));
        const double wb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
        old_slow.join();
        while (!held3.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));   // (the swap returned WHILE this holder sat on the new handle)
        stop3.store(true);
        new_long.join();
        check(old_b != os::kBadFd, "the second swap took its handle over");
        check(wb_ms > 50.0 && wb_ms < 400.0, "the swap waited its old generation out (150 ms) and not the new handle's 800 ms holder");
        os::close_fd(old_b);
        // (c) the lost wakeup, driven for real: a last release and a swap at the same moment, 500 rounds. The
        // swap runs behind a watchdog: a lost notify would hang it, and the test says so by name and leaves.
        int hung = 0;
        for (int it = 0; it < 500; ++it) {
            const os::fd_t fresh = os::open_par(fx.path[0]);
            std::atomic<bool> acq{false}, done{false};
            std::thread last([&] {
                int gen = 0;
                const os::fd_t f = r.acquire(gen);
                acq.store(true, std::memory_order_release);
                r.release(gen);
                (void) f;
            });
            while (!acq.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::atomic<bool> swapped{false};
            std::thread swapper([&] {
                const os::fd_t old = r.replace(fresh);
                if (old != os::kBadFd) os::close_fd(old);
                swapped.store(true);
            });
            last.join();
            const auto t0w = std::chrono::steady_clock::now();
            while (!swapped.load() &&
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0w).count() < 5.0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!swapped.load()) {   // (a hung swap is the defect itself: name it and leave - joining it would hang the box)
                check(false, "500 rounds of last-release-against-swap: no swap ever lost its wakeup");
                std::fflush(stdout);
                std::exit(3);
            }
            swapper.join();
        }
        check(hung == 0, "500 rounds of last-release-against-swap: no swap ever lost its wakeup");
        const os::fd_t last_c = r.replace(os::kBadFd);
        if (last_c != os::kBadFd) os::close_fd(last_c);
    }

    fx.close();
    std::printf("store_close_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
