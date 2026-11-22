// Strata. src/dsv41 store size classes, no model: ds4_store.hpp on synthetic expert files.
// The uniform store cut every slot to the largest layer's blob; the size-class store (the ported
// 00cf5dddab) sizes a slot to its layer. This test feeds the store small binary files with per-
// (layer, expert, matrix) content and checks: the classes it builds, allocation, lookup, content
// integrity against the file (a wrong class' stride or a neighbour's bytes read visibly wrong), the
// byte budget, eviction inside one class, grow and shrink around the budget, and the
// DS4_STORE_UNIFORM=1 fallback. One measurement: how many experts each store shape keeps held under
// the same byte budget before the first eviction. No GGUF, no device, no model.
#include "ds4_store.hpp"

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

static void set_env(const char* kv) {
#ifdef _WIN32
    _putenv(kv);
#else
    std::string s(kv);
    const size_t eq = s.find('=');
    if (eq == std::string::npos) return;
    if (s.size() == eq + 1) unsetenv(s.substr(0, eq).c_str());
    else setenv(s.substr(0, eq).c_str(), s.c_str() + eq + 1, 1);
#endif
}

// the expected byte of matrix r of (layer, e) at position i: position- and identity-dependent, so a
// copy from a wrong place, a wrong class' stride, or a neighbouring slot's bytes read visibly wrong
static uint8_t want_byte(int layer, int e, int r, uint64_t i) {
    return (uint8_t) (i * 31u + (uint64_t) layer * 101u + (uint64_t) e * 7u + (uint64_t) r * 151u + 3u);
}

static void fill_file(uint8_t* data, int layer, int e, const ExpertStore::LayerGeo& g, int r) {
    const uint64_t base = g.off[r] + (uint64_t) e * g.stride[r];
    for (uint64_t i = 0; i < g.bytes[r]; ++i) data[base + i] = want_byte(layer, e, r, i);
}

// the held expert (layer, e) matches the file: every matrix sampled across its span plus its last byte
static bool check_held(const ExpertStore& st, int layer, int e, const std::vector<ExpertStore::LayerGeo>& geo) {
    const int slot = st.find(layer, e);
    if (slot < 0) return false;
    const ExpertStore::LayerGeo& g = geo[(size_t) layer];
    for (int r = 0; r < 3; ++r) {
        const uint8_t* m = st.mat(slot, r);
        for (uint64_t i = 0; i < g.bytes[r]; i += (g.bytes[r] / 64 ? g.bytes[r] / 64 : 1))
            if (m[i] != want_byte(layer, e, r, i)) return false;
        if (m[g.bytes[r] - 1] != want_byte(layer, e, r, g.bytes[r] - 1)) return false;
    }
    return true;
}

// load (layer, e) through the production path; -1 if the store could give no slot at all
static int load(ExpertStore& st, int layer, int e) {
    const int es[1] = {e};
    std::vector<char> ok;
    std::vector<int> sl;
    st.begin(layer, es, 1, ok, sl);
    st.wait_slots(sl, ok, 0, 1);
    int found = ok.size() && ok[0] ? st.find(layer, e) : -1;
    st.release(sl, ok);
    return found;
}

static const int kNExp = 32;
static const int kNLay = 3;

struct Fixture {
    std::vector<ExpertStore::LayerGeo> geo;
    std::string path;
    os::fd_t dfd = os::kBadFd, pfd = os::kBadFd;
    std::vector<os::fd_t> dfds, pfds;

    bool build() {
        // three layers, gate | up | down per expert; layer blobs differ ~4x, so the uniform cut
        // wasted most of a small layer's slot
        const uint64_t mb[kNLay][3] = {
            {256u << 10, 256u << 10, 512u << 10},   // blob 1 MB + slack
            {128u << 10, 128u << 10, 256u << 10},   // 512 KB + slack
            {64u << 10, 64u << 10, 128u << 10},     // 256 KB + slack
        };
        geo.resize(kNLay);
        uint64_t total = 0;
        for (int l = 0; l < kNLay; ++l) {
            ExpertStore::LayerGeo& g = geo[(size_t) l];
            g.n_exp = kNExp;
            uint64_t at = total;
            for (int r = 0; r < 3; ++r) {
                g.file[r] = 0;   // one file holds every layer's matrices
                g.off[r] = at;
                g.stride[r] = mb[l][r];
                g.bytes[r] = mb[l][r];
                at += mb[l][r] * (uint64_t) kNExp;
            }
            total = (at + 4095) & ~4095ull;
            g.tr[0] = 0;
            g.tr[1] = mb[l][0] + 8192;
            g.tr[2] = g.tr[1] + mb[l][1] + 8192;
            g.blob = g.tr[2] + mb[l][2];
        }
        path = (std::getenv("TEMP") ? std::getenv("TEMP") : ".") + std::string("\\ds4_store_classes.bin");
        std::vector<uint8_t> data(total, 0);
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < kNExp; ++e)
                for (int r = 0; r < 3; ++r) fill_file(data.data(), l, e, geo[(size_t) l], r);
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        dfd = os::open_direct(path);
        pfd = os::open_ro(path);
        if (dfd == os::kBadFd || pfd == os::kBadFd) return false;
        dfds = {dfd};
        pfds = {pfd};
        return true;
    }
    void close() {
        if (dfd != os::kBadFd) os::close_fd(dfd);
        if (pfd != os::kBadFd) os::close_fd(pfd);
        remove(path.c_str());
    }
};

int main() {
    set_env("DS4_STORE_CHUNK_SLOTS=8");   // small chunks: the budget bites at slot granularity
    set_env("DS4_STORE_UNIFORM=");
    Fixture fx;
    if (!fx.build()) { std::printf("store_classes_test: cannot build the fixture\n"); return 2; }

    // 1. classes: three blob sizes -> three classes; every expert loads; every byte matches the file
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 64u << 20;   // roomy: every class can commit all its slots
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "class store inits");
        check(st.classes() == 3, "three size classes for three blob sizes");
        const size_t sb_big = (fx.geo[0].blob + 2 * 8192 + 4095) & ~(size_t) 4095;
        check(st.slot_bytes() == sb_big, "slot_bytes stays the largest class (the uniform one)");
        int loaded = 0, good = 0;
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < kNExp; ++e) {
                if (load(st, l, e) < 0) continue;
                ++loaded;
                if (check_held(st, l, e, fx.geo)) ++good;
            }
        check(loaded == kNLay * kNExp, "every expert of every layer takes a slot");
        check(good == loaded, "every held expert's matrices match the file (right class, no overlap)");
        check(st.used_bytes() <= st.budget(), "committed bytes stay inside the budget");
        check(st.used_bytes() > 0, "the store committed something");
        check(st.shrink_chunks(1) <= 1, "shrink returns at most what it was asked to free");
        check(load(st, 1, 0) >= 0 && check_held(st, 1, 0, fx.geo), "load works after a shrink attempt");
    }
    // 2. a class full at the budget's edge: it evicts within itself, the budget holds, content stays right
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 30u << 20;   // class 0's chunk is 8 x 1.032 MB: three chunks commit, the fourth refuses
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "tight store inits");
        int loaded = 0;
        for (int e = 0; e < kNExp; ++e)
            if (load(st, 0, e) >= 0) ++loaded;
        check(loaded == kNExp, "a full class keeps loading by evicting, not by failing");
        check(st.stats().evictions.load() > 0, "the full class evicted within itself");
        check(st.used_bytes() <= st.budget(), "the budget holds through eviction");
        int found = 0, good = 0;
        for (int e = 0; e < kNExp; ++e)
            if (st.find(0, e) >= 0) {
                ++found;
                if (check_held(st, 0, e, fx.geo)) ++good;
            }
        check(found > 0 && good == found, "the evicting class' held experts still read right");
        check(st.stats().fails.load() == 0, "no read failed on the class' own slots");
    }
    // 3. the uniform fallback: one class, everything still exact
    set_env("DS4_STORE_UNIFORM=1");
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 64u << 20;
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "uniform store inits");
        check(st.classes() == 1, "DS4_STORE_UNIFORM=1: one class");
        int loaded = 0, good = 0;
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < 12; ++e) {   // the budget is not the point here: content is
                if (load(st, l, e) < 0) continue;
                ++loaded;
                if (check_held(st, l, e, fx.geo)) ++good;
            }
        check(loaded == 3 * 12 && good == loaded, "the uniform store loads every expert exactly");
    }
    // 4. the point of the classes: under one budget, more experts stay held before the first eviction
    set_env("DS4_STORE_UNIFORM=");
    {
        size_t stored[2] = {0, 0};
        for (int mode = 0; mode < 2; ++mode) {
            set_env(mode == 0 ? "DS4_STORE_UNIFORM=" : "DS4_STORE_UNIFORM=1");
            ExpertStore::Cfg c;
            c.threads = 2;
            c.floor_bytes = 0;
            c.budget_bytes = 50u << 20;   // the class store fits all 3x24 experts; the uniform one thrashes
            ExpertStore st;
            check(st.init(c, fx.geo, fx.dfds, fx.pfds), "budget stores init");
            for (int e = 0; e < 24 && st.stats().evictions.load() == 0; ++e)
                for (int l = 0; l < kNLay; ++l) {
                    if (st.stats().evictions.load() > 0) break;
                    if (st.find(l, e) < 0) load(st, l, e);
                }
            size_t f = 0;
            for (int l = 0; l < kNLay; ++l)
                for (int e = 0; e < 24; ++e)
                    if (st.find(l, e) >= 0) ++f;
            stored[mode] = f;
        }
        std::printf("  before the first eviction: classes held %zu experts, uniform held %zu (same 50 MB)\n",
                    stored[0], stored[1]);
        check(stored[0] > stored[1], "the class store keeps more experts held under the same budget");
    }
    // 5. residency (task spec 8.4) on the store side of the in_ram() fix: an expert the store does not hold is
    // not findable; a held one is findable and its slot reports loaded; find() never names an unfinished slot;
    // after eviction the gone experts are unfindable while every still-held one reads right. (The RAM tier and
    // the model's in_ram() proper live in ds4_model.cu and need the loader; the fix line is store->find()
    // >= 0, so these are the states that line answers for.)
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 64u << 20;
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "residency store inits");
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < kNExp; ++e)
                check(st.find(l, e) < 0, "an expert nothing asked for is not held");
        check(load(st, 1, 5) >= 0, "expert loaded");
        check(st.find(1, 5) >= 0, "a held expert is findable");
        check(st.loaded(st.find(1, 5)), "a findable slot reports loaded (find never names an unfinished read)");
        // begin without waiting: whatever the state, find() and loaded() may not disagree
        {
            const int es[1] = {9};
            std::vector<char> ok;
            std::vector<int> sl;
            st.begin(0, es, 1, ok, sl);
            const int f = st.find(0, 9);
            check(f < 0 || st.loaded(f), "mid-read: a slot find() hands over is already loaded");
            st.wait_slots(sl, ok, 0, 1);
            check(st.loaded(sl[0]) && st.find(0, 9) >= 0, "after the wait the slot is loaded and held");
            st.release(sl, ok);
        }
        check(st.find(kNLay + 3, 0) < 0, "a layer the store does not cover is never held");
        // eviction: hammer layer 0 with a budget that cannot hold everything, then every gone expert is unfindable
        {
            ExpertStore::Cfg c2;
            c2.threads = 2;
            c2.floor_bytes = 0;
            c2.budget_bytes = 12u << 20;   // layer 0 alone needs ~4.5 MB; layers 0+1+2 need ~10 MB of classes
            ExpertStore st2;
            check(st2.init(c2, fx.geo, fx.dfds, fx.pfds), "eviction store inits");
            for (int round = 0; round < 10; ++round)
                for (int e = 0; e < kNExp; ++e)
                    load(st2, 0, e);
            int held = 0, gone = 0, good = 0;
            for (int e = 0; e < kNExp; ++e) {
                const int s = st2.find(0, e);
                if (s < 0) { ++gone; continue; }
                ++held;
                if (!st2.loaded(s)) continue;
                const ExpertStore::LayerGeo& g = fx.geo[0];
                bool right = true;
                for (int r = 0; r < 3; ++r) {
                    const uint8_t* m = st2.mat(s, r);
                    for (uint64_t i = 0; i < g.bytes[r]; i += (g.bytes[r] / 64 ? g.bytes[r] / 64 : 1))
                        if (m[i] != want_byte(0, e, r, i)) right = false;
                }
                if (right) ++good;
            }
            check(st2.stats().evictions.load() > 0, "hammering a full class evicted");
            check(gone + held == kNExp && held > 0 && good == held, "after eviction: held experts read right, gone ones are unfindable");
        }
    }
    // 6. class starvation and the rebalance (hostile review 3): the byte budget can lie entirely in one
    // class's chunks, and then a class that owns none could get nothing forever - the uniform store could
    // always evict somebody, classes cannot. Under DROP_SHARDS (no mapped view) that starves the engine's
    // rescue loop into exiting the process. The store must recast a cold chunk of another class instead,
    // never steal one whose slots are pinned, and every held expert must still match its file.
    set_env("DS4_STORE_CHUNK_SLOTS=8");   // (the test knob: eight-slot chunks, so one chunk fills the budget)
    set_env("DS4_STORE_UNIFORM=");        // (the uniform section above left it on: these blocks need the classes)
    const auto sb_of = [&](int l) {
        return (fx.geo[(size_t) l].blob + 2 * 8192 + 4095) & ~(size_t) 4095;   // the store's own slot-size formula
    };
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        // eight big slots plus a tail too small even for the smallest class's chunk: the first chunk eats
        // the budget, every other class starves by construction
        c.budget_bytes = 8 * sb_of(0) + 200000;
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "starvation store inits");
        check(st.used_bytes() <= st.budget(), "init stays inside the budget");
        // fill the first class: 8 demand loads, all held
        int held0 = 0;
        for (int e = 0; e < 8; ++e)
            if (load(st, 0, e) >= 0) ++held0;
        check(held0 == 8, "the big class fills its chunk");
        // the second class owns no slot and no budget is left for its chunk: the rebalance must give it one
        check(load(st, 1, 3) >= 0, "a starved class gets a recast chunk");
        check(st.stats().rebalances.load() > 0, "the store says it rebalanced");
        check(check_held(st, 1, 3, fx.geo), "the recast class's expert reads its file's bytes");
        check(st.used_bytes() <= st.budget(), "the recast does not grow the arena");
        // and back: the first class is now the starved one - recast must give it room too (its slots are
        // wider than the recast chunk leaves per slot: fewer slots, some entries die - scans must cope)
        for (int e = 0; e < 8; ++e) load(st, 1, e);   // (make the class-1 chunk fully cold-held)
        check(load(st, 0, 1) >= 0, "the wide class gets room from a recast too");
        check(check_held(st, 0, 1, fx.geo), "its expert reads its file's bytes after the recast");
        load(st, 0, 2);
        load(st, 0, 3);   // (more loads than the recast slots: in-class eviction takes over, no -1 spiral)
        // every expert the store still claims must match its file - dead entries and all
        bool all_right = true, any_held = false;
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < kNExp; ++e) {
                if (st.find(l, e) < 0) continue;
                any_held = true;
                if (!check_held(st, l, e, fx.geo)) all_right = false;
            }
        check(any_held && all_right, "after the recasts every held expert reads right, none is a ghost");
        // pinned protection: a chunk whose slot is pinned by a live request must not be recast; the load
        // that cannot be served reports no slot (the engine retries), it does not corrupt or crash
        {
            const int es[1] = {7};
            std::vector<char> ok;
            std::vector<int> sl;
            st.begin(1, es, 1, ok, sl);   // (demand: pin held until release)
            st.wait_slots(sl, ok, 0, 1);   // (let its read land: the probe must meet a kReady pinned slot, not a loading one)
            // class 0, the wide one: its chunk was recast away and no budget is left for a new one, so the
            // store reaches the recast - and the one chunk it could take is the chunk holding a pinned slot.
            // It must answer "no slot" (the engine's rescue retries once pins drop), not steal it.
            const int got = load(st, 0, 9);
            check(got < 0, "with its only chunk pinned, the store gives no slot rather than recasting a live one");
            const int s7 = st.find(1, 7);
            check(s7 >= 0 && st.loaded(s7) && check_held(st, 1, 7, fx.geo), "the pinned expert is intact through the refused recast");
            st.release(sl, ok);   // (second review: by slot, not by (layer, e) - a fresh slot in between keeps its own count)
            // the starved narrow class: the engine's rescue retries until the store can serve; do the same
            int got2 = -1;
            for (int t = 0; t < 500 && (got2 = load(st, 2, 5)) < 0; ++t)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            check(got2 >= 0, "the starved class is served once the pins are down (the rescue's retry shape)");
        }
    }
    // 6b. the recast's dead entries, deterministically: a narrow class's chunk recast to a wider one fits
    // fewer slots - the extra table entries die. Every scan must pass over them, forget the narrow class's
    // experts cleanly, and never show a ghost. Budget: the class-0 chunk + room for six class-2 slots + a
    // tail too small for a class-1 chunk, so the class-1 demand reaches the recast and picks the narrow
    // chunk (five cold slots' score beats the eight loaded class-0 ones).
    {
        const size_t sb0a = sb_of(0), sb2a = sb_of(2);
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 8 * sb0a + 6 * sb2a + 300000;
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "dead-entry store inits");
        int held = 0;
        for (int e = 0; e < 8; ++e)
            if (load(st, 0, e) >= 0) ++held;
        check(held == 8, "the wide class fills its chunk (and out-scores the narrow one later)");
        for (int e = 0; e < 4; ++e)
            check(load(st, 2, e) >= 0, "the narrow class grows its own chunk");
        check(st.live_slots() == st.slots_allocated(), "before the recast every allocated entry is live");
        for (int e = 0; e < 8; ++e)
            st.touch(0, e);   // (keep the wide chunk warm: the coldest-chunk metric - heat or recency, either
                              //  way the touch moves it - so the middle class is offered the narrow chunk)
        check(load(st, 1, 0) >= 0, "the middle class recasts the narrow chunk (its cold score wins)");
        bool forgotten = true, wide_intact = true, mid_held = check_held(st, 1, 0, fx.geo);
        for (int e = 0; e < 4; ++e)
            if (st.find(2, e) >= 0) forgotten = false;   // (the recast forgot them: they must be gone, not ghosts)
        for (int e = 0; e < 8; ++e)
            if (!check_held(st, 0, e, fx.geo)) wide_intact = false;
        check(forgotten, "the recast class's experts are forgotten, not ghosted");
        check(wide_intact, "the untouched class reads right across the dead entries in its neighbour");
        check(mid_held, "the recast's first expert reads its file");
        // (second review 1.3) the dead leftovers count for nothing: the live table shrank by the entries the
        // narrower-to-wider recast killed, and sync()'s reserve arithmetic (nfree) skips them - free slots that
        // the allocator can never hand out would fake the reserve and skip the take-backs below it
        check(st.live_slots() < st.slots_allocated(), "the dead leftovers leave the live count (never handed out, never counted free)");
        check(load(st, 2, 9) >= 0, "the narrow class is served again after the recast (another chunk gives)");
        check(check_held(st, 2, 9, fx.geo), "and its expert reads right past the dead entries");
        bool sweep = true, any = false;
        for (int l = 0; l < kNLay; ++l)
            for (int e = 0; e < kNExp; ++e) {
                if (st.find(l, e) < 0) continue;
                any = true;
                if (!check_held(st, l, e, fx.geo)) sweep = false;
            }
        check(any && sweep, "the dead-entry store's whole sweep: every held expert right, no ghosts");
    }
    // 6c. (hostile review, second round 1.1-1.3) the pin that data access must hold, and the demand fallback
    // that replaces the engine's exit(3): acquire_ready's pin survives the very recast that would forget the
    // expert; the read that answers for a starved lane goes through the caller's handles and never needs a slot.
    {
        const size_t sb0c = sb_of(0), sb2c = sb_of(2);
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 8 * sb0c + 6 * sb2c + 300000;   // (the 6b shape: a later class-1 demand recasts the narrow chunk)
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "pin store inits");
        int held = 0;
        for (int e = 0; e < 8; ++e)
            if (load(st, 0, e) >= 0) ++held;
        check(held == 8, "the wide chunk fills");
        const int ea = 3;
        const int s = st.acquire_ready(0, ea);
        check(s >= 0 && st.pinned(s), "acquire_ready answers with the ready slot, pinned");
        check(st.blob(s) != nullptr && st.live_slots() > 0, "the slot's blob is there, live entries counted");
        // the recast 6b used to forget a chunk with: this chunk's expert ea is pinned now, so the recast either
        // leaves this chunk alone (findings: the pin is what rescue waited 2000 ms for and then exited on)
        for (int e = 0; e < 4; ++e) load(st, 2, e);
        for (int e = 0; e < 8; ++e) st.touch(0, e);
        load(st, 1, 0);   // (the middle class asks: it may recast the narrow chunk, never this pinned expert's slot)
        check(st.find(0, ea) == s && st.pinned(s), "the pinned expert survives the rebalance storm: find names the same slot");
        bool intact = true;
        {   // read it through the pin exactly as the lane does: mat(), sampled against the file
            const ExpertStore::LayerGeo& g = fx.geo[0];
            for (int r = 0; r < 3 && intact; ++r) {
                const uint8_t* m = st.mat(s, r);
                for (uint64_t i = 0; i < g.bytes[r]; i += (g.bytes[r] / 64 ? g.bytes[r] / 64 : 1))
                    if (m[i] != want_byte(0, ea, r, i)) intact = false;
            }
        }
        check(intact, "the pinned blob still holds its expert's bytes across the storm");
        st.unpin(s);
        check(!st.pinned(s), "unpin gives the slot back");
        // demand_read: the starved lane's read with no slot at all - straight through the caller's handles,
        // in the store's blob layout, counted; and a failing handle reports false (the error, not a hang)
        std::vector<uint8_t> blob(st.blob_bytes(2));
        const int ed = 11;
        const bool got = st.demand_read(2, ed, blob.data(),
                                        [&](int f, uint64_t o, uint64_t n, void* d) { return os::pread_full(fx.pfds[f], d, n, o); });
        check(got, "demand_read fills the blob with no slot in the store");
        bool right = true;
        {   // (the lane reads matrices at the trellis places: check those, sampled)
            const ExpertStore::LayerGeo& g = fx.geo[2];
            for (int r = 0; r < 3 && right; ++r) {
                const uint8_t* m = blob.data() + g.tr[r];
                for (uint64_t i = 0; i < g.bytes[r]; i += (g.bytes[r] / 64 ? g.bytes[r] / 64 : 1))
                    if (m[i] != want_byte(2, ed, r, i)) right = false;
                if (m[g.bytes[r] - 1] != want_byte(2, ed, r, g.bytes[r] - 1)) right = false;
            }
        }
        check(right, "the demand blob reads exactly like a slot at the trellis places");
        const bool fail = st.demand_read(2, ed, blob.data(),
                                         [](int, uint64_t, uint64_t, void*) { return false; });
        check(!fail, "a demand_read whose read fails says false (the rescue reports it; it never loops)");
        check(st.demand_read(kNLay + 5, 0, blob.data(), [&](int, uint64_t, uint64_t, void*) { return true; }) == false,
              "demand_read of a layer the store does not cover: false");
    }
    set_env("DS4_STORE_CHUNK_SLOTS=");

    set_env("DS4_STORE_UNIFORM=");

    // 8. (hostile review, third round 5) an abandoned hold: begin() pins the slots it names, and a caller that
    // walks away before wait_slots - the split CPU lane does exactly that when one expert's rescue reports an IO
    // error - owes every pin it was given. The old release returned the ok[i] slots' pins only, so the un-waited
    // ones leaked for good: a pinned slot is never evicted, never recast, and the store ends up holding slots
    // nobody reads from. The real sequence, then the proof the store is still whole.
    {
        ExpertStore::Cfg c;
        c.threads = 2;
        c.floor_bytes = 0;
        c.budget_bytes = 64u << 20;
        ExpertStore st;
        check(st.init(c, fx.geo, fx.dfds, fx.pfds), "the abandon-store inits");
        const int es[4] = {3, 9, 17, 22};
        std::vector<char> ok;
        std::vector<int> sl;
        st.begin(1, es, 4, ok, sl);   // (pins taken, reads on their way - and the caller never waits: the early return)
        bool held = false;
        for (int s : sl) held = held || (s >= 0 && st.pinned(s));
        check(held, "begin holds pins on the experts it named, with nothing waited yet");
        st.release(sl, ok);           // (ok is all zeros: the old release gave NOTHING back; ownership says all)
        bool still = false;
        for (int s : sl) still = still || st.pinned(s);
        check(!still, "release gives back every pin of the abandoned hold, not just the ok ones");
        bool back = true;
        for (int e : es)
            if (load(st, 1, e) < 0 || !check_held(st, 1, e, fx.geo)) back = false;
        check(back, "the abandoned experts load normally afterwards, bytes and all");
    }

    fx.close();
    std::printf("store_classes_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

