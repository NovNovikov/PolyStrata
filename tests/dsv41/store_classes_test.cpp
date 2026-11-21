// Strata. src/dsv41 store size classes, no model: ds4_store.hpp on synthetic expert files.
// The uniform store cut every slot to the largest layer's blob; the size-class store (the ported
// 00cf5dddab) sizes a slot to its layer. This test feeds the store small binary files with per-
// (layer, expert, matrix) content and checks: the classes it builds, allocation, lookup, content
// integrity against the file (a wrong class' stride or a neighbour's bytes read visibly wrong), the
// byte budget, eviction inside one class, grow and shrink around the budget, and the
// DS4_STORE_UNIFORM=1 fallback. One measurement: how many experts each store shape keeps held under
// the same byte budget before the first eviction. No GGUF, no device, no model.
#include "ds4_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
    st.release(layer, es, 1, ok);
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
            st.release(0, es, 1, ok);
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
    set_env("DS4_STORE_UNIFORM=");
    fx.close();
    std::printf("store_classes_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
