// Strata. src/dsv41/ds4_store.hpp - the RAM tier of the DeepSeek-V4.1 routed experts as a cache of its own: whole experts read
// from the model's files by unbuffered requests (the OS page cache neither used nor disturbed), kept in a fixed arena of slots
// that a replacement policy owns, filled on demand (the CPU lane's picks) and ahead of time (the lookahead's guesses).
//
// A slot holds an expert in the engine's blob layout (gate | up | down at the layer's offsets tr[r], the same as a VRAM slot), so
// the CPU lane's rows read it in place and - when the arena is registered with the GPU - the device kernels read it over PCIe
// (zero-copy, 24 GB/s on a 5070) through the same plan pointers as a VRAM expert.  An expert is three matrices at three file
// offsets (the GGUF's one tensor a role and layer): each is read aligned (4096) into the slot and moved into place (the read's
// lead of up to 4095 bytes), the blob layout leaving 8192 bytes between matrices (and around the blob) for the aligned spans.
//
// Memory: slots are allocated in chunks as they are first needed (private memory), never past the budget nor past the RAM this
// process can still get less `floor_bytes` (the owner's session and the system keep theirs).
//
// Policy (the slot to give up, among the unpinned, not GPU-visible ones): lru (oldest touch), lfu (lowest decayed hit count).
// A slot whose expert a window is computing is pinned from ensure() to release(); a slot the GPU may read is `visible` until
// sync() takes it back at a window boundary (the GPU idle), which also publishes the newly ready ones.
#pragma once

#include "ds4_os.hpp"
#include "ds4_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4 {

class ExpertStore {
public:
    struct LayerGeo {
        int n_exp = 0;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};   // matrix r of expert e: file[r], off[r] + e stride[r]
        uint64_t bytes[3] = {0, 0, 0};
        uint64_t tr[3] = {0, 0, 0};                            // matrix r's place in the blob
        uint64_t blob = 0;                                     // the blob's bytes (gaps of 8192 between matrices)
    };
    enum Policy { kLru = 0, kLfu = 1 };
    struct Stats {
        std::atomic<int64_t> hits{0}, misses{0}, prefetched{0}, prefetch_used{0}, prefetch_wasted{0}, evictions{0}, fails{0};
        std::atomic<int64_t> bytes_read{0}, wait_ns{0}, no_slot{0}, cancelled{0}, put{0};
        std::atomic<int64_t> rebalances{0};   // (local) chunks recast between size classes so a class cannot starve on a full budget
    };
    struct Vis {                 // a slot the GPU may read from now on (dev != 0), or no longer (dev == 0)
        int layer = 0, e = 0;
        unsigned long long dev = 0;
    };

    struct Cfg {
        int threads = 8;
        size_t budget_bytes = 0;           // the arena at most (grown in chunks as slots are first needed)
        size_t floor_bytes = 8ull << 30;   // RAM left alone
        Policy policy = kLru;
        float decay = 0.97f;               // lfu: heat x decay every `decay_every` accesses of its layer
        int decay_every = 240;
        std::function<uint64_t()> ram_avail;   // the RAM this process can still get (~0: unknown)
        // a new chunk of the arena for the GPU to read in place: returns the device address of `host` (null: not visible)
        std::function<void*(void* host, size_t bytes)> map_chunk;
        std::function<void(void* host)> unmap_chunk;
        size_t reserve = 192;              // slots kept evictable (not visible) for the window's demand and guesses
        int vis_percent = 100;             // the share of experts the GPU may take (by a fixed hash): the rest stay the CPU lane's, so both lanes work
    };

    void set_pfds(const std::vector<os::fd_t>& pfds) {   // (the fallback handles reopened or closed: swaps them into the
        // live table and closes an old one only once every read that had acquired it is done (hostile review 4): the
        // load may close a file's handles while the background store fill queues reads through them, and a closed
        // handle value gets recycled by the OS - to another shard, whose bytes would land in an expert slot.
        for (size_t i = 0; i < pfds_r_.size(); ++i) {
            const os::fd_t f = i < pfds.size() ? pfds[i] : os::kBadFd;
            if (pfds_r_[i].get() == f) continue;
            if (i < pfds_.size()) pfds_[i] = f;
            pfds_r_[i].replace(f);   // (the old value closes when its LAST in-flight read is done - see FdRef)
        }
        if (pfds_r_.empty()) pfds_ = pfds;   // (before init: no table yet, nothing queued, nothing to close)
    }   // (the fallback handles were reopened)
    bool fd_live() const { return !pfds_r_.empty(); }   // (the store owns the swap-and-close of the parallel handles)
    bool init(const Cfg& c, const std::vector<LayerGeo>& layers, const std::vector<os::fd_t>& dfds, const std::vector<os::fd_t>& pfds) {
        cfg_ = c;
        geo_ = layers;
        dfds_ = dfds;
        pfds_ = pfds;
        // (local) the handle tables a queued read points at instead of carrying a copy (FdRef): set_pfds swaps them
        // while the reader queue runs, so a load closing or reopening a file's handles can never strand one
        size_t nf = 0;
        for (const LayerGeo& g : geo_)
            for (int r = 0; r < 3; ++r) nf = std::max(nf, (size_t) g.file[r] + 1);
        pfds_r_.clear();
        pfds_r_.resize(nf);
        dfds_r_.clear();
        dfds_r_.resize(nf);
        for (size_t i = 0; i < nf; ++i) {
            pfds_r_[i].set(i < pfds_.size() ? pfds_[i] : os::kBadFd);
            dfds_r_[i].set(i < dfds_.size() ? dfds_[i] : os::kBadFd);
        }
        size_t blob = 0;
        for (const LayerGeo& g : geo_) blob = std::max<size_t>(blob, g.blob);
        slot_bytes_ = (blob + 2 * kSlack + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
        if (blob == 0 || c.budget_bytes < slot_bytes_) return false;
        // (local) size classes: a slot sized to its layer's blob, not the largest layer's - a mixed-bpw pack's layers differ by up
        // to 25%, and the uniform slot left ~14% of the budget empty.  A chunk's slots are of one class; an expert takes a slot of
        // its layer's class.  DS4_STORE_UNIFORM=1: one class (the largest), as before.
        const bool uniform = std::getenv("DS4_STORE_UNIFORM") && std::atoi(std::getenv("DS4_STORE_UNIFORM")) != 0;
        cls_of_.assign(geo_.size(), 0);
        cls_bytes_.clear();
        double mean = 0;
        int n_lay = 0;
        for (size_t l = 0; l < geo_.size(); ++l) {
            if (geo_[l].blob == 0) continue;
            const size_t sb = uniform ? slot_bytes_
                                      : (geo_[l].blob + 2 * kSlack + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
            size_t ci = 0;
            while (ci < cls_bytes_.size() && cls_bytes_[ci] != sb) ++ci;
            if (ci == cls_bytes_.size()) cls_bytes_.push_back(sb);
            cls_of_[l] = (int) ci;
            mean += (double) sb;
            ++n_lay;
        }
        if (n_lay == 0) return false;
        mean /= n_lay;
        size_t min_b = slot_bytes_;
        for (size_t b : cls_bytes_) min_b = std::min(min_b, b);
        budget_ = c.budget_bytes;
        nmax_ = (size_t) ((double) budget_ / mean);   // (every layer takes about as many: a token picks k experts in each)
        cap_ = budget_ / min_b + kChunk;
        // one reserved range for the whole arena (chunks committed as they are needed, each of one class's slots)
        arena_bytes_ = budget_ + kChunk * slot_bytes_ + (64u << 10);
        arena_ = (uint8_t*) os::range_reserve(arena_bytes_);
        if (arena_ == nullptr) return false;
        slots_.reset(new Slot[cap_]);
        free_hint_c_.assign(cls_bytes_.size(), 0);
        where_.resize(geo_.size());
        heat_.resize(geo_.size());
        hits_in_layer_.assign(geo_.size(), 0);
        for (size_t l = 0; l < geo_.size(); ++l) {
            where_[l].assign((size_t) geo_[l].n_exp, -1);
            heat_[l].assign((size_t) geo_[l].n_exp, 0.0f);
        }
        io_.reset(new IoQueue(std::max(1, c.threads)));
        grow(0);   // (the first chunk now: whether the GPU can map it is known from the start)
        // DS4_STORE_PREGROW=1 (default on Linux with the GPU mapping, =0 off): the whole arena now.  cuMemHostRegister (the GPU mapping of a chunk) taken while a kernel of
        // the decode is running waited on it for good on Linux (H100, driver 580): registering all chunks before the first
        // kernel avoids it
#if defined(__linux__)
        const bool pregrow = cfg_.map_chunk && !(std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) == 0);   // (Linux: on with the GPU mapping)
#else
        const bool pregrow = std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) != 0;
#endif
        if (pregrow)
            for (bool more = true; more;) {   // (the classes in turn)
                more = false;
                for (size_t ci = 0; ci < cls_bytes_.size(); ++ci) more = grow((int) ci) || more;
            }
        return true;
    }
    ~ExpertStore() {
        io_.reset();
        for (size_t i = 0; i < chunks_.size(); ++i)
            if (cfg_.unmap_chunk && chunk_dev_[i] != nullptr) cfg_.unmap_chunk(chunks_[i].first);
        os::range_release(arena_, arena_bytes_);
    }

    size_t slot_bytes() const { return slot_bytes_; }
    size_t slots_allocated() const { return nalloc_; }
    size_t slots_max() const { return nmax_; }
    size_t used_bytes() const { return used_; }               // (local) the arena committed so far
    size_t budget() const { return budget_; }
    size_t classes() const { return cls_bytes_.size(); }
    const Stats& stats() const { return st_; }
    double busy_ms() const { return io_ ? io_->busy_ms() : 0.0; }
    static constexpr size_t kSlack = 8192;

    /// The slot holding (layer, e) ready to read, or -1.  ADVISORY: it answers "a slot for this expert is here now"
    /// for residency probes (in_ram, the read-ahead's skip).  The slot can be evicted, rebalanced or shrunk the
    /// moment find() returns - never read its bytes from this.  Data access goes through acquire_ready()/unpin(),
    /// or through ensure()/StoreHold, which pin.  mu_ is held for the lookup: where_/state/layer/expert are
    /// written under it elsewhere, and an unsynchronized read of them could pair a fresh layer with an old slot
    /// (hostile review, second round 1.1).
    int find(int layer, int e) const {
        if (layer < 0 || layer >= (int) where_.size()) return -1;   // (a layer the store does not cover: the DSpark drafter's)
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        return sp.state.load(std::memory_order_acquire) == kReady && sp.layer == layer && sp.expert == e ? s : -1;
    }
    /// Data access: the ready slot holding (layer, e), atomically verified to belong to it and PINNED for the
    /// caller - no eviction, no rebalance, no shrink can touch it until unpin().  -1 when it is not here ready
    /// (the caller reads it another way: a demand fill, or the demand_read fallback).  A slot found ready and
    /// GPU-visible is safe to read either way (a visible slot is only ever taken back by sync(), which waits for
    /// the GPU's use; the caller still holds the pin across its own).
    int acquire_ready(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return -1;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        if (sp.state.load(std::memory_order_acquire) != kReady || sp.layer != layer || sp.expert != e) return -1;
        sp.pin++;
        sp.tick = ++clock_;
        sp.prefetch = false;
        return s;
    }
    /// Give back a slot from acquire_ready (or begin/wait_slots: those pin the same way and release through
    /// release(), which decrements under the same mutex).
    void unpin(int s) {
        if (s < 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if ((size_t) s < nalloc_ && slots_[(size_t) s].pin > 0) slots_[(size_t) s].pin--;
    }
    /// A slot's pin count > 0 (host tests, diagnostics): a pin held by acquire_ready/begin.  (Second review 1.1:
    /// the tests must see that a recast waits for the holder.)
    bool pinned(int s) const {
        std::lock_guard<std::mutex> lk(mu_);
        return s >= 0 && (size_t) s < nalloc_ && slots_[(size_t) s].pin > 0;
    }
    /// Slot entries still part of a size class (host tests, diagnostics): dead leftovers of a narrower recast
    /// (cls -1) are not counted, cannot be handed out, and must not be mistaken for free space.
    size_t live_slots() const {
        std::lock_guard<std::mutex> lk(mu_);
        size_t n = 0;
        for (size_t i = 0; i < nalloc_; ++i)
            if (slots_[i].cls >= 0) ++n;
        return n;
    }
    /// The demand fallback's read (hostile review, second round 1.2): the cache is starved and there is no mapped
    /// view to read through - the expert's bytes go into a caller-owned blob in the store's blob layout, through
    /// the caller's own synchronous read (at(file, off, len, dst): the engine hands its Pack::read_at, which works
    /// with the buffered handles closed - DROP_SHARDS - through the direct ones).  Independent of cache slots:
    /// this succeeds whenever the file is readable, which is exactly what rescue() needed and exit(3) replaced.
    /// False on a read failure - a real IO error the caller must report, not a starvation to retry.
    template <class ReadAt>
    bool demand_read(int layer, int e, uint8_t* blob, ReadAt&& at) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return false;
        const LayerGeo& g = geo_[(size_t) layer];
        bool ok = true;
        for (int r = 0; r < 3 && ok; ++r) ok = at(g.file[r], g.off[r] + (uint64_t) e * g.stride[r], g.bytes[r], blob + g.tr[r]);
        if (ok) st_.bytes_read.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
        return ok;
    }
    size_t blob_bytes(int layer) const { return layer >= 0 && layer < (int) geo_.size() ? (size_t) geo_[(size_t) layer].blob : 0; }
    const uint8_t* mat(int slot, int r) const { return slots_[(size_t) slot].base + geo_[(size_t) slots_[(size_t) slot].layer].tr[r]; }
    const uint8_t* blob(int slot) const { return slots_[(size_t) slot].base; }   // (only while the slot is pinned or visible: see acquire_ready)
    /// The device address of the slot's blob (0: the arena is not mapped for the GPU)
    unsigned long long dev_blob(int slot) const {
        const size_t c = (size_t) slots_[(size_t) slot].chunk;
        if (c >= chunk_dev_.size() || chunk_dev_[c] == nullptr) return 0;
        return (unsigned long long) ((uint8_t*) chunk_dev_[c] + (slots_[(size_t) slot].base - (uint8_t*) chunks_[c].first));
    }

    /// Makes the experts es[0..n) of `layer` ready and pinned (reading what is not here), waiting for the reads; the
    /// caller releases them.  False if a read failed or no slot could be had for one (ok[i] says which are fine; the caller
    /// reads the others another way).
    bool ensure(int layer, const int* es, int n, std::vector<char>& ok) {
        std::vector<int> sl;
        begin(layer, es, n, ok, sl);
        return wait_slots(sl, ok, 0, n);
    }
    /// The first half of ensure(): the experts' slots found or their reads started (pinned), no waiting.  sl[i] is the slot or -1.
    void begin(int layer, const int* es, int n, std::vector<char>& ok, std::vector<int>& sl) {
        ok.assign((size_t) n, 0);
        sl.assign((size_t) n, -1);
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i) {
            const int e = es[i];
            touch_heat(layer, e);
            int s = where_[(size_t) layer][(size_t) e];
            if (s >= 0) {
                Slot& sp = slots_[(size_t) s];
                try_ready(sp);
                if (sp.state.load() == kReady && !sp.prefetch) st_.hits.fetch_add(1, std::memory_order_relaxed);
                if (sp.prefetch) st_.prefetch_used.fetch_add(1, std::memory_order_relaxed);
                sp.prefetch = false;
                sp.pin++;
                sp.tick = ++clock_;
                sl[(size_t) i] = s;
                continue;
            }
            st_.misses.fetch_add(1, std::memory_order_relaxed);
            s = start_fill(layer, e, true);
            sl[(size_t) i] = s;
        }
    }
    /// True when the reads of slot sl (from begin) are all done.
    bool loaded(int s) const { return s >= 0 && slots_[(size_t) s].left.load(std::memory_order_acquire) <= 0; }
    /// The second half: waits for the slots sl[a..b) and marks ok[i] for the good ones.  False if any failed or had no slot.
    bool wait_slots(std::vector<int>& sl, std::vector<char>& ok, int a, int b) {
        const auto t0 = std::chrono::steady_clock::now();
        bool all = true;
        for (int i = a; i < b; ++i) {
            const int s = sl[(size_t) i];
            if (s < 0) {
                all = false;
                continue;
            }
            Slot& sp = slots_[(size_t) s];
            while (sp.left.load(std::memory_order_acquire) > 0) io_->wait_any(2000);
            std::lock_guard<std::mutex> lk(mu_);
            finish(sp);
            if (sp.state.load() == kReady) ok[(size_t) i] = 1;
            else {
                sp.pin--;
                sl[(size_t) i] = -1;
                all = false;
            }
        }
        st_.wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        return all;
    }
    /// Give back the pins held on sl (second review 1.1: by slot, not by a where_ lookup - the expert may have
    /// got a new slot in between, and decrementing that one would cut short another holder's pin).  Third round 5:
    /// the give-back keys on the SLOT LIST ALONE, not on ok[i]: a pin is held for exactly as long as sl[i] >= 0 -
    /// begin() took it when it named the slot, and wait_slots hands a failed one back itself and clears sl[i].
    /// ok[i] is the READ'S RESULT, and an early way out (the split lane's rescue reporting an IO error mid-hold)
    /// never waited for its other entries: gating the release on ok leaked those pins for good - a pinned slot is
    /// never evicted, never recast, and the store filled with slots nobody read from.  ok stays in the signature
    /// for the callers that pass it; ownership, not outcome, decides.
    void release(const std::vector<int>& sl, const std::vector<char>& /*ok*/) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < sl.size(); ++i)
            if (sl[(size_t) i] >= 0) {
                Slot& sp = slots_[(size_t) sl[(size_t) i]];
                if (sp.pin > 0) sp.pin--;
            }
    }
    /// Starts reading (layer, e) if it is neither here nor on its way (a guess: unpinned, may be given up before it is used).
    void prefetch(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return;
        std::lock_guard<std::mutex> lk(mu_);   // (the earlier residency peek was an unsynchronized where_ read - the same race the second review's 1.1 named; the locked check is just as cheap)
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        if (start_fill(layer, e, false) >= 0) st_.prefetched.fetch_add(1, std::memory_order_relaxed);
    }
    /// Reads the experts of `list` (layer, expert; hottest first) into free slots and returns when they are in (a start that
    /// is already warm: the profile's hottest, so the first tokens find them).  The first `list.size()` fit the arena less
    /// `leave` slots; returns how many were read.
    size_t preload(const std::vector<std::pair<int, int>>& list, size_t leave) {
        size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t room = nmax_ > leave ? nmax_ - leave : 0;
            const size_t take = std::min(list.size(), room);
            for (size_t k = take; k-- > 0;) {   // (coldest first: the hottest get the newest ticks)
                const int layer = list[k].first, e = list[k].second;
                if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) continue;
                if (where_[(size_t) layer][(size_t) e] >= 0) continue;
                const int s = start_fill(layer, e, false);
                if (s < 0) continue;   // (local: its class has no slot to give; the others still may)
                slots_[(size_t) s].prefetch = false;   // (warm, not a guess)
                ++n;
            }
        }
        for (size_t i = 0;; ++i) {   // until every read is done
            bool busy = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (size_t j = 0; j < nalloc_; ++j) {
                    Slot& sp = slots_[j];
                    if (sp.state.load(std::memory_order_acquire) != kLoading) continue;
                    if (sp.left.load(std::memory_order_acquire) > 0) busy = true;
                    else {
                        try_ready(sp);
                        if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                    }
                }
            }
            if (!busy) break;
            io_->wait_any(5000);
        }
        return n;
    }
    /// A slot for (layer, e) that the caller fills with the expert's blob itself (an expert the VRAM cache gives back to the CPU lane: its bytes
    /// are in memory already, so no read from the SSD while the lane waits).  Returns the blob's address, or nullptr (it is here already, or
    /// no slot can be had).  The slot is pinned and not findable until put_done().
    uint8_t* put_begin(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return nullptr;
        const int s = take_slot(false, cls_of_[(size_t) layer]);
        if (s < 0) return nullptr;
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = 1;
        sp.prefetch = false;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.left.store(1, std::memory_order_release);
        sp.state.store(kLoading, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        st_.put.fetch_add(1, std::memory_order_relaxed);
        return sp.base;
    }
    void put_done(int layer, int e, bool ok) {
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return;
        Slot& sp = slots_[(size_t) s];
        if (sp.pin > 0) sp.pin--;
        if (ok) {
            sp.left.store(0, std::memory_order_release);
            sp.state.store(kReady, std::memory_order_release);
        } else {
            where_[(size_t) layer][(size_t) e] = -1;
            sp.layer = sp.expert = -1;
            sp.left.store(0, std::memory_order_release);
            sp.state.store(kFree, std::memory_order_release);
            hint_down(s);
        }
    }
    /// Guesses still waiting in the queue for a layer the token is past: dropped unread (their slots free once the queue has gone by)
    void cancel_stale(int layer_now) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.prefetch && sp.pin == 0 && sp.layer >= 0 && sp.layer <= layer_now && sp.state.load(std::memory_order_acquire) == kLoading)
                sp.cancel.store(true, std::memory_order_release);
        }
    }
    /// An access the host saw for an expert the GPU computed (keeps it warm in the policy's eyes).
    void touch(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        touch_heat(layer, e);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].tick = ++clock_;
    }
    bool gpu_mapped() const { return !chunk_dev_.empty() && chunk_dev_[0] != nullptr; }
    /// The arena's address range (the device sees its host pointers as they are when the mapping is the identity)
    unsigned long long arena_lo() const { return (unsigned long long) (uintptr_t) arena_; }
    unsigned long long arena_hi() const { return (unsigned long long) (uintptr_t) (arena_ + arena_bytes_); }
    bool identity_mapped() const {
        for (size_t i = 0; i < chunks_.size(); ++i)
            if (chunk_dev_[i] != nullptr && chunk_dev_[i] != chunks_[i].first) return false;
        return true;
    }
    size_t max_blob() const { return slot_bytes_ - 2 * kSlack; }

    /// At a window boundary (nothing reading the slots; the caller applies `ev` to the device's table before the next window):
    /// every ready slot not visible yet becomes visible (`gpu_ok(layer, e)`: false for an expert VRAM holds), and visible
    /// slots of the lowest score are taken back until `cfg.reserve` slots can be given up.  Returns the number of events.
    size_t sync(std::vector<Vis>& ev, const std::function<bool(int, int)>& gpu_ok) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t nvis = 0, nfree = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.cls < 0) continue;   // (dead after a rebalance recast a chunk narrower: kFree but the allocator can never hand it out - counting it as free would fake the reserve and skip the take-backs)
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
            }
            const int st = sp.state.load(std::memory_order_acquire);
            if (st == kFree) {
                ++nfree;
                continue;
            }
            if (st != kReady) continue;
            if (sp.visible) {
                ++nvis;
                continue;
            }
            if (sp.pin == 0) ++nfree;
            if (sp.pin == 0 && dev_blob((int) i) != 0 && gpu_ok(sp.layer, sp.expert) && !sp.prefetch && vis_hash(sp.layer, sp.expert) < cfg_.vis_percent) {
                sp.visible = true;
                ++nvis;
                if (sp.pin == 0) --nfree;
                ev.push_back({sp.layer, sp.expert, dev_blob((int) i)});
            }
        }
        // give slots back until the reserve is there: the visible ones with the lowest score
        if (nfree < cfg_.reserve && nvis > 0) {
            std::vector<std::pair<double, size_t>> cand;
            cand.reserve(nvis);
            for (size_t i = 0; i < nalloc_; ++i)
                if (slots_[i].visible && slots_[i].state.load(std::memory_order_acquire) == kReady) cand.push_back({score(slots_[i]), i});
            const size_t want = std::min(cand.size(), cfg_.reserve - nfree);
            std::partial_sort(cand.begin(), cand.begin() + (long) want, cand.end());
            for (size_t k = 0; k < want; ++k) {
                Slot& sp = slots_[cand[k].second];
                sp.visible = false;
                ev.push_back({sp.layer, sp.expert, 0ull});
            }
        }
        return ev.size();
    }
    /// RAM is short: gives back whole chunks from the end of the arena (their experts forgotten) - up to `want`, stopping at one
    /// with a slot in use (pinned, loading, or GPU-visible).  Returns the chunks given back.
    size_t shrink_chunks(size_t want) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t done = 0;
        while (done < want && !chunks_.empty()) {
            const size_t c = chunks_.size() - 1, first = chunk_first_[c];
            bool busy = false;
            for (size_t i = first; i < nalloc_ && !busy; ++i) {
                Slot& sp = slots_[i];
                if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {   // (read by now)
                    try_ready(sp);
                    if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                }
                busy = sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) == kLoading;
            }
            if (busy) break;
            for (size_t i = first; i < nalloc_; ++i) {
                Slot& sp = slots_[i];
                if (sp.layer >= 0 && sp.expert >= 0) {
                    where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
                    st_.evictions.fetch_add(1, std::memory_order_relaxed);
                }
                sp.layer = sp.expert = -1;
                sp.prefetch = false;
                sp.state.store(kFree, std::memory_order_release);
                sp.base = nullptr;
            }
            if (cfg_.unmap_chunk && chunk_dev_[c] != nullptr) cfg_.unmap_chunk(chunks_[c].first);
            os::range_decommit(chunks_[c].first, chunks_[c].second);
            used_ -= chunks_[c].second;
            chunks_.pop_back();
            chunk_dev_.pop_back();
            chunk_first_.pop_back();
            nalloc_ = first;
            for (size_t& h : free_hint_c_) h = std::min(h, first);
            ++done;
            ++shrunk_;
        }
        return done;
    }
    size_t shrunk() const { return shrunk_; }
    /// An expert VRAM took over: no longer a visible slot
    void unmark(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].visible = false;
    }
    size_t visible_count() const {
        size_t n = 0;
        for (size_t i = 0; i < nalloc_; ++i) n += slots_[i].visible ? 1 : 0;
        return n;
    }

private:
    static constexpr int kFree = 0, kLoading = 1, kReady = 2, kFailed = 3;
    // slots a chunk of the arena (the unit the GPU mapping registers; DS4_STORE_CHUNK_SLOTS: a test knob for the registration segment size)
    static size_t chunk_slots() {
        const char* e = std::getenv("DS4_STORE_CHUNK_SLOTS");
        const long v = e ? std::atol(e) : 0;
        return v >= 8 && v <= 4096 ? (size_t) v : 64;
    }
    const size_t kChunk = chunk_slots();
    struct Slot {
        uint8_t* base = nullptr;   // the blob's first byte
        int layer = -1, expert = -1, pin = 0;
        int cls = 0, chunk = 0;    // (local) its size class and its chunk
        bool prefetch = false, visible = false;
        std::atomic<int> state{kFree}, left{0};
        std::atomic<bool> bad{false}, cancel{false};
        uint64_t tick = 0;
    };

    static int vis_hash(int layer, int e) { return (int) (((unsigned) layer * 2654435761u + (unsigned) e * 40503u) >> 7) % 100; }
    double score(const Slot& sp) const {
        if (cfg_.policy == kLru) return (double) sp.tick;
        return (double) heat_[(size_t) sp.layer][(size_t) sp.expert] + 1e-9 * (double) sp.tick;
    }
    void touch_heat(int layer, int e) {
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy != kLru && ++hits_in_layer_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0)
            for (float& h : heat_[(size_t) layer]) h *= cfg_.decay;   // (a linear pass over 384 floats every 240 accesses of that layer)
    }
    // a slot for (layer, e) and its reads started; -1 if there is none to be had.  mu_ held.
    int start_fill(int layer, int e, bool demand) {
        int s = take_slot(demand, cls_of_[(size_t) layer]);
        if (s < 0) {
            st_.no_slot.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = demand ? 1 : 0;
        sp.prefetch = !demand;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.state.store(kLoading, std::memory_order_release);
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.left.store(3, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        const LayerGeo& g = geo_[(size_t) layer];
        IoQueue::Job jobs[3];
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
            const uint64_t lead = off % os::kDirectAlign;
            uint8_t* fin = sp.base + g.tr[r];   // where the matrix belongs
            // the aligned read lands the file's byte `off` at dst + lead: dst is fin - lead rounded down to 4096 (its span spills
            // under 4096 bytes either way into the blob's gaps), the matrix is then moved up to fin
            uint8_t* dst = (uint8_t*) ((uintptr_t) (fin - lead) & ~(uintptr_t) (os::kDirectAlign - 1));
            IoQueue::Job& j = jobs[r];
            j.bref = &pfds_r_[(size_t) g.file[r]];   // (handles resolved at dispatch: see FdRef)
            j.dref = &dfds_r_[(size_t) g.file[r]];
            j.off = off;
            j.len = g.bytes[r];
            j.dst = dst;
            j.keep = fin;
            j.left = &sp.left;
            j.bad = &sp.bad;
            j.cancel = demand ? nullptr : &sp.cancel;
        }
        st_.bytes_read.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
        io_->push(jobs, 3, !demand);   // (a guess waits behind every demand read)
        return s;
    }
    // loading -> ready once its reads are done and whole (lock-free; a failed one is cleaned up by finish / take_slot)
    void try_ready(Slot& sp) const {
        int want = kLoading;
        if (sp.left.load(std::memory_order_acquire) == 0 && !sp.bad.load(std::memory_order_acquire))
            sp.state.compare_exchange_strong(want, kReady, std::memory_order_acq_rel);
    }
    void finish(Slot& sp) {   // mu_ held; reads done
        if (sp.state.load() != kLoading) return;
        if (sp.bad.load()) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            sp.layer = sp.expert = -1;
            sp.state.store(kFree, std::memory_order_release);
            sp.prefetch = false;
            if (sp.cancel.load()) st_.cancelled.fetch_add(1, std::memory_order_relaxed);
            else st_.fails.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        sp.state.store(kReady, std::memory_order_release);
    }
    bool grow(int cls) {   // a chunk of class cls's slots, inside the byte budget
        if (nalloc_ >= cap_ || cls < 0 || cls >= (int) cls_bytes_.size()) return false;
        const size_t sb = cls_bytes_[(size_t) cls];
        const size_t n = std::min(std::min(kChunk, cap_ - nalloc_), (budget_ > used_ ? budget_ - used_ : 0) / sb);
        if (n == 0) return false;
        const size_t bytes = n * sb;
        const uint64_t avail = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (avail != ~uint64_t{0} && avail < bytes + cfg_.floor_bytes) return false;
        void* p = arena_ + used_;
        if (!os::range_commit(p, bytes)) return false;
        chunks_.push_back({p, bytes});
        chunk_first_.push_back(nalloc_);
        void* dev = cfg_.map_chunk ? cfg_.map_chunk(p, bytes) : nullptr;
        chunk_dev_.push_back(dev);
        for (size_t i = 0; i < n; ++i) {
            Slot& sp = slots_[nalloc_ + i];
            sp.base = (uint8_t*) p + i * sb + kSlack;
            sp.cls = cls;
            sp.chunk = (int) chunks_.size() - 1;
        }
        nalloc_ += n;
        used_ += bytes;
        return true;
    }
    void hint_down(int s) {
        const int c = slots_[(size_t) s].cls;
        if (c >= 0 && c < (int) free_hint_c_.size()) free_hint_c_[(size_t) c] = std::min(free_hint_c_[(size_t) c], (size_t) s);
    }
    // the slot to give up (or a free one): the policy's lowest among the unpinned, not visible, ready ones.  mu_ held.
    int take_slot(bool demand, int cls) {
        size_t& hint = free_hint_c_[(size_t) cls];
        for (size_t i = hint; i < nalloc_; ++i)
            if (slots_[i].cls == cls && slots_[i].state.load(std::memory_order_acquire) == kFree) {
                hint = i + 1;
                return (int) i;
            }
        hint = nalloc_;
        if (grow(cls)) return take_slot(demand, cls);
        int best = -1;
        double bv = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                if (sp.state.load(std::memory_order_acquire) == kFree) {
                    hint_down((int) i);
                    if (sp.cls == cls) return (int) i;
                }
            }
            if (sp.cls != cls || sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) != kReady) continue;
            const double v = score(sp);
            if (best < 0 || v < bv) {
                best = (int) i;
                bv = v;
            }
        }
        if (best < 0) {
            // (local) the class cannot feed itself: it grows nothing (the byte budget lies committed in other
            // classes' chunks) and has nothing of its own to evict (all pinned, visible, loading, or none yet).
            // Under the uniform store this state could not outlast one eviction; with size classes it can be
            // permanent, and under STRATA_DS4_DROP_SHARDS (no mapped view to fall back on) a store that will
            // never hold an expert makes the engine's rescue loop give the process up. Reclaim instead: give
            // this class a cold chunk of another one, whole.
            if (rebalance(cls) >= 0) return take_slot(demand, cls);
            return -1;
        }
        return best;
    }

    // (local) Class starvation relief: the coldest chunk whose every slot is free, or ready and neither pinned
    // nor GPU-visible, is recast as slots of class `cls` - the same committed bytes, another class's shape.
    // A chunk of the wanted class is never recast (feeding this class from itself is the eviction loop's job,
    // and a chunk full of pinned or loading slots gives nothing up). The recast group cannot be wider than
    // the old one: later chunks' first entries are fixed, so leftover entries die (cls -1: every scan skips
    // them, the bytes stay committed). Only shrink_chunks, which ends the arena, may retire entries - a dead
    // middle entry costs nothing but its scan step.  Returns the group's first index, or -1.  mu_ held.
    int rebalance(int cls) {
        const size_t sb = cls_bytes_[(size_t) cls];
        int best = -1;
        double bs = 0.0;
        for (size_t c = 0; c < chunks_.size(); ++c) {
            const size_t first = chunk_first_[c], last = c + 1 < chunks_.size() ? chunk_first_[c + 1] : nalloc_;
            if (first >= last || sb > chunks_[c].second) continue;   // (not even one wanted slot fits)
            double s = 0.0;
            bool ours = false, can = true;
            for (size_t i = first; i < last && can; ++i) {
                Slot& sp = slots_[i];
                if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {   // (read by now: land it)
                    try_ready(sp);
                    if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                }
                const int st = sp.state.load(std::memory_order_acquire);
                if (st == kLoading) { can = false; break; }
                if (sp.cls == cls) ours = true;
                if (st == kReady) {
                    if (sp.pin > 0 || sp.visible) { can = false; break; }
                    s += score(sp);
                }
            }
            if (!can || ours) continue;
            if (best < 0 || s < bs || (s == bs && (size_t) best < c)) { best = (int) c; bs = s; }   // (ties: the later chunk - the arena's head is the older, usually hotter, ground)
        }
        if (best < 0) return -1;
        const size_t c = (size_t) best, first = chunk_first_[c], last = chunk_first_[c] + (chunk_span(c));
        void* p = chunks_[c].first;
        const size_t n = std::min(last - first, chunks_[c].second / sb);
        for (size_t i = first; i < last; ++i) {
            Slot& sp = slots_[i];
            if (sp.layer >= 0 && sp.expert >= 0) {   // (what it held is forgotten: the whole point of the reclaim)
                where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
                st_.evictions.fetch_add(1, std::memory_order_relaxed);
            }
            sp.layer = sp.expert = -1;
            sp.pin = 0;
            sp.prefetch = false;
            sp.visible = false;
            sp.bad.store(false);
            sp.cancel.store(false);
            sp.left.store(0);
            sp.tick = 0;
            if (i - first < n) {
                sp.base = (uint8_t*) p + (i - first) * sb + kSlack;
                sp.cls = cls;
                sp.chunk = (int) c;
            } else {
                sp.base = nullptr;
                sp.cls = -1;
            }
            sp.state.store(kFree, std::memory_order_release);
        }
        st_.rebalances.fetch_add(1, std::memory_order_relaxed);
        free_hint_c_[(size_t) cls] = std::min(free_hint_c_[(size_t) cls], first);   // (the scan must reach the new slots)
        return (int) first;
    }

    size_t chunk_span(size_t c) const {
        return (c + 1 < chunks_.size() ? chunk_first_[c + 1] : nalloc_) - chunk_first_[c];
    }

    Cfg cfg_;
    std::vector<LayerGeo> geo_;
    std::vector<os::fd_t> dfds_, pfds_;
    std::deque<FdRef> dfds_r_, pfds_r_;   // (local) the handles the reader queue resolves at dispatch (set_pfds swaps them live)
    size_t slot_bytes_ = 0, nmax_ = 0, nalloc_ = 0, arena_bytes_ = 0;
    size_t budget_ = 0, used_ = 0, cap_ = 0;          // (local) the byte budget, the bytes committed, the slot table's size
    std::vector<int> cls_of_;                         // (local) a layer's size class
    std::vector<size_t> cls_bytes_, free_hint_c_, chunk_first_;
    uint8_t* arena_ = nullptr;
    size_t shrunk_ = 0;
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::pair<void*, size_t>> chunks_;
    std::vector<void*> chunk_dev_;
    std::vector<std::vector<int>> where_;
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> hits_in_layer_;
    uint64_t clock_ = 0;
    mutable std::mutex mu_;   // (find() is a const advisory probe and takes it too - the metadata it reads is written under this lock)
    std::unique_ptr<IoQueue> io_;
    Stats st_;
};

}  // namespace strata::ds4
