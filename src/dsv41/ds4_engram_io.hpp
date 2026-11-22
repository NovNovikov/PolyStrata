// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_engram_io.hpp - DeepSeek's Engram (n-gram memory) rows read off the decode's critical path.
//
// A token's Engram rows depend only on the token ids, so they are asked for as soon as the window's tokens are known
// and collected only right before the layer that adds them (layer 1's hide behind the embedding and layer 0, layer
// 14's behind layers 0-13).  The reads run on the reader's own threads, each with its own handles to the model's files
// (on Windows the ReadFile calls on one synchronous handle run one at a time), FIFO: the first Engram layer's rows
// first.  A bounded row cache (clock eviction) keeps rows already fetched: a row that recurs (the same n-gram again,
// in the prompt or later) is a copy, not a read.  The rows are the file's bytes either way, so the output is the same.
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4 {

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Rows already fetched: key -> `bytes` of payload, at most `cap` rows (grown on demand in blocks), clock eviction.
/// Not thread-safe (the caller locks).
class EgRowCache {
public:
    void init(size_t cap, size_t bytes) {
        cap_ = cap;
        bytes_ = bytes;
        map_.reserve(std::min<size_t>(cap, 1u << 20));
    }
    size_t capacity() const { return cap_; }
    size_t size() const { return keys_.size(); }
    /// The row's payload, or nullptr (a hit marks it recently used).
    const uint8_t* find(uint64_t key) {
        const auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        ref_[it->second] = 1;
        return at(it->second);
    }
    /// Room for key's payload (the caller fills it): a new slot while below capacity, else the clock's victim.
    uint8_t* put(uint64_t key) {
        if (cap_ == 0 || bytes_ == 0) return nullptr;
        const auto it = map_.find(key);
        if (it != map_.end()) return at(it->second);
        uint32_t s;
        if (keys_.size() < cap_) {
            s = (uint32_t) keys_.size();
            if (s % kBlock == 0) blocks_.emplace_back(new uint8_t[(size_t) kBlock * bytes_]);
            keys_.push_back(key);
            ref_.push_back(0);
        } else {
            while (ref_[hand_]) {   // (a second chance for rows used since the hand last passed)
                ref_[hand_] = 0;
                hand_ = (hand_ + 1) % (uint32_t) cap_;
            }
            s = hand_;
            hand_ = (hand_ + 1) % (uint32_t) cap_;
            map_.erase(keys_[s]);
            keys_[s] = key;
            ref_[s] = 0;
        }
        map_.emplace(key, s);
        return at(s);
    }

private:
    static constexpr uint32_t kBlock = 4096;
    uint8_t* at(uint32_t s) { return blocks_[s / kBlock].get() + (size_t) (s % kBlock) * bytes_; }
    size_t cap_ = 0, bytes_ = 0;
    uint32_t hand_ = 0;
    std::unordered_map<uint64_t, uint32_t> map_;
    std::vector<uint64_t> keys_;
    std::vector<uint8_t> ref_;
    std::vector<std::unique_ptr<uint8_t[]>> blocks_;
};

/// One row to fetch: its bytes to dst, its scales to scales.  mx8: n bytes at off of a file, [scale, 32 codes] blocks
/// (a GGUF's MXFP8 row); else n row bytes at off and n2 scale bytes at off2 of file2 (two reads; n2 0: none - a GGUF's
/// Q6_K or IQ4_NL row, their scales inside their n bytes: 210 a 256 values, 18 a 32).
struct EgRead {
    int file = -1;
    uint64_t off = 0;
    uint32_t n = 0;
    uint8_t* dst = nullptr;
    uint8_t* scales = nullptr;
    bool mx8 = false;
    int file2 = -1;
    uint64_t off2 = 0;
    uint32_t n2 = 0;
    uint64_t key = 0;
};

/// A batch of rows (one Engram layer of a window): reads, the other places a row read once goes, and its state.
struct EgBatch {
    std::vector<EgRead> reads;
    std::vector<std::pair<int, std::pair<uint8_t*, uint8_t*>>> dups;   // read i -> another (codes, scales)
    std::vector<uint8_t> ok;                                            // per read: 1 read whole
    std::atomic<int> left{0};                                           // reads not yet done
    std::atomic<int> post{0};                                           // 0 pending, 1 finishing, 2 finished
    std::atomic<size_t> bad_n{0};                                       // (second review 4.1) failed reads of a finished batch:
                                                                        //   sticky, so every later finisher of this batch - and
                                                                        //   a caller that consumes its rows - sees them
    int64_t t0 = 0;                                                     // submitted (now_ns())
    std::atomic<int64_t> t1{0};                                         // the last read done
    bool live = false;                                                  // submitted and not yet finished
    void clear() {
        reads.clear();
        dups.clear();
        ok.clear();
        left.store(0);
        post.store(0);
        bad_n.store(0);
        live = false;
    }
};

/// The reader's threads: each read on whichever is free, in submission order, with that thread's own file handles.
class EgReader {
public:
    ~EgReader() { stop(); }
    /// direct[i]: file i read unbuffered (Windows, a file nothing maps: os::open_unbuffered), else buffered
    void start(const std::vector<std::string>& paths, int threads, std::vector<char> direct = {}) {
        paths_ = paths;
        direct_ = std::move(direct);
        for (int i = 0; i < threads; ++i) th_.emplace_back([this] { loop(); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
        if (check_n_.load() > 0)
            std::fprintf(stderr, "ds4 Engram check (DS4_EG_CHECK): %lld rows read unbuffered, %lld differed from a buffered read\n",
                         (long long) check_n_.load(), (long long) check_bad_.load());
    }
    int threads() const { return (int) th_.size(); }
    void submit(EgBatch& b) {
        b.ok.assign(b.reads.size(), 0);
        b.post.store(0);
        b.t0 = now_ns();
        b.t1.store(b.t0);
        b.live = true;
        b.left.store((int) b.reads.size(), std::memory_order_release);
        const int n = (int) b.reads.size();
        if (n == 0) return;
        // a token's rows one job each; a prompt chunk's in runs (fewer trips through the queue's lock)
        const int step = std::max(1, std::min(32, n / (int) (th_.size() * 4)));
        int jobs = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; i += step, ++jobs) q_.push_back({&b, i, std::min(n, i + step)});
        }
        if (jobs >= (int) th_.size()) cv_.notify_all();
        else
            for (int i = 0; i < jobs; ++i) cv_.notify_one();
    }
    /// Until every read of b is done.
    void wait(EgBatch& b) {
        if (b.left.load(std::memory_order_acquire) == 0) return;
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return b.left.load(std::memory_order_acquire) == 0; });
    }

private:
    void loop() {
        std::vector<os::fd_t> fds(paths_.size(), os::kBadFd);   // this thread's own handles, opened on first use
        std::vector<os::fd_t> ufds(paths_.size(), os::kBadFd);  // (and its unbuffered ones)
        std::vector<uint8_t> buf, abuf(3 * os::kDirectAlign);
        uint8_t* al = abuf.data() + (os::kDirectAlign - (uintptr_t) abuf.data() % os::kDirectAlign) % os::kDirectAlign;
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
                if (quit_ && q_.empty()) break;
                job = q_.front();
                q_.pop_front();
            }
            EgBatch& b = *job.b;
            for (int i = job.i0; i < job.i1; ++i) b.ok[(size_t) i] = read(fds, ufds, al, b.reads[(size_t) i], buf) ? 1 : 0;
            const int64_t t = now_ns();
            for (int64_t prev = b.t1.load(); t > prev && !b.t1.compare_exchange_weak(prev, t);) {
            }
            if (b.left.fetch_sub(job.i1 - job.i0, std::memory_order_acq_rel) == job.i1 - job.i0) {   // (b is the waiter's again)
                std::lock_guard<std::mutex> lk(done_mu_);
                done_cv_.notify_all();
            }
        }
        for (os::fd_t f : fds) os::close_fd(f);
        for (os::fd_t f : ufds) os::close_fd(f);
    }
    bool read(std::vector<os::fd_t>& fds, std::vector<os::fd_t>& ufds, uint8_t* al, const EgRead& r,
              std::vector<uint8_t>& buf) {
        if ((r.mx8 || r.n2 == 0) && r.file >= 0 && (size_t) r.file < direct_.size() && direct_[(size_t) r.file] &&
            os::direct_span(r.off, r.n) <= 2 * os::kDirectAlign) {
            os::fd_t& u = ufds[(size_t) r.file];
            if (u == os::kBadFd) u = os::open_unbuffered(paths_[(size_t) r.file]);
            if (u != os::kBadFd && os::pread_unbuffered(u, al, r.n, r.off)) {
                const uint8_t* src = al + r.off % os::kDirectAlign;
                static const bool check = std::getenv("DS4_EG_CHECK") != nullptr;   // (each row read buffered too)
                if (check) {
                    std::vector<uint8_t> ref(r.n);
                    os::fd_t& f2 = fds[(size_t) r.file];
                    if (f2 == os::kBadFd) f2 = os::open_ro(paths_[(size_t) r.file]);
                    const bool ok = f2 != os::kBadFd && os::pread_full(f2, ref.data(), r.n, r.off);
                    if (!ok || std::memcmp(ref.data(), src, r.n) != 0) ++check_bad_;
                    ++check_n_;
                }
                if (!r.mx8) {   // (a GGUF's Q6_K / IQ4_NL row: its bytes as they are, the scales inside)
                    std::memcpy(r.dst, src, r.n);
                    return true;
                }
                for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
                    r.scales[k] = src[(size_t) k * 33];
                    std::memcpy(r.dst + (size_t) k * 32, src + (size_t) k * 33 + 1, 32);
                }
                return true;
            }   // (else buffered, below)
        }
        auto fd = [&](int file) {
            if (file < 0 || file >= (int) fds.size()) return os::kBadFd;
            if (fds[(size_t) file] == os::kBadFd) fds[(size_t) file] = os::open_ro(paths_[(size_t) file]);
            return fds[(size_t) file];
        };
        const os::fd_t f = fd(r.file);
        if (f == os::kBadFd) return false;
        if (!r.mx8) {
            if (!os::pread_full(f, r.dst, r.n, r.off)) return false;
            if (r.n2 == 0) return true;
            const os::fd_t f2 = fd(r.file2);
            return f2 != os::kBadFd && os::pread_full(f2, r.scales, r.n2, r.off2);
        }
        buf.resize(r.n);
        if (!os::pread_full(f, buf.data(), r.n, r.off)) return false;
        for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
            r.scales[k] = buf[(size_t) k * 33];
            std::memcpy(r.dst + (size_t) k * 32, buf.data() + (size_t) k * 33 + 1, 32);
        }
        return true;
    }

    std::vector<std::string> paths_;
    std::vector<char> direct_;

public:
    std::atomic<int64_t> check_n_{0}, check_bad_{0};   // (DS4_EG_CHECK: rows compared, rows that differed)
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    struct Job {
        EgBatch* b = nullptr;
        int i0 = 0, i1 = 0;   // reads [i0, i1)
    };
    std::deque<Job> q_;
    bool quit_ = false;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
};

/// The Engram ring's book: n slots, unit u's rows into slot u % n, a slot refilled only once unit u - n's rows
/// have been copied out of it (the engine records the copy on a CUDA event and waits on it - the ring says when
/// the copy was recorded, the event wait stays the engine's). Pure bookkeeping, so the wraparound, the stopping
/// and the reset-on-resize are testable off-device. reset() to a bigger slot size happens only after every copy
/// of the old slots is done (the engine syncs the events first) - slots are never reused across a reset.
class EgRing {
public:
    void reset(size_t slots_n, size_t bytes) {
        n_ = slots_n;
        bytes_ = bytes;
        done_.store(0, std::memory_order_relaxed);
        used_.store(0, std::memory_order_relaxed);
    }
    size_t slots() const { return n_; }
    size_t bytes() const { return bytes_; }
    size_t slot_of(size_t u) const { return u % n_; }
    /// The producer may refill unit u's slot: false while unit u - n's copy has not been recorded yet.
    bool can_fill(size_t u) const { return u < n_ || used_.load(std::memory_order_acquire) + n_ > u; }
    /// The producer's: unit u's rows are in the slot.
    void filled(size_t u) { done_.store(u + 1, std::memory_order_release); }
    /// The consumer's: unit u's rows are in place (ready to copy to the device).
    bool ready(size_t u) const { return done_.load(std::memory_order_acquire) > u; }
    /// The consumer's: unit u's H2D copy is recorded (the slot may be refilled once that copy is done).
    void copied(size_t u) { used_.store(u + 1, std::memory_order_release); }

private:
    size_t n_ = 0, bytes_ = 0;
    std::atomic<size_t> done_{0}, used_{0};
};

/// Finish a batch (the engine's eg_finish without the engine): wait for its reads, then serve the rows the batch
/// reads once but needs twice - and a row whose read failed reads as zeros EVERYWHERE it lands, never the slot's
/// stale bytes (the ported order copied the dups before zeroing the failure, so a repeated row of a failed read
/// got garbage). Whole rows go to the cache; failed ones do not, so a later window retries the file. on_fail
/// fires per failed read (the engine warns once a run through it, counting the rest). The first caller finishes;
/// a second one waits for it. Returns the failed reads (0 when the batch was finished already).
/// `retry` (second review 4.1) gets every read whose reader failed BEFORE it is zeroed: an engine hook reading
/// the same row through the engine's own Pack handles - the reader's private handle failing is often transient
/// (the unbuffered path under contention), and the row belongs to a file the engine can still read. A retry that
/// lands fills r.dst/r.scales and counts as a good row (cached, not zeroed, not failed); one that fails leaves
/// the zeros, and the caller must treat them as an error, never as an answer: the engine stops the run and the
/// API reports the request's failure, so failed rows are never published as ready data.
inline size_t eg_complete(EgBatch& b, EgReader& io, EgRowCache& cache, std::mutex& cache_mu, uint32_t hd,
                          const std::function<void(const EgRead&)>& on_fail,
                          const std::function<bool(const EgRead&)>& retry = nullptr) {
    if (b.post.load(std::memory_order_acquire) == 2) return b.bad_n.load(std::memory_order_relaxed);
    io.wait(b);
    int want = 0;
    if (!b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
        while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        return b.bad_n.load(std::memory_order_relaxed);   // (sticky: a batch whose rows are zeros stays bad)
    }
    size_t bad = 0;
    for (size_t i = 0; i < b.reads.size(); ++i)
        if (!b.ok[i]) {
            if (retry && retry(b.reads[i])) {   // (second review 4.1) the engine's own read: a transient reader failure ends here
                b.ok[i] = 1;
                continue;
            }
            ++bad;
            EgRead& r = b.reads[i];
            std::memset(r.dst, 0, hd);
            std::memset(r.scales, 0, hd / 32);
            if (on_fail) on_fail(r);
        }
    for (const auto& d : b.dups) {   // (after the zeroing: a failed source zeroes its dups too)
        const EgRead& r = b.reads[(size_t) d.first];
        std::memcpy(d.second.first, r.dst, hd);
        std::memcpy(d.second.second, r.scales, hd / 32);
    }
    {
        std::lock_guard<std::mutex> lk(cache_mu);
        for (size_t i = 0; i < b.reads.size(); ++i) {
            const EgRead& r = b.reads[i];
            if (!b.ok[i]) continue;   // (zeros are not the file's bytes: never cached)
            if (uint8_t* c = cache.put(r.key)) {
                std::memcpy(c, r.dst, hd);
                std::memcpy(c + hd, r.scales, hd / 32);
            }
        }
    }
    b.live = false;
    b.bad_n.store(bad, std::memory_order_relaxed);   // (before post=2: a later finisher of this batch reads it back)
    b.post.store(2, std::memory_order_release);
    return bad;
}

}  // namespace strata::ds4
