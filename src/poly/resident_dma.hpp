// Register the original resident bytes for CUDA/HIP DMA, never a second host staging copy.
#pragma once
#include "dma_ranges.hpp"
#include "dma_cost.hpp"
#include "transfer_audit_cuda.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata::poly {

class ResidentDma {
    struct Use { cudaStream_t stream; cudaEvent_t done; bool pending = false; };
    uint8_t* base_;
    size_t arena_, limit_, page_, peak_ = 0;
    std::vector<DmaRange> pinned_;
    std::map<size_t, uint64_t> ages_;
    uint64_t age_ = 0;
    uint64_t registered_bytes_ = 0, unregistered_bytes_ = 0;
    DmaCost cost_;
    bool trace_ = audit::enabled() || std::getenv("STRATA_TRACE_STAGING");
    std::vector<Use> uses_;
    using Clock = std::chrono::steady_clock;
    static double ms_since(Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    }
    static void check(cudaError_t e, const std::string& what) {
        if (e == cudaSuccess) return;
        (void) cudaGetLastError();
        throw std::runtime_error("no-disk: " + what + ": " + cudaGetErrorString(e));
    }
    void drain() {
        const auto t0 = trace_ ? Clock::now() : Clock::time_point{};
        for (auto& u : uses_)
            check(u.pending ? PA_WAIT_STREAM(u.stream) : PA_WAIT_EVENT(u.done), "waiting for source DMA");
        if (trace_) cost_.drain_ms += ms_since(t0);
    }
    void verify(void* p) {
        audit::HostSpan audit_span(__FILE__, __LINE__, "registration-verify");
#ifdef HIP_VERSION
        unsigned flags = 0;
        check(hipHostGetFlags(&flags, p), "verifying HIP-registered RAM");
#else
        cudaPointerAttributes a{};
        check(cudaPointerGetAttributes(&a, p), "verifying CUDA-registered RAM");
        if (a.type != cudaMemoryTypeHost) throw std::runtime_error("no-disk: DMA source is not CUDA Host memory");
#endif
    }
public:
    static size_t page_size() {
#ifdef _WIN32
        SYSTEM_INFO s{}; GetSystemInfo(&s); return s.dwPageSize;
#else
        const long n = sysconf(_SC_PAGESIZE);
        if (n <= 0) throw std::runtime_error("no-disk: cannot determine DMA page size");
        return (size_t) n;
#endif
    }
    ResidentDma(void* base, size_t arena, size_t limit)
        : base_((uint8_t*) base), arena_(arena), limit_(limit), page_(page_size()) {
        if (!base_ || !arena_ || (uintptr_t) base_ % page_)
            throw std::runtime_error("no-disk: DMA arena must be page-aligned and nonempty");
    }
    ResidentDma(const ResidentDma&) = delete;
    ResidentDma& operator=(const ResidentDma&) = delete;
    ~ResidentDma() {
        // Also fence a partly issued copy whose exception prevented end() from recording its event.
        for (const auto& u : uses_) {
            if (u.pending) (void) PA_WAIT_STREAM(u.stream);
            else (void) PA_WAIT_EVENT(u.done);
        }
        for (const auto& r : pinned_) (void) audit::unreg(__FILE__, __LINE__, base_ + r.first, r.second, r.first);
        for (const auto& u : uses_) (void) cudaEventDestroy(u.done);
        (void) cudaGetLastError();
    }
    const uint8_t* base() const { return base_; }
    size_t limit() const { return limit_; }
    size_t bytes() const { size_t n = 0; for (const auto& r : pinned_) n += r.second; return n; }
    size_t peak() const { return peak_; }
    uint64_t registered_bytes() const { return registered_bytes_; }
    uint64_t unregistered_bytes() const { return unregistered_bytes_; }
    const DmaCost& cost() const { return cost_; }
    // Opt-in prefill policy. A whole-layer window prepared before attention also covers all
    // later routed subsets; do not tear it down just because the subset is smaller.
    void prepare_stage(std::vector<DmaRange> sources, bool retain_window = false) {
        auto wanted = dma_stage_ranges(std::move(sources), arena_, limit_, page_);
        if (dma_missing(wanted, pinned_).empty()) {
            audit::count("dma-stage-covered-skip");
            return;
        }
        if (retain_window) retain(std::move(wanted));
        else prepare(std::move(wanted));
    }
    void clear() {
        drain();
        const auto t0 = trace_ ? Clock::now() : Clock::time_point{};
        while (!pinned_.empty()) {
            check(audit::unreg(__FILE__, __LINE__, base_ + pinned_.back().first, pinned_.back().second, pinned_.back().first), "unregistering the previous DMA window");
            unregistered_bytes_ += pinned_.back().second;
            if (trace_) cost_.removed += (int64_t) pinned_.back().second;
            pinned_.pop_back();
        }
        if (trace_) cost_.unregister_ms += ms_since(t0);
        ages_.clear();
    }
    void prepare(std::vector<DmaRange> sources) {
        audit::HostSpan audit_span(__FILE__, __LINE__, "dma-prepare-total");
        const auto wanted = dma_ranges(std::move(sources), arena_, limit_, page_);
        if (wanted == pinned_) { audit::count("dma-identical-skip"); return; }
        if (trace_) ++cost_.calls;
        clear();
        const auto t0 = trace_ ? Clock::now() : Clock::time_point{};
        try {
            for (const auto& r : wanted) {
                check(audit::reg(__FILE__, __LINE__, base_ + r.first, r.second, 0, r.first), "CUDA/HIP registration of " +
                      std::to_string(r.second) + " RAM bytes (no pageable fallback)");
                registered_bytes_ += r.second;
                if (trace_) cost_.added += (int64_t) r.second;
                try { pinned_.push_back(r); }
                catch (...) { (void) audit::unreg(__FILE__, __LINE__, base_ + r.first, r.second, r.first); throw; }
                verify(base_ + r.first);
#ifndef HIP_VERSION
                verify(base_ + r.first + r.second - 1);
#endif
            }
            peak_ = std::max(peak_, bytes());
        } catch (...) { clear(); throw; }
        if (trace_) cost_.register_ms += ms_since(t0);
    }
    // Exchange mode only: retain useful registrations, add uncovered pages, evict old whole
    // registrations only under budget pressure. All users finish before unregistering any page.
    void retain(std::vector<DmaRange> sources) {
        audit::HostSpan audit_span(__FILE__, __LINE__, "dma-retain-total");
        const auto wanted = dma_ranges(std::move(sources), arena_, limit_, page_);
        ++age_;
        const auto intersects = [&](const DmaRange& r) {
            for (const auto& w : wanted)
                if (r.first < w.first + w.second && w.first < r.first + r.second) return true;
            return false;
        };
        for (const auto& r : pinned_) if (intersects(r)) ages_[r.first] = age_;
        auto missing = dma_missing(wanted, pinned_);
        const auto missing_bytes = [&] { size_t n = 0; for (const auto& r : missing) n += r.second; return n; };
        if (missing.empty()) { audit::count("dma-covered-skip"); return; }
        if (trace_) ++cost_.calls;
        bool drained = false;
        auto t_evict = trace_ ? Clock::now() : Clock::time_point{};
        const double drain_before = cost_.drain_ms;
        while (bytes() + missing_bytes() > limit_) {
            if (!drained) { drain(); drained = true; }
            if (pinned_.empty()) throw std::logic_error("exchange: DMA window cannot hold requested pages");
            auto victim = std::min_element(pinned_.begin(), pinned_.end(), [&](const auto& a, const auto& b) {
                if (intersects(a) != intersects(b)) return !intersects(a);
                return ages_[a.first] < ages_[b.first];
            });
            check(audit::unreg(__FILE__, __LINE__, base_ + victim->first, victim->second, victim->first), "evicting an idle DMA registration");
            unregistered_bytes_ += victim->second;
            if (trace_) cost_.removed += (int64_t) victim->second;
            ages_.erase(victim->first);
            pinned_.erase(victim);
            missing = dma_missing(wanted, pinned_);
        }
        if (trace_) cost_.unregister_ms += ms_since(t_evict) - (cost_.drain_ms - drain_before);
        const auto t0 = trace_ ? Clock::now() : Clock::time_point{};
        for (const auto& r : missing) {
            check(audit::reg(__FILE__, __LINE__, base_ + r.first, r.second, 0, r.first), "registering exchange RAM pages");
            registered_bytes_ += r.second;
            if (trace_) cost_.added += (int64_t) r.second;
            try { ages_[r.first] = age_; pinned_.push_back(r); }
            catch (...) { ages_.erase(r.first); (void) audit::unreg(__FILE__, __LINE__, base_ + r.first, r.second, r.first); throw; }
            std::sort(pinned_.begin(), pinned_.end());
            peak_ = std::max(peak_, bytes());
            try { verify(base_ + r.first); }
            catch (...) { clear(); throw; }
        }
        if (trace_) cost_.register_ms += ms_since(t0);
    }
    void begin(cudaStream_t stream) {
        for (auto& u : uses_) if (u.stream == stream) { u.pending = true; return; }
        cudaEvent_t event = nullptr;
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "creating a source DMA fence");
        try { uses_.push_back({stream, event, true}); }
        catch (...) { (void) cudaEventDestroy(event); throw; }
    }
    void end(cudaStream_t stream) {
        for (auto& u : uses_) if (u.stream == stream) {
            check(cudaEventRecord(u.done, stream), "recording a source DMA fence");
            u.pending = false;
            return;
        }
        throw std::logic_error("no-disk: DMA stream was not begun");
    }
};
} // namespace strata::poly
