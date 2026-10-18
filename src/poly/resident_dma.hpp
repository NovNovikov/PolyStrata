// Register the original resident bytes for CUDA/HIP DMA, never a second host staging copy.
#pragma once
#include "dma_ranges.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
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
    std::vector<Use> uses_;
    static void check(cudaError_t e, const std::string& what) {
        if (e == cudaSuccess) return;
        (void) cudaGetLastError();
        throw std::runtime_error("no-disk: " + what + ": " + cudaGetErrorString(e));
    }
    void drain() {
        for (auto& u : uses_)
            check(u.pending ? cudaStreamSynchronize(u.stream) : cudaEventSynchronize(u.done), "waiting for source DMA");
    }
    void verify(void* p) {
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
            if (u.pending) (void) cudaStreamSynchronize(u.stream);
            else (void) cudaEventSynchronize(u.done);
        }
        for (const auto& r : pinned_) (void) cudaHostUnregister(base_ + r.first);
        for (const auto& u : uses_) (void) cudaEventDestroy(u.done);
        (void) cudaGetLastError();
    }
    const uint8_t* base() const { return base_; }
    size_t bytes() const { size_t n = 0; for (const auto& r : pinned_) n += r.second; return n; }
    size_t peak() const { return peak_; }
    void clear() {
        drain();
        while (!pinned_.empty()) {
            check(cudaHostUnregister(base_ + pinned_.back().first), "unregistering the previous DMA window");
            pinned_.pop_back();
        }
    }
    void prepare(std::vector<DmaRange> sources) {
        const auto wanted = dma_ranges(std::move(sources), arena_, limit_, page_);
        if (wanted == pinned_) return;
        clear();
        try {
            for (const auto& r : wanted) {
                check(cudaHostRegister(base_ + r.first, r.second, 0), "CUDA/HIP registration of " +
                      std::to_string(r.second) + " RAM bytes (no pageable fallback)");
                try { pinned_.push_back(r); }
                catch (...) { (void) cudaHostUnregister(base_ + r.first); throw; }
                verify(base_ + r.first);
#ifndef HIP_VERSION
                verify(base_ + r.first + r.second - 1);
#endif
            }
            peak_ = std::max(peak_, bytes());
        } catch (...) { clear(); throw; }
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
