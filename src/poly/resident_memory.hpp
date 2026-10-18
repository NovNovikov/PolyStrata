// RAM whose complete residency is required, not a best-effort hint. No model mapping or CUDA host arena.
#pragma once
#include "strata/platform/memory.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace strata::poly {
class ResidentMemory {
    struct Region { void* p; size_t n; bool owned; };
    std::vector<Region> regions_;
    std::vector<std::pair<void*, uint64_t>> pending_;
    uint64_t bytes_ = 0;
#ifdef _WIN32
    SIZE_T old_min_ = 0, old_max_ = 0;
    DWORD old_flags_ = 0;
    bool restore_ws_ = false;
#endif
public:
    ResidentMemory() {
#ifdef _WIN32
        restore_ws_ = GetProcessWorkingSetSizeEx(GetCurrentProcess(), &old_min_, &old_max_, &old_flags_) != 0;
#endif
    }
    ResidentMemory(const ResidentMemory&) = delete;
    ResidentMemory& operator=(const ResidentMemory&) = delete;
    ~ResidentMemory() {
        for (const auto& r : regions_) {
            platform::unlock_resident(r.p, r.n);
            if (!r.owned) continue;
#ifdef _WIN32
            VirtualFree(r.p, 0, MEM_RELEASE);
#else
            munmap(r.p, r.n);
#endif
        }
#ifdef _WIN32
        if (restore_ws_) SetProcessWorkingSetSizeEx(GetCurrentProcess(), old_min_, old_max_, old_flags_);
#endif
    }
    void* allocate(size_t n, const char* what) {
        if (!n) return nullptr;
#ifdef _WIN32
        void* p = VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) p = nullptr;
#endif
        if (!p) throw std::runtime_error("no-disk: cannot allocate " + std::to_string(n) + " RAM bytes for " + what);
        // Record it before locking: a partial refusal must release even the pages already locked.
        try { regions_.push_back({p, n, true}); }
        catch (...) {
#ifdef _WIN32
            VirtualFree(p, 0, MEM_RELEASE);
#else
            munmap(p, n);
#endif
            throw;
        }
        lock({{p, n}}, what);
        return p;
    }
    void add_external(void* p, size_t n) {
        if (!p || !n) return;
        regions_.push_back({p, n, false});
        pending_.push_back({p, n});
    }
    void lock_external() { if (!pending_.empty()) { lock(pending_, "host weights"); pending_.clear(); } }
    uint64_t bytes() const { return bytes_; }
private:
    void lock(const std::vector<std::pair<void*, uint64_t>>& ranges, const char* what) {
        uint64_t n = 0;
        for (const auto& r : ranges) n += r.second;
        const auto result = platform::lock_resident_ranges(ranges);
        if (!result.ok || result.locked_bytes != n)
            throw std::runtime_error("no-disk: could not lock " + std::to_string(n) + " RAM bytes for " + what +
                                     "; locked " + std::to_string(result.locked_bytes) + ": " + result.note);
        bytes_ += n;
    }
};
} // namespace strata::poly
