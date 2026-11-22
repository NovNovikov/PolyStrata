// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/dsv41/ds4_rc_stage.hpp - one staging blob for the recent-slot fills' H2D copies, and the rule that keeps it
// honest: nobody writes it while a device copy is still reading it (hostile review, third round item 2).
//
// The recent-slot filler stages each expert's bytes once, then lets the device pull them with ASYNC copies, so the
// next expert's bytes must not overwrite the blob before those copies drained. record() notes the end of one fill on
// EVERY member stream its copies ran on (a tensor-parallel expert's pieces go to several devices); drain() waits for
// all of them before the buffer may be written again - and before it is freed, so teardown never frees a buffer a
// DMA still reads. Event operations are checked: a failed create/record/sync says so, and the caller declines the
// fill rather than staging blind - a declined recent-slot fill only means the expert routes over the RAM lane.
//
// The blob itself is pinned when CUDA can give pinned memory, plain heap when it cannot: an async copy FROM pageable
// memory completes before cudaMemcpyAsync returns (the driver stages it internally), so the no-overwrite rule holds
// trivially there, and a failed pinned allocation no longer sends the copies back to the store's recastable slots -
// the defect the staging was added for. No allocation at all: alloc() says false and the caller declines the fills.
//
// Tests: tests/dsv41/ds4_rc_stage_test.cpp (device): two experts, a deliberately slow copy for the first, a refill
// attempt for the second - both destinations keep their bytes, and a stage destroyed mid-DMA stays alive for it.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <new>
#include <vector>

namespace strata::ds4 {

class RcStage {
public:
    ~RcStage() {
        // (the DMA must be done before the bytes it reads go away, pinned or not - and if that completion can no
        // longer be proven (poisoned below), the blob is deliberately LEAKED rather than freed under a live copy)
        if (!poisoned_) (void) drain();
        if (!poisoned_ && p_ != nullptr) {
            if (pinned_) cudaFreeHost(p_);
            else delete[] p_;
        }
        p_ = nullptr;
    }
    bool alloc(size_t bytes) {   // (false: no pinned AND no heap room - the caller declines the staging fills)
        if (bytes == 0) return false;
        bytes_ = bytes;
        if (cudaHostAlloc((void**) &p_, bytes, cudaHostAllocPortable) == cudaSuccess) {
            pinned_ = true;
            return true;
        }
        (void) cudaGetLastError();   // (the failure this fallback exists for: not the caller's error to carry)
        try {
            p_ = new (std::nothrow) uint8_t[bytes];
        } catch (...) {
            p_ = nullptr;
        }
        return p_ != nullptr;
    }
    uint8_t* data() const { return p_; }
    size_t size() const { return bytes_; }
    // (after enqueuing this fill's copies on one member's stream): a note that the device may still be reading the
    // blob through this stream. Every member's stream gets one; false means the note could not be made, and the
    // fill cannot be proven done - the caller declines it (nothing was written blind).
    bool record(int dev_id, cudaStream_t cs) {
        if (poisoned_) return false;
        cudaEvent_t e = nullptr;
        if (cudaSetDevice(dev_id) != cudaSuccess || cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (cudaEventRecord(e, cs) != cudaSuccess) {
            (void) cudaGetLastError();
            (void) cudaEventDestroy(e);
            return false;
        }
        infl_.push_back({dev_id, e});
        return true;
    }
    bool has_notes() const { return !infl_.empty(); }
    // (before the blob may be written again, and before it is freed): wait every noted copy out. false: some event
    // operation failed - the fills' completion can no longer be vouched for, decline rather than overwrite.
    bool drain() {
        if (poisoned_) return false;
        bool ok = true;
        size_t kept = 0;   // (a note whose wait failed STAYS: the copy it stands for may still be reading the blob,
        for (size_t i = 0; i < infl_.size(); ++i) {   //   and the stage is poisoned exactly for that case below)
            const Note& n = infl_[i];
            if (cudaSetDevice(n.dev) == cudaSuccess && cudaEventSynchronize(n.e) == cudaSuccess)
                (void) cudaEventDestroy(n.e);
            else {
                (void) cudaGetLastError();
                ok = false;
                infl_[kept++] = n;
            }
        }
        infl_.resize(kept);
        if (!ok) poisoned_ = true;   // (the blob's history is unprovable: never written again, never freed blind)
        return ok;
    }

private:
    struct Note {
        int dev;
        cudaEvent_t e;
    };
    uint8_t* p_ = nullptr;
    size_t bytes_ = 0;
    bool pinned_ = false;
    bool poisoned_ = false;   // (an event operation failed: completions are no longer vouchable for this blob)
    std::vector<Note> infl_;
};

}  // namespace strata::ds4
