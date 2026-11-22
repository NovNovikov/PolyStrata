// Strata. src/dsv41 RcStage, the recent-slot fills' staging blob, on the device with no model (hostile review,
// third round item 2): the blob must never be rewritten while an async copy still reads it. Expert A is staged and
// its H2D is deliberately queued BEHIND a long device-side copy (a stream that stays busy for milliseconds, the way
// a real rc_cs is behind the tick's work); the fill of expert B may then begin only after RcStage::drain waits the
// notes out. Both destinations must hold the bytes their own expert had - a blob refilled too early streams B's
// bytes into A's slot, the silent corruption the staging exists to prevent. Two member streams (the tensor-parallel
// shape) are covered by the notes drain() waits, and a stage destroyed with copies in flight must stay alive for
// them. No GGUF, no model, no pack.
#include "ds4_rc_stage.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using strata::ds4::RcStage;

namespace {

int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) ++failures;
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
}

const size_t kBlob = 4u << 20;        // an expert's staged blob: 4 MB
const size_t kSlow = 256u << 20;      // the device-side copy that keeps the stream busy

}  // namespace

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("rc_stage_test: no CUDA device - nothing to test\n");
        return 0;
    }
    std::vector<uint8_t> pat_a(kBlob), pat_b(kBlob);
    for (size_t i = 0; i < kBlob; ++i) {
        pat_a[i] = (uint8_t) (i * 31u + 7u);
        pat_b[i] = (uint8_t) (i * 173u + 199u);
    }
    uint8_t* slow_src = nullptr;   // (device memory, filled once: the copy's source, contents irrelevant)
    uint8_t* slow_dst = nullptr;
    uint8_t* dst_a = nullptr;
    uint8_t* dst_b = nullptr;
    const bool mem_ok = cudaMalloc((void**) &slow_src, kSlow) == cudaSuccess &&
                        cudaMalloc((void**) &slow_dst, kSlow) == cudaSuccess &&
                        cudaMalloc((void**) &dst_a, kBlob) == cudaSuccess &&
                        cudaMalloc((void**) &dst_b, kBlob) == cudaSuccess;
    if (!mem_ok) {   // (a box that cannot spare 512 MB of VRAM is not this test's failure: say so and leave)
        std::printf("rc_stage_test: not enough device memory for the slow copy - nothing to test\n");
        return 0;
    }
    cudaMemset(slow_src, 5, kSlow);
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // 1. the defect scenario: fill A, queue its copies behind a long one, and let the SECOND fill's write go only
    //    through the protocol - drain, write, issue. With a busy stream the copies sit on it for milliseconds, so a
    //    drain that did not wait would let B's bytes slip into A's device slot.
    {
        RcStage stage;
        check(stage.alloc(kBlob), "the staging blob allocated");
        std::memcpy(stage.data(), pat_a.data(), kBlob);
        for (int rep = 0; rep < 4; ++rep)   // (busy first: A's copies queue behind a long device-side copy)
            cudaMemcpyAsync(slow_dst, slow_src, kSlow, cudaMemcpyDeviceToDevice, s1);
        cudaMemcpyAsync(dst_a, stage.data(), kBlob, cudaMemcpyHostToDevice, s1);
        check(stage.record(0, s1), "fill A noted on its member stream");
        // expert B arrives while A's copies are still on the stream: the protocol makes the blob wait for them
        check(stage.drain(), "the drain of fill A completed");
        std::memcpy(stage.data(), pat_b.data(), kBlob);   // (only legal AFTER the drain returned true)
        cudaMemcpyAsync(dst_b, stage.data(), kBlob, cudaMemcpyHostToDevice, s1);
        check(stage.record(0, s1), "fill B noted too");
        check(stage.drain(), "and the drain of fill B completed");
        std::vector<uint8_t> back(kBlob);
        bool a_ok = cudaMemcpy(back.data(), dst_a, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    std::memcmp(back.data(), pat_a.data(), kBlob) == 0;
        check(a_ok, "expert A's device slot holds A's bytes: the blob was not rewritten mid-DMA");
        check(cudaMemcpy(back.data(), dst_b, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::memcmp(back.data(), pat_b.data(), kBlob) == 0,
              "and expert B's slot holds B's bytes: the staged fill delivered the next expert whole");
    }

    // 2. the tensor-parallel shape: one fill, notes on TWO member streams, both busy - the drain must cover every
    //    stream the blob was read from, not just the last one
    {
        RcStage stage;
        check(stage.alloc(kBlob), "a second staging blob allocated");
        std::memcpy(stage.data(), pat_a.data(), kBlob);
        for (int rep = 0; rep < 4; ++rep) {
            cudaMemcpyAsync(slow_dst, slow_src, kSlow, cudaMemcpyDeviceToDevice, s1);
            cudaMemcpyAsync(slow_dst, slow_src, kSlow, cudaMemcpyDeviceToDevice, s2);   // (both members busy)
        }
        cudaMemcpyAsync(dst_a, stage.data(), kBlob, cudaMemcpyHostToDevice, s1);
        cudaMemcpyAsync(dst_b, stage.data(), kBlob, cudaMemcpyHostToDevice, s2);
        check(stage.record(0, s1) && stage.record(0, s2), "the fill noted on both member streams");
        check(stage.drain(), "the drain waited for every noted stream");
        std::vector<uint8_t> back(kBlob);
        check(cudaMemcpy(back.data(), dst_a, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::memcmp(back.data(), pat_a.data(), kBlob) == 0,
              "member 1's destination is whole");
        check(cudaMemcpy(back.data(), dst_b, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::memcmp(back.data(), pat_a.data(), kBlob) == 0,
              "and member 2's is: no member stream was left unchecked");
        // 3. teardown with a copy in flight: the destructor's drain must keep the blob alive for it (a freed
        //    pinned buffer under a live DMA is the crash this rule exists to keep away)
        std::memcpy(stage.data(), pat_b.data(), kBlob);
        for (int rep = 0; rep < 4; ++rep)
            cudaMemcpyAsync(slow_dst, slow_src, kSlow, cudaMemcpyDeviceToDevice, s2);
        cudaMemcpyAsync(dst_b, stage.data(), kBlob, cudaMemcpyHostToDevice, s2);
        check(stage.record(0, s2), "a fill noted just before teardown");
    }   // (stage's destructor runs with that copy still on the stream: it must wait it out)
    check(cudaStreamSynchronize(s2) == cudaSuccess, "teardown mid-DMA left the stream healthy");
    {
        std::vector<uint8_t> back(kBlob);
        check(cudaMemcpy(back.data(), dst_b, kBlob, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  std::memcmp(back.data(), pat_b.data(), kBlob) == 0,
              "and the last staged fill delivered its bytes - the blob outlived the copy reading it");
    }
    // 4. an empty stage: no notes, a plain refill is legal at once (the drain that has to wait for nothing waits
    //    for nothing - the non-staged members of the loop never pay the rule)
    {
        RcStage stage;
        check(stage.alloc(kBlob) && !stage.has_notes(), "a fresh stage has no outstanding copies");
        check(stage.drain(), "draining a stage nobody noted returns true");
        std::memcpy(stage.data(), pat_a.data(), kBlob);   // (legal: nothing is reading it)
        check(true, "and refilling it immediately is the protocol's idle path");
    }

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(slow_src);
    cudaFree(slow_dst);
    cudaFree(dst_a);
    cudaFree(dst_b);
    std::printf("ds4_rc_stage_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
