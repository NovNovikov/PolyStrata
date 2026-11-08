// Compiled only into poly_gpu_phase_probe. Events are captured as graph nodes; the production engine has none.
#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::poly::gpu_probe {
inline void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Probe {
    // A decode step is one token or a speculation window's few (window() runs 1 to opt_.window rows); a prompt
    // batch runs thousands and its phases are the ones prefill's own counters already tell.  Rows are printed so
    // a window's lines can be told from a single token's when the two are read side by side.
    static constexpr int kDecodeRows = 8;
    std::vector<std::array<cudaEvent_t, 7>> layers;
    bool pending = false;
    int pending_rows = 0;
    int active_layer = 0;
    long long position = 0;
    bool quiet = false;   // one message per failure kind: a probe that kills the run measures nothing
    void mark(int layer, int point, cudaStream_t stream) {
        while ((int) layers.size() <= layer) {
            std::array<cudaEvent_t, 7> events{};
            for (auto& event : events) if (cudaEventCreate(&event) != cudaSuccess) return note("create", layer, point);
            layers.push_back(events);
        }
        // cudaEventRecordExternal means "make an external event node" and is only legal while the stream is being
        // captured: outside a capture it fails with cudaErrorIllegalState.  The engine captures a layer's head and
        // replays it, but a MoE layer's experts and tail run eager, as does a draft block, so both kinds exist and
        // each takes the form its stream is in: a node inside a capture, a plain record outside one.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess) return note("capture query", layer, point);
        const cudaError_t e = cap == cudaStreamCaptureStatusNone
                                  ? cudaEventRecord(layers[layer][point], stream)
                                  : cudaEventRecordWithFlags(layers[layer][point], stream, cudaEventRecordExternal);
        if (e != cudaSuccess) note(cap == cudaStreamCaptureStatusNone ? "record" : "record in capture", layer, point);
    }
    void note(const char* what, int layer, int point) {
        if (quiet) return;
        quiet = true;
        std::fprintf(stderr, "GPU_PHASE_MARK_FAIL %s layer=%d point=%d: %s (further marks stay silent)\n", what,
                     layer, point, cudaGetErrorString(cudaGetLastError()));
    }
    // Called after run()'s existing stream synchronization. No new host/GPU synchronization.
    void begin(int rows, long long pos) {
        if (pending) {
            static const bool detail = std::getenv("STRATA_GPU_PHASE_DETAIL") != nullptr;
            std::array<double, 6> ms{};
            int complete = 0;
            for (size_t l = 0; l < layers.size(); ++l) {
                auto& events = layers[l];
                if (cudaEventQuery(events[6]) != cudaSuccess) continue;
                std::array<double, 6> one{};
                bool timed = true;
                for (int j = 0; j < 6; ++j) {
                    float el = 0;
                    if (cudaEventElapsedTime(&el, events[j], events[j + 1]) != cudaSuccess) { timed = false; break; }
                    one[j] += el;
                    ms[j] += el;
                }
                if (!timed) continue;   // an event pair one of the marks never reached: not this layer's turn
                if (detail)
                    std::fprintf(stderr, "GPU_PHASE_DETAIL pos=%lld rows=%d layer=%zu a=%.6f b=%.6f c=%.6f "
                                         "d=%.6f e=%.6f f=%.6f\n",
                                 position, pending_rows, l, one[0], one[1], one[2], one[3], one[4], one[5]);
                ++complete;
            }
            std::fprintf(stderr, "GPU_PHASE pos=%lld rows=%d layers=%d hc_norm_ms=%.6f qkv_ms=%.6f "
                                 "index_ms=%.6f attend_ms=%.6f out_proj_ms=%.6f router_ms=%.6f\n",
                         position, pending_rows, complete, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5]);
        }
        pending = rows <= kDecodeRows;
        pending_rows = rows;
        position = pos;
    }
    ~Probe() { for (auto& l : layers) for (auto e : l) cudaEventDestroy(e); }
};
inline Probe& state() { static thread_local Probe probe; return probe; }
inline void mark(int n, int layer, int point, void* stream) {
    if (n <= Probe::kDecodeRows) state().mark(layer, point, static_cast<cudaStream_t>(stream));
}
inline int self_test() {
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    void* memory = nullptr; check(cudaMalloc(&memory, 1u << 20));
    cudaGraph_t graph; cudaGraphExec_t exec;
    state().begin(1, 0);
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int j = 0; j < 7; ++j) {
        if (j) check(cudaMemsetAsync(memory, j, 1u << 20, stream));
        state().mark(0, j, stream);
    }
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
    float elapsed = 0; check(cudaEventElapsedTime(&elapsed, state().layers[0][0], state().layers[0][6]));
    if (!(elapsed > 0)) throw std::runtime_error("captured GPU phase events did not record time");
    state().begin(Probe::kDecodeRows + 1, 1);   // a batch's rows: nothing stays pending after the self-test
    // The eager pass: a MoE layer's experts and tail, and a draft block, mark a stream that is NOT being captured,
    // where cudaEventRecordExternal is illegal (cudaErrorIllegalState) and the mark takes cudaEventRecord.  A layer
    // that never got its two records cannot time, so a mark that failed quietly is caught here by an empty event.
    state().mark(1, 0, stream);
    for (int j = 0; j < 7; ++j) check(cudaMemsetAsync(memory, j, 1u << 20, stream));
    state().mark(1, 6, stream);
    check(cudaStreamSynchronize(stream));
    float eager = 0;
    check(cudaEventElapsedTime(&eager, state().layers[1][0], state().layers[1][6]));
    if (!(eager > 0)) throw std::runtime_error("non-captured GPU phase events did not record time");
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    check(cudaFree(memory)); check(cudaStreamDestroy(stream));
    std::printf("GPU phase captured-event self-test passed: %.6f ms\n", elapsed);
    std::printf("GPU phase eager-event self-test passed: %.6f ms\n", eager);
    return 0;
}
} // namespace strata::poly::gpu_probe
