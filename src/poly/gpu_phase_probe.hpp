// Compiled only into poly_gpu_phase_probe. Events are captured as graph nodes; the production engine has none.
#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace strata::poly::gpu_probe {
inline void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Probe {
    std::vector<std::array<cudaEvent_t, 7>> layers;
    bool pending = false;
    int active_layer = 0;
    long long position = 0;
    void mark(int layer, int point, cudaStream_t stream) {
        while ((int) layers.size() <= layer) {
            std::array<cudaEvent_t, 7> events{};
            for (auto& event : events) check(cudaEventCreate(&event));
            layers.push_back(events);
        }
        check(cudaEventRecordWithFlags(layers[layer][point], stream, cudaEventRecordExternal));
    }
    // Called after run()'s existing stream synchronization. No new host/GPU synchronization.
    void begin(int rows, long long pos) {
        if (pending) {
            std::array<double, 6> ms{};
            int complete = 0;
            for (const auto& events : layers) {
                if (cudaEventQuery(events[6]) != cudaSuccess) continue;
                for (int j = 0; j < 6; ++j) {
                    float one = 0; check(cudaEventElapsedTime(&one, events[j], events[j + 1])); ms[j] += one;
                }
                ++complete;
            }
            std::fprintf(stderr, "GPU_PHASE pos=%lld layers=%d hc_norm_ms=%.6f qkv_ms=%.6f "
                                 "index_ms=%.6f attend_ms=%.6f out_proj_ms=%.6f router_ms=%.6f\n",
                         position, complete, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5]);
        }
        pending = rows == 1;
        position = pos;
    }
    ~Probe() { for (auto& l : layers) for (auto e : l) cudaEventDestroy(e); }
};
inline Probe& state() { static thread_local Probe probe; return probe; }
inline void mark(int n, int layer, int point, void* stream) {
    if (n == 1) state().mark(layer, point, static_cast<cudaStream_t>(stream));
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
    state().begin(0, 1);
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    check(cudaFree(memory)); check(cudaStreamDestroy(stream));
    std::printf("GPU phase captured-event self-test passed: %.6f ms\n", elapsed);
    return 0;
}
} // namespace strata::poly::gpu_probe
