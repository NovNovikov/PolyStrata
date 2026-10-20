// Isolated GPU time for the DeepSeek V4 decode attention kernels, with the local model's geometry.
// Synthetic values; CUDA graphs remove host launch gaps. Repeated buffers are warm in the GPU cache.
#include "strata/poly/kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T* device(const std::vector<T>& data) {
    T* p = nullptr; check(cudaMalloc(&p, data.size() * sizeof(T)));
    check(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static double timed(cudaStream_t stream, const std::function<void()>& run) {
    run(); check(cudaStreamSynchronize(stream)); // attributes and lazy initialization before capture
    cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 64; ++i) run();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, 0));
    check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
    cudaEvent_t start, end;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    std::vector<float> times;
    for (int r = 0; r < 7; ++r) {
        check(cudaEventRecord(start, stream));
        check(cudaGraphLaunch(exec, stream));
        check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
        float ms = 0; check(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms * 1000 / 64);
    }
    std::sort(times.begin(), times.end());
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    return times[times.size() / 2];
}
int main() {
    try {
        constexpr int dim = 512, heads = 64, idx_dim = 128, top = 512, window = 128, ring = 4224;
        constexpr int stride = 131072 / 4;
        std::mt19937 rng(42);
        auto floats = [&](size_t n) {
            std::vector<float> v(n);
            for (auto& f : v) f = std::normal_distribution<float>(0.f, .1f)(rng);
            return v;
        };
        auto halves = [&](size_t n) {
            const auto f = floats(n); std::vector<uint16_t> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = __half_as_ushort(__float2half(f[i]));
            return v;
        };
        auto *q = device(floats(dim * heads)), *iq = device(floats(idx_dim * heads));
        auto *w = device(floats(heads)), *sink = device(floats(heads)), *out = device(floats(dim * heads));
        auto *raw = device(halves(ring * dim)), *comp = device(halves(stride * dim));
        auto *keys = device(halves(stride * idx_dim));
        auto *scores = device(floats(stride));
        std::vector<int32_t> selected(top); for (int i = 0; i < top; ++i) selected[i] = i;
        auto *sel = device(selected);
        auto *at = device(std::vector<long long>{0, 0});
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        std::puts("context,room,kernel,us_per_call,layer_count,ms_per_43_layers");
        for (int context : {1536, 22446, 90000, 131072}) {
            int room = 1024; while (room < context) room *= 2;
            const long long position[2] = {context - 1, 0};
            check(cudaMemcpy(at, position, sizeof(position), cudaMemcpyHostToDevice));
            for (int ratio : {0, 4, 128}) {
                const auto run = [&] { strata::poly::dsv4_attend(q, raw, ring, window, ratio ? comp : nullptr,
                    std::max(1, ratio), ratio == 4 ? sel : nullptr, top, top, sink, out, dim, heads,
                    1.f / std::sqrt(float(dim)), context - 1, 1, stream, at, room); };
                const double us = timed(stream, run);
                const int count = ratio == 4 ? 21 : ratio == 128 ? 20 : 2;
                std::printf("%d,%d,attend_ratio%d,%.4f,%d,%.4f\n", context, room, ratio, us, count, us * count / 1000);
            }
            const auto score = [&] { strata::poly::dsv4_idx_score(iq, w, keys, scores, stride, idx_dim, heads, 4,
                context - 1, 1, 1.f / std::sqrt(float(idx_dim * heads)), stream, at, room); };
            const auto select = [&] { strata::poly::dsv4_select(scores, stride, sel, top, top, 4, context - 1, 1, stream, at); };
            for (bool selection : {false, true}) {
                const double us = timed(stream, selection ? std::function<void()>(select) : std::function<void()>(score));
                std::printf("%d,%d,%s,%.4f,21,%.4f\n", context, room, selection ? "select" : "index_score", us, 21 * us / 1000);
            }
            std::fflush(stdout);
        }
        check(cudaStreamDestroy(stream));
        for (float* p : {q, iq, w, sink, out}) check(cudaFree(p));
        for (uint16_t* p : {raw, comp, keys}) check(cudaFree(p));
        check(cudaFree(scores)); check(cudaFree(sel)); check(cudaFree(at));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
