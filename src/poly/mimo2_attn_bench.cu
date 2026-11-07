// src/poly/mimo2_attn_bench.cu - MiMo attention kernels, timed alone: no model, no engine, no MoE.
//
// Each row of the matrix is one attend: MiMo's own geometry (64 query heads, keys 192, values 128), the
// model's own caches (fp16, the engine's layout), a random q and caches. The same attend runs twice: once
// with STRATA_MIMO_ATTN_NO_MMA=1 (the reference's split kernel) and once without (the tensor-core tile the
// entry picks from 64 rows), CUDA-event timed, best of the passes, and the two outputs compared - the
// kernel-vs-kernel cross-check at depths where the host reference of poly_mimo2_attn_test would take
// minutes per row. A card shared with a running server adds noise to every pass, which is why the bench
// keeps the best.
//
// usage: poly_mimo2_attn_bench [depth_cap]   (default 100000; the model's 131072 fits with a bigger one)
// not a test: run it by hand, e.g. from build-poly-cuda12. Times are per attend, milliseconds.
#include "mimo2_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define putenv_str(s) _putenv(s)
#else
#include <stdlib.h>
#define putenv_str(s) ::putenv(s)
#endif

using namespace strata::poly;

namespace {

constexpr int kDimK = 192, kDimV = 128, kNHead = 64;
constexpr int kWindow = 128, kRingRows = kWindow + 4096;   // the engine's window ring at a rows_ of 4096

unsigned rng_state = 20261107u;
float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (int(rng_state >> 8) % 20000) / 10000.f - 1.f;
}

struct Card {
    float* q = nullptr;           // the biggest batch's queries
    uint16_t* kc = nullptr;
    uint16_t* vc = nullptr;
    float* out[2] = {nullptr, nullptr};
};

Card card(int max_rows, long long max_depth) {
    Card c;
    cudaMalloc(&c.q, size_t(max_rows) * kNHead * kDimK * sizeof(float));
    std::vector<float> h(size_t(max_rows) * kNHead * kDimK);
    for (float& x : h) x = frand() * 2.f;
    cudaMemcpy(c.q, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    const size_t krows = size_t(max_depth) * 4 * kDimK, vrows = size_t(max_depth) * 8 * kDimV;
    cudaMalloc(&c.kc, krows * 2);
    cudaMalloc(&c.vc, vrows * 2);
    std::vector<uint16_t> hk(krows), hv(vrows);
    for (auto& x : hk) x = __half_as_ushort(__float2half_rn(frand()));
    for (auto& x : hv) x = __half_as_ushort(__float2half_rn(frand()));
    cudaMemcpy(c.kc, hk.data(), krows * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c.vc, hv.data(), vrows * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&c.out[0], size_t(max_rows) * kNHead * kDimV * sizeof(float));
    cudaMalloc(&c.out[1], size_t(max_rows) * kNHead * kDimV * sizeof(float));
    return c;
}

double timed(const Card& c, int rows, long long pos, int kv_heads, int window, int ring, float* out, int iters) {
    const float scale = 1.f / std::sqrt((float) kDimK);
    double best = 1e30;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        mimo2_attend(c.q, c.kc, c.vc, pos, rows, out, kDimK, kDimV, kNHead, kv_heads, scale, nullptr,
                     window, ring, nullptr, 0, window > 0 ? nullptr : nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

void cell(const Card& c, const char* kind, int rows, long long depth, int kv_heads, int window, int ring) {
    const long long pos = depth - rows;                     // the call's last token stands at depth - 1
    const int iters = rows * depth > 8ll << 20 ? 3 : 10;
    putenv_str((char*) "STRATA_MIMO_ATTN_NO_MMA=1");
    const double t_split = timed(c, rows, pos, kv_heads, window, ring, c.out[0], iters);
    putenv_str((char*) "STRATA_MIMO_ATTN_NO_MMA=0");
    const double t_mma = timed(c, rows, pos, kv_heads, window, ring, c.out[1], iters);
    const size_t n = size_t(rows) * kNHead * kDimV;
    std::vector<float> a(n), b(n);
    cudaMemcpy(a.data(), c.out[0], n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), c.out[1], n * sizeof(float), cudaMemcpyDeviceToHost);
    double worst = 0.;
    for (size_t i = 0; i < n; ++i)
        worst = std::max(worst, double(std::fabs(a[i] - b[i])));
    std::printf("%-7s rows %4lld depth %7lld  split %9.2f ms   mma %8.2f ms   x %6.1f   maxdiff %.3e\n",
                kind, (long long) rows, depth, t_split, t_mma, t_split / (t_mma > 0 ? t_mma : 1e-9), worst);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const long long cap = argc > 1 ? atoll(argv[1]) : 100000;
    const int max_rows = 4096;
    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    std::printf("card: %s (sm_%d%d), tile smem %zu bytes\n", prop.name, prop.major, prop.minor,
                size_t(2) * 64 * (192 + 8 + 128 + 8) * sizeof(unsigned short));
    Card c = card(max_rows, cap + max_rows);

    std::printf("\nglobal 64/4 (a window layer's nine partners read the whole history):\n");
    for (const long long depth : {4096ll, 16384ll, 32768ll, 50000ll})
        for (const int rows : {64, 128, 512, 4096})
            cell(c, "global", rows, depth, 4, 0, 0);
    if (cap >= 100000) {
        cell(c, "global", 64, 100000, 4, 0, 0);
        cell(c, "global", 512, 100000, 4, 0, 0);
        cell(c, "global", 4096, 100000, 4, 0, 0);
    }

    std::printf("\nwindow 64/8 window %d ring %d (39 of the 48 layers):\n", kWindow, kRingRows);
    for (const long long depth : {32768ll, 50000ll})
        for (const int rows : {512, 4096})
            cell(c, "window", rows, depth, 8, kWindow, kRingRows);

    cudaFree(c.q); cudaFree(c.kc); cudaFree(c.vc); cudaFree(c.out[0]); cudaFree(c.out[1]);
    return 0;
}
