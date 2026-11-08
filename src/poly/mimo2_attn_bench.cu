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
// usage: poly_mimo2_attn_bench [depth_cap] [decode]   (cap default 100000; the model's 131072 fits with a bigger one)
// not a test: run it by hand, e.g. from build-poly-cuda12. Times are per attend, milliseconds.
//
// `decode` answers a different question and needs a tenth of the card for it: an ANSWER's call carries one to a
// few rows (a token and its guessed drafts), which is under kMmaRows, so the tensor-core tile never answers one -
// the split kernel does, and the matrix above, which starts at 64 rows, never times it. The decode matrix prints
// microseconds and the bytes an attend has to move: `useful` counts a KV head's rows once, `through the card`
// counts them n_head/kv_heads times, which is what the kernel does when each query head's block walks them on
// its own. With a few rows the difference between the two is the difference between the card's bandwidth and the
// few warps that are actually in flight.
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
    bool ok = true;
};

Card card(int max_rows, long long max_depth) {
    Card c;
    // a card a server is using has less room than this file's default matrix asks for: say so and leave, rather
    // than launch a kernel on a null cache and read whatever the driver feels like answering with
    const auto give = [&](cudaError_t e, const char* what, size_t bytes) {
        if (e == cudaSuccess) return;
        std::printf("no room for %s (%.1f MiB): %s\n", what, bytes / 1048576.0, cudaGetErrorString(e));
        c.ok = false;
    };
    give(cudaMalloc(&c.q, size_t(max_rows) * kNHead * kDimK * sizeof(float)), "queries",
         size_t(max_rows) * kNHead * kDimK * sizeof(float));
    std::vector<float> h(size_t(max_rows) * kNHead * kDimK);
    for (float& x : h) x = frand() * 2.f;
    if (c.ok) give(cudaMemcpy(c.q, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "query copy", 0);
    const size_t krows = size_t(max_depth) * 4 * kDimK, vrows = size_t(max_depth) * 8 * kDimV;
    if (c.ok) give(cudaMalloc(&c.kc, krows * 2), "key cache", krows * 2);
    if (c.ok) give(cudaMalloc(&c.vc, vrows * 2), "value cache", vrows * 2);
    std::vector<uint16_t> hk(krows), hv(vrows);
    for (auto& x : hk) x = __half_as_ushort(__float2half_rn(frand()));
    for (auto& x : hv) x = __half_as_ushort(__float2half_rn(frand()));
    if (c.ok) give(cudaMemcpy(c.kc, hk.data(), krows * 2, cudaMemcpyHostToDevice), "key copy", krows * 2);
    if (c.ok) give(cudaMemcpy(c.vc, hv.data(), vrows * 2, cudaMemcpyHostToDevice), "value copy", vrows * 2);
    if (c.ok) give(cudaMalloc(&c.out[0], size_t(max_rows) * kNHead * kDimV * sizeof(float)), "output 0", 0);
    if (c.ok) give(cudaMalloc(&c.out[1], size_t(max_rows) * kNHead * kDimV * sizeof(float)), "output 1", 0);
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

/// One answer's attend, timed: the split kernel's own microseconds and the bytes it moves.
void cell_decode(const Card& c, const char* kind, int rows, long long depth, int kv_heads, int window, int ring,
                 int warps, int splits = 1, int batch = 1) {
    const long long pos = depth - rows;
    const int iters = depth > 32768ll ? 5 : 10;
    putenv_str((char*) "STRATA_MIMO_ATTN_NO_MMA=1");   // under kMmaRows the entry takes the split kernel either way
    putenv_str((char*) ("STRATA_MIMO_ATTN_WARPS=" + std::to_string(warps)).c_str());
    putenv_str((char*) ("STRATA_MIMO_ATTN_SPLITK=" + std::to_string(splits)).c_str());
    putenv_str((char*) ("STRATA_MIMO_ATTN_BATCH=" + std::to_string(batch)).c_str());
    const double ms = timed(c, rows, pos, kv_heads, window, ring, c.out[0], iters);
    const long long keys = window > 0 ? std::min<long long>(window, depth) : depth;
    const double useful = double(keys) * kv_heads * (kDimK + kDimV) * 2.0;
    const double moved = useful * (kNHead / kv_heads);
    std::printf("%-7s rows %2d depth %7lld warps %2d splits %d batch %d  %9.1f us   useful %6.1f GB/s"
                "   through the card %7.1f GB/s\n", kind, rows, depth, warps, splits, batch, ms * 1000.0,
                useful / (ms * 1e6), moved / (ms * 1e6));
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const long long cap = argc > 1 ? atoll(argv[1]) : 100000;
    const bool decode = argc > 2 && std::string(argv[2]) == "decode";
    const int max_rows = decode ? 8 : 4096;
    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    std::printf("card: %s (sm_%d%d), tile smem %zu bytes\n", prop.name, prop.major, prop.minor,
                size_t(2) * 64 * (192 + 8 + 128 + 8) * sizeof(unsigned short));
    size_t free_before = 0, total = 0, free_after = 0;
    cudaMemGetInfo(&free_before, &total);
    Card c = card(max_rows, cap + max_rows);
    cudaMemGetInfo(&free_after, &total);
    std::printf("this bench's own card memory: %.1f MiB (free before %.1f, after %.1f of %.1f MiB)\n",
                (free_before - free_after) / 1048576.0, free_before / 1048576.0, free_after / 1048576.0,
                total / 1048576.0);
    if (!c.ok) return 1;

    if (decode) {
        // a cell reads cache rows up to `depth`, so a cap below it would read past the allocation
        std::printf("\nan answer's attend, rows 2 - a token with its drafts - by the warps a block gets"
                    " (4 is today's launch):\n");
        for (const long long depth : {1000ll, 4096ll, 16384ll, 32768ll, 50000ll})
            if (depth <= cap)
                for (const int w : {4, 8, 16, 32})
                    cell_decode(c, "global", 2, depth, 4, 0, 0, w);
        std::printf("\nwarps by row count at the deepest depth at or under the cap (an answer's calls are rows 1"
                    " to 4; the rows over that are a short prompt's, which take this kernel too):\n");
        for (long long depth : {50000ll, 32768ll, 16384ll, 4096ll, 1000ll})
            if (depth <= cap) {
                for (const int rows : {1, 2, 3, 4, 8, 16})
                    for (const int w : {4, 8, 16, 32})
                        cell_decode(c, "global", rows, depth, 4, 0, 0, w);
                break;
            }
        std::printf("\nkeys split over block groups, at each rows' best warp count (1 = no split, the"
                    " launch today):\n");
        for (long long depth : {16384ll, 50000ll})
            if (depth <= cap)
                for (const int rows : {1, 2, 3, 4})
                    for (const int sk : {1, 2, 4, 8})
                        cell_decode(c, "global", rows, depth, 4, 0, 0, rows <= 2 ? 32 : 16, sk);
        std::printf("\nkeys a warp takes per round, at each rows' best warps and splits (1 = the loop of"
                    " every build before this; the wall being tested is one dependent memory round trip"
                    " per key):\n");
        for (long long depth : {16384ll, 50000ll})
            if (depth <= cap)
                for (const int rows : {1, 2, 3, 4}) {
                    const int sk = rows == 1 ? 2 : rows == 3 ? 8 : 1;
                    for (const int b : {1, 2, 4, 8}) {
                        const int w = b == 1 ? (rows <= 2 ? 32 : 16) : 16;   // batch caps warps - see the entry
                        cell_decode(c, "global", rows, depth, 4, 0, 0, w, sk, b);
                        if (b == 1 && rows <= 2)
                            cell_decode(c, "global", rows, depth, 4, 0, 0, 16, sk, 1);   // the same warps as batch
                    }
                }
        std::printf("\nthe 39 window layers, 128 rows of a ring:\n");
        if (kRingRows <= cap + max_rows)
            for (const long long depth : {16384ll, 50000ll})
                if (depth <= cap)
                    for (const int rows : {1, 4})
                        for (const int w : {4, 32})
                            cell_decode(c, "window", rows, depth, 8, kWindow, kRingRows, w);
        cudaFree(c.q); cudaFree(c.kc); cudaFree(c.vc); cudaFree(c.out[0]); cudaFree(c.out[1]);
        return 0;
    }

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
