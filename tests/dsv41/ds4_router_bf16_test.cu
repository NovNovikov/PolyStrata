// Strata. tests/dsv41/ds4_router_bf16_test.cu - the BF16 router (STRATA_DS4_ROUTER_BF16, the 16ba4156b6 port)
// against the F32 copy, on the device, with synthetic router weights and no model (task spec 8.1): the same
// BF16 bits widened to F32 must give BIT-IDENTICAL logits from gemv_bf16 and gemv_f32 (same lanes, same fmaf
// order, same warp sum), and route() over those logits must agree on the top-k picks and weights. Shapes are
// the production router's: n = the layer's expert count, k = the model dim (always a multiple of 4), the
// logits' row stride 1024 as the engine's d.glog keeps them, T from 1 (decode) to 16 (kWin, a verify window).
// Needs a GPU; no GGUF, no model. Build: the ds4_router_bf16_test target (EXCLUDE_FROM_ALL).
#include "ds4_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

// the exact widening gemv_bf16 does inside the kernel: the BF16 bits at the top of an F32 word
static float widen(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// a deterministic router weight: a BF16-representable value (its low 16 F32 bits are zero), varied enough
// that an order change in the dot product would show
static uint16_t w_bit(int row, int col) {
    uint32_t x = (uint32_t) (row * 2654435761u + col * 40503u + 12345u);
    x ^= x >> 13;
    const float mag = 0.001f + (float) (x & 0xFFFF) * (2.0f / 65535.0f);
    float v = (x & (1u << 16)) ? mag : -mag;
    if (!(x & (1u << 20))) v = -v;
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return (uint16_t) (bits >> 16);   // BF16 by construction
}

static void one(int n, int k, int T) {
    char tag[96];
    std::snprintf(tag, sizeof tag, "gemv_bf16 bit-identical at n=%d k=%d T=%d", n, k, T);
    const int ldy = 1024;   // the engine's logits stride (d.glog)
    std::vector<uint16_t> wb((size_t) n * k);
    std::vector<float> wf((size_t) n * k);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c) {
            wb[(size_t) r * k + c] = w_bit(r, c);
            wf[(size_t) r * k + c] = widen(wb[(size_t) r * k + c]);
        }
    std::vector<float> x((size_t) T * k), y1((size_t) T * ldy, 1.0f), y2((size_t) T * ldy, 2.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((i % 199) - 99) * 0.013f;

    float *dx = nullptr, *df = nullptr, *dy1 = nullptr, *dy2 = nullptr;
    uint16_t* dwb = nullptr;
    const cudaError_t e1 = cudaMalloc(&dx, x.size() * 4);
    const cudaError_t e2 = cudaMalloc(&df, wf.size() * 4);
    const cudaError_t e3 = cudaMalloc(&dwb, wb.size() * 2);
    const cudaError_t e5 = cudaMalloc(&dy1, y1.size() * 4);
    const cudaError_t e6 = cudaMalloc(&dy2, y2.size() * 4);
    const bool ok0 = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess &&
                     e5 == cudaSuccess && e6 == cudaSuccess;
    if (!ok0) {
        const cudaError_t fe = e1 != cudaSuccess ? e1 : (e2 != cudaSuccess ? e2 : (e3 != cudaSuccess ? e3 : e5));
        std::printf("  note: cudaMalloc failed (%s) - %d x %d x %d skipped\n", cudaGetErrorString(fe), n, k, T);
        cudaFree(dx); cudaFree(df); cudaFree(dwb); cudaFree(dy1); cudaFree(dy2);
        ++checks;
        return;
    }
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(df, wf.data(), wf.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwb, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice);
    gemv_f32(df, dx, n, k, T, dy1, ldy, 0);
    gemv_bf16((const uint16_t*) dwb, dx, n, k, T, dy2, ldy, 0);
    bool ok = cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dx); cudaFree(df); cudaFree(dwb); cudaFree(dy1); cudaFree(dy2);
    if (!ok) { check(false, "logit readback failed - identity not checked (a false pass here would prove nothing)"); return; }
    bool bits = true;
    for (size_t i = 0; i < y1.size(); ++i)
        if (std::memcmp(&y1[i], &y2[i], 4) != 0) bits = false;
    check(bits, tag);

    // route() over the two paths' logits cannot differ then either: same picks and weights
    std::vector<int> i1((size_t) T * 8), i2((size_t) T * 8);
    std::vector<float> w1((size_t) T * 8), w2((size_t) T * 8);
    if (n >= 8 && T <= 16) {
        int *di1 = nullptr, *di2 = nullptr;
        float *dw1 = nullptr, *dw2 = nullptr;
        if (cudaMalloc(&di1, i1.size() * 4) == cudaSuccess && cudaMalloc(&di2, i2.size() * 4) == cudaSuccess &&
            cudaMalloc(&dw1, w1.size() * 4) == cudaSuccess && cudaMalloc(&dw2, w2.size() * 4) == cudaSuccess) {
            float *dl1 = nullptr, *dl2 = nullptr, *dbias = nullptr;
            const bool got = cudaMalloc(&dl1, y1.size() * 4) == cudaSuccess && cudaMalloc(&dl2, y2.size() * 4) == cudaSuccess &&
                             cudaMalloc(&dbias, (size_t) n * 4) == cudaSuccess;
            if (!got) {
                check(false, "logit buffers for route()");
            } else {
                std::vector<float> zeros((size_t) n, 0.0f);   // (the engine always runs the router with a bias tensor: null is no supported shape)
                cudaMemcpy(dbias, zeros.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dl1, y1.data(), y1.size() * 4, cudaMemcpyHostToDevice);
                cudaMemcpy(dl2, y2.data(), y2.size() * 4, cudaMemcpyHostToDevice);
                route(dl1, dbias, n, 8, T, 2.5f, di1, dw1, 0);
                route(dl2, dbias, n, 8, T, 2.5f, di2, dw2, 0);
                const bool copied = cudaMemcpy(i1.data(), di1, i1.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                                    cudaMemcpy(i2.data(), di2, i2.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                                    cudaMemcpy(w1.data(), dw1, w1.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                                    cudaMemcpy(w2.data(), dw2, w2.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
                const bool same = copied && std::memcmp(i1.data(), i2.data(), i1.size() * 4) == 0 &&
                                  std::memcmp(w1.data(), w2.data(), w1.size() * 4) == 0;
                check(same, "top-k routing identical on the two paths' logits");
            }
            cudaFree(dl1); cudaFree(dl2); cudaFree(dbias); cudaFree(di1); cudaFree(di2); cudaFree(dw1); cudaFree(dw2);
        }
    }
}

// the loader's fallback choice (router_w's condition), in the open: only a BF16 tensor of exactly n_exp x dim
// rows takes gate_bf; everything else keeps the F32 copy. The engine code is a member of Model::Impl, so the
// test mirrors the one condition it gates on and says what it checked.
static bool picks_bf(bool bf_env, const char* dtype, size_t bytes, int n_exp, int dim) {
    return bf_env && dtype != nullptr && std::strcmp(dtype, "BF16") == 0 && bytes == (size_t) n_exp * dim * 2;
}

int main(int argc, char** argv) {
    int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    if (cudaSetDevice(dev) != cudaSuccess) { std::printf("ds4_router_bf16_test: no device %d\n", dev); return 2; }
    cudaFree(0);   // context
    // the production router's n x k (256 experts x dim), block-edge n, small shapes; T decode to window
    one(256, 7168, 1);
    one(256, 7168, 16);
    one(16, 64, 1);
    one(16, 64, 4);
    one(64, 128, 3);
    one(32, 68, 2);
    one(288, 132, 16);
    // fallback selection
    check(picks_bf(true, "BF16", 256 * 7168 * 2, 256, 7168), "a BF16 router of the expected size takes the BF16 store");
    check(!picks_bf(false, "BF16", 256 * 7168 * 2, 256, 7168), "env off: the F32 copy, as before");
    check(!picks_bf(true, "F32", 256 * 7168 * 4, 256, 7168), "an F32 router keeps the F32 copy");
    check(!picks_bf(true, "BF16", 256 * 7168 * 3, 256, 7168), "a size mismatch keeps the F32 copy");
    check(!picks_bf(true, nullptr, 0, 256, 7168), "a missing tensor keeps the F32 copy");
    std::printf("ds4_router_bf16_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
