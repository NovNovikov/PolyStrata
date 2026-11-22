// Strata. src/dsv41/ds4_defer_math.hpp - the arithmetic of the STRATA_DS4_DEFER late add, in one copy shared by
// the device kernels (ds4_kernels.cu: defer_add_kernel, defer_mix_kernel) and the host test
// (tests/dsv41/defer_math_test.cpp), which checks it against the CPU reference for what the on-time add would
// have become (hostile review 7): the CPU lane's part is rank one (the vector y with per-stream weights c), so
// deferring it only composes c through the mixings the streams went through in the meantime, and the composed
// late add must equal the on-time add mixed. No device, no model: the host compiles the same functions.
#pragma once

#ifdef __CUDACC__
#define DS4_DEFER_FN __host__ __device__ inline
DS4_DEFER_FN float ds4_defer_fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);   // the intrinsic: safe against -use_fast_math, the kernel's current op
#else
    return ::fmaf(a, b, c);      // the host pass of nvcc (and the host tests): the same IEEE RN
#endif
}
#else
#include <cmath>
#define DS4_DEFER_FN inline
#define ds4_defer_fma(a, b, c) ::fmaf((a), (b), (c))
#endif

namespace strata::ds4 {

/// The late add of one element e of one token's part y: h's hc streams, `stride` apart (dim on the device),
/// gain c_k * y with c_k composed through the mixings the streams went through:
///   comb null:                       c_k = post[k]                       (the part lands on the streams as is;
///                                     a lone comb2 changes nothing - same branch as the kernel's)
///   comb only (DEFER=2,4):           c_k = sum_j comb[j][k] * post[j]    (one hc_post mixing since the add)
///   comb and comb2 (DEFER=3):        c_k = sum_j (sum_i comb[i][j] * post[i]) * comb2[j][k]
///                                     (the add would have passed the next layer's attention and FFN mixing)
/// The kernel reads its own per-token slices (post + t*8, comb + t*64); the arithmetic is this loop.
DS4_DEFER_FN void defer_add_elem(float* h, int stride, float y, const float* post, const float* comb,
                                 const float* comb2, int hc) {
    for (int k = 0; k < hc; ++k) {
        float c;
        if (comb != nullptr && comb2 != nullptr) {   // (STRATA_DS4_DEFER=3: through the next layer's attention and FFN mixing)
            c = 0.0f;
            for (int j = 0; j < hc; ++j) {
                float m = 0.0f;
                for (int i = 0; i < hc; ++i) m = ds4_defer_fma(comb[i * hc + j], post[i], m);
                c = ds4_defer_fma(comb2[j * hc + k], m, c);
            }
        } else if (comb != nullptr) {
            c = 0.0f;
            for (int j = 0; j < hc; ++j) c = ds4_defer_fma(comb[j * hc + k], post[j], c);
        } else c = post[k];
        h[(size_t) k * stride] = ds4_defer_fma(c, y, h[(size_t) k * stride]);
    }
}

/// STRATA_DS4_DEFER=4's weight remap (defer_mix_kernel): c[k] <- sum_j comb[j][k] * c[j]. A part deferred
/// further carries its weights c instead of its values and takes this once per extra layer, so it lands any
/// number of layers late with exactly the coefficients the on-time add would have had. hc <= 8.
DS4_DEFER_FN void defer_mix_row(float* c, const float* comb, int hc) {
    float v[8];
    for (int j = 0; j < hc; ++j) v[j] = c[j];
    for (int k = 0; k < hc; ++k) {
        float a = 0.0f;
        for (int j = 0; j < hc; ++j) a = ds4_defer_fma(comb[j * hc + k], v[j], a);
        c[k] = a;
    }
}

}  // namespace strata::ds4
