// src/poly/mimo2_attn.cu - see the header.  The reference for every formula is llama.cpp's mimo2 graph
// (src/models/mimo2.cpp): NeoX rotary on the first n_rot values, scores q.k / sqrt(dim_k), a key hidden when
// p - key_pos >= window, and on a window layer one sink logit a query head in the softmax's sum.
//
// Two attend kernels behind one entry, split by how many rows the call carries:
// - a call of kMmaRows rows or more (a prompt batch) takes mimo2_attend_mma_kernel: the tensor cores, the shape
//   of kernels.cu's attn_flash128_kernel drawn for MiMo's head - 64 query rows a block (8 tokens x 8 heads),
//   keys 64 at a time through shared memory, S = Q K^T and O += P V on mma.m16n8k16 with the sums in floats,
//   each row's softmax kept online so no score ever leaves the registers.  A block's eight heads share their
//   kv head (which the model's ratios 16 to 1 and 8 to 1 give), so one fetch of a key row feeds eight heads'
//   scores where the reference walked the rows eight times at two bytes a value.  The walk starts at a
//   multiple of 64 keys, so a token's keys fall in the same blocks whatever tokens the call has beside it,
//   and the causal and window masks, the ring's rows and the sink's weight join per row as the reference has
//   them.  The queries and the weights round to halves and the products sum floats, as Flint's does.
// - fewer rows (a decoded token, a guess block) take mimo2_attend_split_kernel unchanged: with few queries the
//   card is kept busy splitting one head's keys across warps, not by walking per (token, head), and the
//   recorded decode launches keep the grid they were captured with.
// The launch picks the tensor cores only where they are (Ampere or later) and the head ratios divide the
// blocks; everywhere else the split kernel answers.  STRATA_MIMO_ATTN_TRACE=1 prints every call's chosen
// kernel and CUDA time to stderr - off by default, and the time it measures is the attention's own.
#include "mimo2_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int kMaxRot = 256;      // a rope block keeps a head's turned part in shared
constexpr int kMaxDims = 256;     // dim_k values a rope block keeps in shared
constexpr int kMmaRows = 64;      // rows from which the tensor cores' tile has enough blocks to win

// The tensor cores' m16n8k16 step and ldmatrix need Turing; AMD builds have neither and never launch these
// kernels (the entry checks the card).  Turing runs the step as two of k = 8 on the same registers.
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define STRATA_MIMO_MMA 0
#else
#define STRATA_MIMO_MMA 1
#endif

#if STRATA_MIMO_MMA
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 16 bytes from global to shared memory, or zeros where `whole` is false; asynchronous from Ampere on.
__device__ __forceinline__ void flash_copy(half* dst, const half* src, bool whole) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
#else
    *reinterpret_cast<uint4*>(dst) = whole ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void flash_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most `ahead` of the thread's groups are still on their way.
template <int ahead> __device__ __forceinline__ void flash_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
#else
    (void) ahead;
#endif
}
#endif  // STRATA_MIMO_MMA

// The fused row of one token: q (n_head heads of dim_k), k (kv_heads heads of dim_k), v (kv_heads heads of dim_v).
// A block turns one head: its values i < n_rot pair with i - n_rot / 2 and i + n_rot / 2 (ggml's NeoX rule,
// the same formula head_norm_rope_kernel uses with the norm's scale at 1), the rest leave as they came.
// No norm: MiMo norms its streams before the projection, not a head at a time.
__global__ void mimo2_rope_split_kernel(const float* __restrict__ src, int row, float* __restrict__ q,
                                        float* __restrict__ k, float* __restrict__ v, int dim_k, int dim_v,
                                        int n_head, int kv_heads, int n_rot, float theta_scale,
                                        long long pos, const long long* __restrict__ at) {
    const int tok = blockIdx.y;
    const int h = blockIdx.x;
    const float* x = src + size_t(tok) * row;
    const long long own = (at ? *at : pos) + tok;
    const int half = n_rot / 2;
    if (h < n_head) {                                   // a query head: turn, into q
        x += size_t(h) * dim_k;
        float* y = q + (size_t(tok) * n_head + h) * dim_k;
        for (int i = threadIdx.x; i < dim_k; i += blockDim.x) {
            if (i >= n_rot) { y[i] = x[i]; continue; }
            const int j = i < half ? i : i - half;
            const float turn = float(own) * powf(theta_scale, float(j));
            const float x0 = x[j], x1 = x[j + half];
            y[i] = i < half ? x0 * cosf(turn) - x1 * sinf(turn) : x0 * sinf(turn) + x1 * cosf(turn);
        }
        return;
    }
    const int kvh = h - n_head;
    const int n_kv = kv_heads;
    if (kvh < n_kv) {                                   // a key head: turn, into k
        x += size_t(n_head) * dim_k + size_t(kvh) * dim_k;
        float* y = k + (size_t(tok) * n_kv + kvh) * dim_k;
        for (int i = threadIdx.x; i < dim_k; i += blockDim.x) {
            if (i >= n_rot) { y[i] = x[i]; continue; }
            const int j = i < half ? i : i - half;
            const float turn = float(own) * powf(theta_scale, float(j));
            const float x0 = x[j], x1 = x[j + half];
            y[i] = i < half ? x0 * cosf(turn) - x1 * sinf(turn) : x0 * sinf(turn) + x1 * cosf(turn);
        }
        return;
    }
    if (kvh - n_kv < n_kv) {                            // a value head: as it came, into v
        x += size_t(n_head + n_kv) * dim_k + size_t(kvh - n_kv) * dim_v;
        float* y = v + (size_t(tok) * n_kv + (kvh - n_kv)) * dim_v;
        for (int i = threadIdx.x; i < dim_v; i += blockDim.x) y[i] = x[i];
    }
}

// One block a (token, query head): the warp's keys from the token's window, an online softmax a warp, the four
// warps' sums merged through shared.  A key's dim_k values are a lane's dim_k / 32 of them, its dim_v values a
// lane's dim_v / 32; the head's sink, when the layer has them, joins the merged sum as one more logit.
// The reference's shape, kept for the calls with few rows: see the file's header for why.
__global__ void mimo2_attend_split_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                                          const uint16_t* __restrict__ vcache, long long pos, int rows,
                                          float* __restrict__ out, int dim_k, int dim_v, int n_head, int kv_heads,
                                          float scale, int window, int ring, const long long* __restrict__ at,
                                          const float* __restrict__ sinks) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= rows) return;
    const long long p = (at ? *at : pos) + t;           // this token's position
    const int per = n_head / kv_heads;
    const int kvh = h / per;
    const long long lo = window > 0 && p >= window ? p - window + 1 : 0;
    const int n_keys = int(p - lo) + 1;

    const float* qh = q + (size_t(t) * n_head + h) * dim_k;
    const uint16_t* kc = kcache + size_t(kvh) * dim_k;  // + row * kv_heads * dim_k
    const uint16_t* vc = vcache + size_t(kvh) * dim_v;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = int(threadIdx.x >> 5);
    const int warps = int(blockDim.x >> 5);

    // the head's sink: one more logit, weighing no value
    float m = -INFINITY, s = 0.f;
    float acc[8];
    const int vpr = dim_v / WARP;                       // a lane's values of a key, 8 at most
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;

    __shared__ float qs[kMaxDims];                      // the query's values, read into shared once
    for (int i = int(threadIdx.x); i < dim_k; i += int(blockDim.x)) qs[i] = qh[i];
    __syncthreads();

    for (int key = warp; key < n_keys; key += warps) {
        const long long row = lo + key;
        const long long cr = ring ? row % ring : row;   // a ring layer's row p lies at cache row p % ring
        const uint16_t* krow = kc + size_t(cr) * kv_heads * dim_k;
        float dot = 0.f;
        for (int i = lane; i < dim_k; i += WARP) dot += qs[i] * __half2float(__ushort_as_half(krow[i]));
#pragma unroll
        for (int d = WARP / 2; d >= 1; d >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, d);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float nm = score > m ? score : m;
        const float shrink = expf(m - nm), add = expf(score - nm);
        s = s * shrink + add;
        const uint16_t* vrow = vc + size_t(cr) * kv_heads * dim_v;
        for (int j = 0; j < vpr; ++j) {
            acc[j] = acc[j] * shrink + add * __half2float(__ushort_as_half(vrow[WARP * j + lane]));
        }
        m = nm;
    }

    // merge the warps: the running max, the sums, and each lane's share of the value's sum
    extern __shared__ float merge[];                    // warps * (2 + dim_v)
    float* wm = merge;                                  // a warp's max
    float* ws = merge + WARP;                           // and its sum
    float* wacc = merge + 2 * WARP;                     // and its value, dim_v floats
    if (lane == 0) { wm[warp] = m; ws[warp] = s; }
    for (int j = 0; j < vpr; ++j) wacc[size_t(warp) * dim_v + WARP * j + lane] = acc[j];
    __syncthreads();

    float all_m = sinks ? sinks[h] : -INFINITY;         // the sink joins the max: exp(sink - max) cannot overflow
    for (int w = 0; w < warps; ++w) all_m = fmaxf(all_m, wm[w]);
    if (all_m == -INFINITY) all_m = 0.f;                // a token with no keys and no sink: cannot happen, but do not NaN
    float all_s = sinks ? expf(sinks[h] - all_m) : 0.f;
    for (int w = 0; w < warps; ++w) all_s += ws[w] * expf(wm[w] - all_m);

    float* o = out + (size_t(t) * n_head + h) * dim_v;
    for (int j = 0; j < vpr; ++j) {
        const int i = WARP * j + lane;
        float vsum = 0.f;
        for (int w = 0; w < warps; ++w) vsum += wacc[size_t(w) * dim_v + i] * expf(wm[w] - all_m);
        o[i] = vsum / all_s;
    }
}

// The tile's shape (host dispatch reads these whether or not the kernel below is compiled): one block a group
// of eight heads (one kv head at the model's ratios) and eight of its tokens, keys 64 at a time.
constexpr int MMA_T = 8;                 // tokens a block carries: 8 x 8 heads = 64 rows = 4 warps x 2 tokens
constexpr int MMA_G = 8;                 // heads a block carries, sharing their kv head
constexpr int MMA_KEYS = 64;             // keys a block takes at a time
constexpr int MMA_ROWK = 192 + 8;        // a shared row of keys, padded so ldmatrix misses no bank
constexpr int MMA_ROWV = 128 + 8;        // and of values
constexpr size_t MMA_SMEM = size_t(2) * MMA_KEYS * (MMA_ROWK + MMA_ROWV) * sizeof(half);

#if STRATA_MIMO_MMA
// A batch read's attend on the tensor cores - MiMo's head of kernels.cu's attn_flash128_kernel: dim_k of 192
// (twelve k-steps a score) and dim_v of 128, both read from the model's own fp16 caches.  One block a group of
// eight heads (one kv head at the model's ratios) and eight of its tokens: the 16 query rows a warp owns are
// two tokens' eight heads, as the m16 fragment has them.  Keys come 64 at a time into shared (cp.async, two
// buffers, the next on its way while one is worked on) and leave it for no other reader: a key row fed eight
// heads' scores here the reference walked eight times over.  The scores stay in the fragments; the row max
// and sum of the softmax go on online in log2 units, the causal and window masks strike whole fragments
// outside the row's range, and the sink joins at the end as one more weight of no value.  The queries and
// the weights round to halves and the products sum floats, as Flint's and llama's do.
__global__ void __launch_bounds__(4 * WARP)
mimo2_attend_mma_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                        const uint16_t* __restrict__ vcache, long long pos, int rows, float* __restrict__ out,
                        int n_head, int kv_heads, float scale_log2, int window, int ring,
                        const long long* __restrict__ at, const float* __restrict__ sinks) {
    constexpr int DK = 192, DV = 128, KS = DK / 16;   // twelve k-steps a score, eight n-tiles a key block
    extern __shared__ __align__(16) half flash_sm[];
    half(*ks)[MMA_KEYS][MMA_ROWK] = reinterpret_cast<half(*)[MMA_KEYS][MMA_ROWK]>(flash_sm);
    half(*vs)[MMA_KEYS][MMA_ROWV] = reinterpret_cast<half(*)[MMA_KEYS][MMA_ROWV]>(flash_sm + 2 * MMA_KEYS * MMA_ROWK);
    const int g = blockIdx.y, t0 = blockIdx.x * MMA_T, lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int gid = lane / 4, tig = lane % 4, head = g * MMA_G + gid;
    const int kvh = (g * MMA_G) / (n_head / kv_heads);  // the block's heads share it: the entry checked the ratio
    const long long p0 = at ? *at : pos;                 // the launch's first token's position
    const int tok[2] = {t0 + 2 * w, t0 + 2 * w + 1};     // the thread's rows gid and gid + 8: a head of two tokens
    uint32_t qa[KS][4];   // the queries, twelve k-steps, in log2 units
    for (int k = 0; k < KS; ++k)
        for (int i = 0; i < 2; ++i)
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < rows)
                    x = *reinterpret_cast<const float2*>(q + (size_t(tok[i]) * n_head + head) * DK + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = half2_bits(x.x * scale_log2, x.y * scale_log2);
            }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = p0 + min(t0 + MMA_T, rows);
    const long long lo = window > 0 ? max(0LL, p0 + t0 - window + 1) : 0;
    const auto fetch = [&](int buf, long long k0) {
        for (int i = threadIdx.x; i < MMA_KEYS * DK / 8; i += 4 * WARP) {
            const int r = i / (DK / 8), c = 8 * (i % (DK / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            const size_t slot = whole ? (size_t(ring > 0 ? key % ring : key) * kv_heads + kvh) * DK + c : 0;
            flash_copy(&ks[buf][r][c], reinterpret_cast<const half*>(kcache) + slot, whole);
        }
        for (int i = threadIdx.x; i < MMA_KEYS * DV / 8; i += 4 * WARP) {
            const int r = i / (DV / 8), c = 8 * (i % (DV / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;
            const size_t slot = whole ? (size_t(ring > 0 ? key % ring : key) * kv_heads + kvh) * DV + c : 0;
            flash_copy(&vs[buf][r][c], reinterpret_cast<const half*>(vcache) + slot, whole);
        }
        flash_commit();
    };
    const long long first = lo / MMA_KEYS * MMA_KEYS;
    fetch(0, first);
    int buf = 0;
    for (long long k0 = first; k0 < hi; k0 += MMA_KEYS, buf ^= 1) {
        __syncthreads();   // the buffer the last block was worked on in is free
        if (k0 + MMA_KEYS < hi) {
            fetch(buf ^ 1, k0 + MMA_KEYS);
            flash_wait<1>();
        } else {
            flash_wait<0>();
        }
        __syncthreads();
        float s[8][4];
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < KS; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        for (int i = 0; i < 2; ++i) {
            const long long own = p0 + tok[i];   // the row's token attends positions (own - window, own]
            float mx = -INFINITY;
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    if (key > own || (window > 0 && key <= own - window)) s[n][j] = -INFINITY;
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
            for (int n = 0; n < 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        float f = 1.0f;
        if (sinks) {   // the sink, in log2 units like the scores, as one more weight of no value
            const float sink = sinks[head] * 1.44269504f, now = fmaxf(m[i], sink);
            f = exp2f(m[i] - now);
            l[i] = l[i] * f + exp2f(sink - now);
        }
        if (tok[i] >= rows) continue;
        float* y = out + (size_t(tok[i]) * n_head + head) * DV + 2 * tig;
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] * f / l[i], o[n][2 * i + 1] * f / l[i]);
    }
}
#endif  // STRATA_MIMO_MMA

}  // namespace

void mimo2_rope_split(const float* src, int row, int rows, float* q, float* k, float* v, int dim_k, int dim_v,
                      int n_head, int kv_heads, int n_rot, float theta_scale, int64_t pos, void* stream,
                      const long long* at) {
    if (dim_k > kMaxDims || n_rot > kMaxRot || n_rot % 2 || dim_k % 4)
        throw std::invalid_argument("mimo2_rope_split: bad head dimensions");
    if (at && !rows) throw std::invalid_argument("mimo2_rope_split: no rows");
    const dim3 grid(n_head + 2 * kv_heads, rows);
    mimo2_rope_split_kernel<<<grid, 192, 0, (cudaStream_t) stream>>>(
        src, row, q, k, v, dim_k, dim_v, n_head, kv_heads, n_rot, theta_scale, (long long) pos, at);
}

namespace {
// The card answers once: the tensor cores want Ampere or later, and the shared memory the tile wants wants
// asking once per kernel too.
bool mma_ready() {
    static const bool ready = [] {
#if !defined(__HIPCC__) && STRATA_MIMO_MMA
        int dev = 0, cc_major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
        if (cc_major < 8) return false;     // the tile's cp.async and m16n8k16 are Ampere's
        return cudaFuncSetAttribute(mimo2_attend_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(MMA_SMEM)) == cudaSuccess;
#else
        return false;
#endif
    }();
    return ready;
}

// STRATA_MIMO_ATTN_TRACE=1: every attend prints its chosen kernel and its CUDA time to stderr.  The events
// and the wait they ask for exist only while tracing - an untraced call launches exactly what it would have
// launched without the knob.  The engine calls attend from its own thread alone, so one pair of events is
// enough for the whole process.
struct Trace {
    cudaEvent_t a = nullptr, b = nullptr;
    bool on = false;
    Trace() {
        const char* e = getenv("STRATA_MIMO_ATTN_TRACE");
        on = e && e[0] == '1';
        if (on && cudaEventCreate(&a) != cudaSuccess) a = nullptr;
        if (on && cudaEventCreate(&b) != cudaSuccess) b = nullptr;
        if (!a || !b) on = false;
    }
};

Trace& trace() {
    static Trace t;
    return t;
}

// STRATA_MIMO_ATTN_NO_MMA=1: the entry never picks the tensor-core tile - the bench's own control, and the
// engine's kill switch if a card ever disagrees with the dispatch. Default off; the launch is then exactly
// the choice the checks make. Read every call, so a process can measure both ways; a getenv per attend,
// which is a pointer walk beside a kernel launch.
bool mma_off() {
    const char* e = getenv("STRATA_MIMO_ATTN_NO_MMA");
    return e && e[0] == '1';
}
}  // namespace

void mimo2_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* out, int dim_k, int dim_v, int n_head, int kv_heads, float scale, void* stream,
                  int window, int ring, const long long* at, int room, const float* sinks) {
    if (dim_k % WARP || dim_v % WARP || dim_v > 8 * WARP || dim_k > kMaxDims)
        throw std::invalid_argument("mimo2_attend: a head's values must split over a warp");
    if (n_head % kv_heads) throw std::invalid_argument("mimo2_attend: the heads do not divide");
    // one block per query head and token: a position on the card changes where the rows read, not how many
    // there are - and a room over 65535 (gridDim.y's ceiling) would refuse the launch outright
    (void) room;
    if (rows < 1) return;
    const cudaStream_t st = (cudaStream_t) stream;
    const bool mma = rows >= kMmaRows && dim_k == 192 && dim_v == 128 && n_head % MMA_G == 0 &&
                     (n_head / kv_heads) % MMA_G == 0 && !mma_off() && mma_ready();
    Trace& tr = trace();
    if (tr.on) cudaEventRecord(tr.a, st);
#if !defined(__HIPCC__) && STRATA_MIMO_MMA
    if (mma) {
        const dim3 grid((rows + MMA_T - 1) / MMA_T, n_head / MMA_G);
        mimo2_attend_mma_kernel<<<grid, 4 * WARP, MMA_SMEM, st>>>(
            q, kcache, vcache, (long long) pos, rows, out, n_head, kv_heads, scale * 1.44269504f, window,
            ring, at, sinks);
    } else
#endif
    {
        const int warps = 4;
        const size_t merge_bytes = size_t(2 * WARP + warps * dim_v) * sizeof(float);
        const dim3 grid(n_head, rows);
        mimo2_attend_split_kernel<<<grid, warps * WARP, merge_bytes, st>>>(
            q, kcache, vcache, (long long) pos, rows, out, dim_k, dim_v, n_head, kv_heads, scale, window, ring,
            at, sinks);
    }
    if (tr.on) {
        cudaEventRecord(tr.b, st);
        if (cudaEventSynchronize(tr.b) == cudaSuccess) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, tr.a, tr.b);
            std::fprintf(stderr, "mimo2-attend: kernel=%s rows=%d %s pos=%lld %.2f ms\n",
                         mma ? "tiled_mma" : "split", rows, window > 0 ? "swa" : "global",
                         at ? (long long) -1 : (long long) pos, double(ms));   // pos on the card stays there
        }
    }
}

}  // namespace strata::poly
