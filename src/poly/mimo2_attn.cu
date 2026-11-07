// src/poly/mimo2_attn.cu - see the header.  The reference for every formula is llama.cpp's mimo2 graph
// (src/models/mimo2.cpp): NeoX rotary on the first n_rot values, scores q.k / sqrt(dim_k), a key hidden when
// p - key_pos >= window, and on a window layer one sink logit a query head in the softmax's sum.
//
// Two attend kernels behind one entry, split by how many rows the call carries:
// - a call of kStagedRows rows or more (a prompt batch) takes mimo2_attend_staged_kernel: one block a token
//   and eight query heads, so its warps walk the kv head's rows once between them, staged through shared
//   memory in uint4 pieces - which asks the eight heads to share one kv head, true of both of the model's
//   geometries and checked before the launch.  The row count says a call has blocks enough to keep the card busy with one
//   warp's serial walk per (token, head), and then the walk is the whole of the read's bandwidth too.
// - fewer rows (a decoded token, a guess block) take mimo2_attend_split_kernel: the reference's shape, four
//   warps sharing one head's walk with a merge behind them.  Few queries cannot fill the card with walks of
//   their own; splitting one head's keys across warps is what keeps the latency down when there is nothing
//   else to overlap with.
#include "mimo2_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int kMaxRot = 256;      // a rope block keeps a head's turned part in shared
constexpr int kMaxDims = 256;     // dim_k values a rope block keeps in shared
constexpr int kStagedRows = 64;   // rows from which the staged walk has enough blocks to win
constexpr int kBatch = 32;        // cache rows a staged block brings through shared at a time

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

// A batch read's attend: one block a token and the eight heads of one kv group (both of MiMo's geometries
// give a group of eight or sixteen, so eight consecutive heads share their kv head).  The block's warps keep
// their own queries in registers and share the walk over the kv head's rows: kBatch rows at a time come into
// shared as uint4, and every warp reads them with the same strided values the reference reads from the cache.
// One walk over the rows feeds eight heads where the reference walked eight - and it arrives 16 bytes a lane
// where the reference took the values two bytes at a time, which is the whole of why a global layer of a
// 32,768-token prompt used to take twelve seconds.  The online softmax, the sink's join and the written row
// are the reference's own arithmetic, a warp to itself.
__global__ void mimo2_attend_staged_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                                           const uint16_t* __restrict__ vcache, long long pos, int rows,
                                           float* __restrict__ out, int dim_k, int dim_v, int n_head,
                                           int kv_heads, float scale, int window, int ring,
                                           const long long* __restrict__ at, const float* __restrict__ sinks) {
    const int t = blockIdx.y;
    if (t >= rows) return;
    const long long p = (at ? *at : pos) + t;           // this token's position
    const int h = int(blockIdx.x) * 8 + int(threadIdx.x >> 5);
    const int kvh = h / (n_head / kv_heads);
    const long long lo = window > 0 && p >= window ? p - window + 1 : 0;
    const int n_keys = int(p - lo) + 1;

    const int lane = threadIdx.x & (WARP - 1);

    // the query's values this lane owns, kept in registers for the whole walk
    const float* qh = q + (size_t(t) * n_head + h) * dim_k;
    float qreg[kMaxDims / WARP];
    for (int i = lane, j = 0; i < dim_k; i += WARP, ++j) qreg[j] = qh[i];

    float m = -INFINITY, s = 0.f;
    float acc[8];
    const int vpr = dim_v / WARP;
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;

    extern __shared__ __half stage[];                   // kBatch rows of dim_k keys then dim_v values
    const int krow = dim_k + dim_v;                     // halves in a staged row
    const int ku = dim_k / 8, vu = dim_v / 8;           // uint4 of a staged row's key and value part

    for (int base = 0; base < n_keys; base += kBatch) {
        const int cnt = min(kBatch, n_keys - base);
        __syncthreads();                                // the rows staged last turn have been read out
        for (int i = int(threadIdx.x), n = cnt * (ku + vu); i < n; i += int(blockDim.x)) {
            const int r = i / (ku + vu), c = i % (ku + vu);
            const long long cr = ring ? (lo + base + r) % ring : lo + base + r;
            const bool keys = c < ku;                   // the row's key part, then its values
            const uint4* g = keys
                ? (const uint4*)(kcache + (size_t(cr) * kv_heads + kvh) * dim_k) + c
                : (const uint4*)(vcache + (size_t(cr) * kv_heads + kvh) * dim_v) + (c - ku);
            *(uint4*)(stage + size_t(r) * krow + (keys ? c * 8 : dim_k + (c - ku) * 8)) = *g;
        }
        __syncthreads();
        for (int key = 0; key < cnt; ++key) {
            const __half* kr = stage + size_t(key) * krow;
            float dot = 0.f;
            for (int i = lane, j = 0; i < dim_k; i += WARP, ++j) dot += qreg[j] * __half2float(kr[i]);
#pragma unroll
            for (int d = WARP / 2; d >= 1; d >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, d);
            const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
            const float nm = score > m ? score : m;
            const float shrink = expf(m - nm), add = expf(score - nm);
            s = s * shrink + add;
            const __half* vr = kr + dim_k;
            for (int j = 0; j < vpr; ++j) acc[j] = acc[j] * shrink + add * __half2float(vr[WARP * j + lane]);
            m = nm;
        }
    }

    // the sink joins as the reference's merge joined it, and the head's row goes out
    float all_m = sinks ? sinks[h] : -INFINITY;         // exp(sink - max) cannot overflow
    all_m = m > all_m ? m : all_m;
    if (all_m == -INFINITY) all_m = 0.f;                // a token with no keys and no sink: cannot happen, but do not NaN
    const float all_s = (sinks ? expf(sinks[h] - all_m) : 0.f) + s * expf(m - all_m);
    const float keep = expf(m - all_m);
    float* o = out + (size_t(t) * n_head + h) * dim_v;
    for (int j = 0; j < vpr; ++j) o[WARP * j + lane] = acc[j] * keep / all_s;
}

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
    if (rows >= kStagedRows && n_head % 8 == 0 && n_head / kv_heads >= 8) {   // a batch read: the staged walk,
        // a block a token - and a block's eight heads must share their kv head, which they do only when a
        // group holds eight of them (both of the model's geometries do; a cheap test geometry may not)
        const dim3 grid(n_head / 8, rows);
        const size_t stage_bytes = size_t(kBatch) * (dim_k + dim_v) * sizeof(__half);
        mimo2_attend_staged_kernel<<<grid, 8 * WARP, stage_bytes, (cudaStream_t) stream>>>(
            q, kcache, vcache, (long long) pos, rows, out, dim_k, dim_v, n_head, kv_heads, scale, window,
            ring, at, sinks);
        return;
    }
    const int warps = 4;
    const size_t merge_bytes = size_t(2 * WARP + warps * dim_v) * sizeof(float);
    const dim3 grid(n_head, rows);
    mimo2_attend_split_kernel<<<grid, warps * WARP, merge_bytes, (cudaStream_t) stream>>>(
        q, kcache, vcache, (long long) pos, rows, out, dim_k, dim_v, n_head, kv_heads, scale, window, ring, at,
        sinks);
}

}  // namespace strata::poly
