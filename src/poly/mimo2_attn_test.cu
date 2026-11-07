// src/poly/mimo2_attn_test.cu - MiMo's rope_split and attend against a scalar host reference.
//
// The reference is written twice on purpose: the kernels in mimo2_attn.cu, and here a plain double loop over the
// same fp16 numbers, the math taken from llama.cpp's mimo2 graph (MIMO_ATTN's rule): NeoX rotary on the first 64
// of 192 values, scores q.k / sqrt(192), a key hidden when p - key_pos >= 128, and on a window layer one sink
// logit a head in the sum.  This reference shares no code with the kernels.
//
// usage: poly_mimo2_attn_test   (exit 0: every case within 2e-3 of the reference and no NaN)
#include "mimo2_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using namespace strata::poly;

// MiMo's own numbers: key 192, value 128, the rope's 64 of the key, the base 1e7 and 1e4 (theta as
// powers of it stay the test's own, both sides turn by the same scale).  The head count is a
// parameter: the cheap cases run 8, the cases below run the model's 64 with its true ratios.
constexpr int kDimK = 192, kDimV = 128, kRot = 64;
constexpr float kThetaGlobal = 0.9f, kThetaSwa = 0.7f;

unsigned rng_state = 12345;
float frand() {   // deterministic: every run checks the same numbers
    rng_state = rng_state * 1664525u + 1013904223u;
    return (int(rng_state >> 8) % 20000) / 10000.f - 1.f;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

int fails = 0;
void report(const char* what, double most, double tol) {
    const bool ok = !(most > tol) && !(std::isnan(most) || std::isinf(most));
    std::printf("%s %-44s most %.3e (tol %.0e)\n", ok ? "ok  " : "FAIL", what, most, tol);
    if (!ok) ++fails;
}

// the host's NeoX rule: value i goes with value i + n_rot / 2, pair i turning by position * theta_scale^i
void ref_rope(const float* head, int dim, int n_rot, float theta_scale, long long pos, std::vector<float>& y) {
    y.assign(dim, 0.f);
    const int half = n_rot / 2;
    for (int i = 0; i < dim; ++i) {
        if (i >= n_rot) { y[i] = head[i]; continue; }
        const int j = i < half ? i : i - half;
        const double turn = double(pos) * std::pow(double(theta_scale), double(j));
        const double x0 = head[j], x1 = head[j + half];
        y[i] = float(i < half ? x0 * std::cos(turn) - x1 * std::sin(turn)
                              : x0 * std::sin(turn) + x1 * std::cos(turn));
    }
}

// one test: a random fused row per token; the kernel splits and turns it, the host does the same twice over.
// then the q/k/v go through fp16 into caches and the attention runs against the host's sums.
void test_layer(int n_head, int n_kv, bool swa, long long pos, int rows, bool with_at, int window = 64,
                int room = 0) {
    const int kWindow = window;
    const int row = kDimK * (n_head + n_kv) + kDimV * n_kv;
    // the engine's ring: window + the rows one call takes (mimo2.cpp: ring_ = window_ + rows_) - with any
    // smaller ring a call's own rows would overwrite history its first tokens still read, and the host,
    // reading the cache AFTER those writes, would compare against the corruption and call it the truth
    const int ring = swa ? kWindow + rows : 0;
    std::vector<float> src(size_t(rows) * row);
    for (float& x : src) x = frand();

    float* dsrc = upload(src);
    std::vector<float> dq(size_t(rows) * n_head * kDimK, -777.f), dk(size_t(rows) * n_kv * kDimK, -777.f),
        dv(size_t(rows) * n_kv * kDimV, -777.f);
    float *dq_d = nullptr, *dk_d = nullptr, *dv_d = nullptr;
    cudaMalloc(&dq_d, dq.size() * sizeof(float));
    cudaMalloc(&dk_d, dk.size() * sizeof(float));
    cudaMalloc(&dv_d, dv.size() * sizeof(float));

    long long* at_d = nullptr;
    if (with_at) {
        cudaMalloc(&at_d, sizeof(long long));
        cudaMemcpy(at_d, &pos, sizeof(long long), cudaMemcpyHostToDevice);
        mimo2_rope_split(dsrc, row, rows, dq_d, dk_d, dv_d, kDimK, kDimV, n_head, n_kv, kRot,
                         swa ? kThetaSwa : kThetaGlobal, 0, nullptr, at_d);
    } else {
        mimo2_rope_split(dsrc, row, rows, dq_d, dk_d, dv_d, kDimK, kDimV, n_head, n_kv, kRot,
                         swa ? kThetaSwa : kThetaGlobal, pos, nullptr, nullptr);
    }
    dq = download(dq_d, dq.size());
    dk = download(dk_d, dk.size());
    dv = download(dv_d, dv.size());

    // the host split and turn, in double over the same floats.  A turn of 100000 radians is stored in a
    // float, and a float at that size has steps of 0.0078: the angle alone moves sin and cos by about
    // that, the same in the engine's kernel and in llama.cpp's rope.  The tolerance grows with the
    // position so a deep conversation tests the split and the pairs, not the float's ceiling.
    const double rope_tol = std::max(3e-5, double(pos) * 2.5e-7);
    double most = 0.;
    std::vector<float> y;
    for (int t = 0; t < rows; ++t) {
        const float* src_row = src.data() + size_t(t) * row;
        for (int h = 0; h < n_head; ++h) {
            ref_rope(src_row + size_t(h) * kDimK, kDimK, kRot, swa ? kThetaSwa : kThetaGlobal, pos + t, y);
            for (int i = 0; i < kDimK; ++i) {
                const double d = std::fabs(y[i] - dq[size_t(t) * n_head * kDimK + size_t(h) * kDimK + i]);
                if (d == most && most > rope_tol)
                    std::printf("  first diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, y[i],
                                dq[size_t(t) * n_head * kDimK + size_t(h) * kDimK + i]);
                most = std::max(most, d);
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            ref_rope(src_row + size_t(n_head + h) * kDimK, kDimK, kRot, swa ? kThetaSwa : kThetaGlobal, pos + t, y);
            for (int i = 0; i < kDimK; ++i) {
                const double d = std::fabs(y[i] - dk[size_t(t) * n_kv * kDimK + size_t(h) * kDimK + i]);
                if (d > rope_tol)
                    std::printf("  k diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, y[i],
                                dk[size_t(t) * n_kv * kDimK + size_t(h) * kDimK + i]);
                most = std::max(most, d);
            }
            for (int i = 0; i < kDimV; ++i) {   // the rows v is packed dim_v a head, after the keys dim_k a head
                const double want = src_row[size_t(n_head + n_kv) * kDimK + size_t(h) * kDimV + i];
                const double d = std::fabs(want - dv[size_t(t) * n_kv * kDimV + size_t(h) * kDimV + i]);
                if (d > rope_tol)
                    std::printf("  v diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, want,
                                dv[size_t(t) * n_kv * kDimV + size_t(h) * kDimV + i]);
                most = std::max(most, d);
            }
        }
    }
    char name[160];
    std::snprintf(name, sizeof(name), "rope_split %d/%d %s win %d pos %lld rows %d%s", n_head, n_kv,
                  swa ? "swa" : "global", swa ? window : 0, pos, rows, with_at ? " (pos on card)" : "");
    report(name, most, rope_tol);

    // ---- the attention: the fp16-rounded sides become the caches, and past rows come from a filled cache ----
    // rows before the call: a fresh fill per position, stored the way the engine stores (pos % ring)
    const long long first = swa ? std::max(0LL, pos - kWindow) : 0;
    std::vector<float> qh = dq;   // the kernel's own q: the test compares sums over these exact numbers
    std::vector<uint16_t> kcache(size_t(ring ? ring : pos + rows) * n_kv * kDimK, 0),
        vcache(size_t(ring ? ring : pos + rows) * n_kv * kDimV, 0);
    auto store_row = [&](long long p, const std::vector<float>& kflat, const std::vector<float>& vflat, int t) {
        const long long cr = ring ? p % ring : p;
        for (int h = 0; h < n_kv; ++h)
            for (int i = 0; i < kDimK; ++i)
                kcache[size_t(cr) * n_kv * kDimK + size_t(h) * kDimK + i] =
                    __half_as_ushort(__float2half_rn(kflat[size_t(t) * n_kv * kDimK + size_t(h) * kDimK + i]));
        for (int h = 0; h < n_kv; ++h)
            for (int i = 0; i < kDimV; ++i)
                vcache[size_t(cr) * n_kv * kDimV + size_t(h) * kDimV + i] =
                    __half_as_ushort(__float2half_rn(vflat[size_t(t) * n_kv * kDimV + size_t(h) * kDimV + i]));
    };
    for (long long p = first; p < pos; ++p) {   // the rows the call attends that it did not write itself
        std::vector<float> kprev(kDimK * n_kv), vprev(kDimV * n_kv);
        for (float& x : kprev) x = frand();
        for (float& x : vprev) x = frand();
        store_row(p, kprev, vprev, 0);
    }
    // host-side copies of the stored rows, in fp16, so the reference sums over exactly the cache's numbers
    std::vector<std::vector<float>> kref(size_t(ring ? ring : pos + rows)), vref(size_t(ring ? ring : pos + rows));
    for (size_t r = 0; r < kref.size(); ++r) {
        kref[r].assign(size_t(n_kv) * kDimK, 0.f);
        vref[r].assign(size_t(n_kv) * kDimV, 0.f);
        for (int i = 0; i < n_kv * kDimK; ++i)
            kref[r][i] = __half2float(__ushort_as_half(kcache[size_t(r) * n_kv * kDimK + i]));
        for (int i = 0; i < n_kv * kDimV; ++i)
            vref[r][i] = __half2float(__ushort_as_half(vcache[size_t(r) * n_kv * kDimV + i]));
    }
    for (int t = 0; t < rows; ++t) store_row(pos + t, dk, dv, t);   // this call's rows, as the engine stores them
    for (size_t r = 0; r < kref.size(); ++r) {
        for (int i = 0; i < n_kv * kDimK; ++i)
            kref[r][i] = __half2float(__ushort_as_half(kcache[size_t(r) * n_kv * kDimK + i]));
        for (int i = 0; i < n_kv * kDimV; ++i)
            vref[r][i] = __half2float(__ushort_as_half(vcache[size_t(r) * n_kv * kDimV + i]));
    }

    std::vector<float> sinks(n_head);
    if (swa) for (float& x : sinks) x = frand() * 3.f;
    uint16_t* dkc = upload(kcache);
    uint16_t* dvc = upload(vcache);
    float* dsinks = swa ? upload(sinks) : nullptr;
    float* dout_d = nullptr;
    cudaMalloc(&dout_d, size_t(rows) * n_head * kDimV * sizeof(float));
    if (with_at) {
        cudaMemcpy(at_d, &pos, sizeof(long long), cudaMemcpyHostToDevice);
        // the room the engine would give a recorded launch at this position: the sizes that ran the
        // grid past gridDim.y's ceiling before (window layers a small one, global layers room(n))
        mimo2_attend(dq_d, (uint16_t*) dkc, (uint16_t*) dvc, 0, rows, dout_d, kDimK, kDimV, n_head, n_kv,
                     1.f / std::sqrt((float) kDimK), nullptr, swa ? kWindow : 0, ring, at_d,
                     room ? room : std::max(rows, 8), swa ? dsinks : nullptr);
    } else {
        mimo2_attend(dq_d, (uint16_t*) dkc, (uint16_t*) dvc, pos, rows, dout_d, kDimK, kDimV, n_head, n_kv,
                     1.f / std::sqrt((float) kDimK), nullptr, swa ? kWindow : 0, ring, nullptr, 0,
                     swa ? dsinks : nullptr);
    }
    const std::vector<float> got = download(dout_d, size_t(rows) * n_head * kDimV);

    const double scale = 1.0 / std::sqrt((double) kDimK);
    most = 0.;
    bool nan = false;
    for (int t = 0; t < rows; ++t) {
        const long long p = pos + t;
        const long long lo = swa && p >= kWindow ? p - kWindow + 1 : 0;
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / (n_head / n_kv);
            const float* q = qh.data() + (size_t(t) * n_head + h) * kDimK;
            std::vector<double> score;
            std::vector<const std::vector<float>*> rows_of;
            double mx = swa ? sinks[h] : -INFINITY;
            for (long long r = lo; r <= p; ++r) {
                const long long cr = ring ? r % ring : r;
                const std::vector<float>& krow = kref[cr];
                double dot = 0.;
                for (int i = 0; i < kDimK; ++i) dot += double(q[i]) * krow[size_t(kvh) * kDimK + i];
                score.push_back(dot * scale);
                rows_of.push_back(&vref[cr]);
                mx = std::max(mx, score.back());
            }
            double sum = swa ? std::exp(sinks[h] - mx) : 0.;
            for (double s : score) sum += std::exp(s - mx);
            for (int i = 0; i < kDimV; ++i) {
                double acc = 0.;
                for (size_t j = 0; j < score.size(); ++j)
                    acc += std::exp(score[j] - mx) / sum * (*rows_of[j])[size_t(kvh) * kDimV + i];
                const double gotv = got[size_t(t) * n_head * kDimV + size_t(h) * kDimV + i];
                if (std::isnan(gotv) || std::isinf(gotv)) nan = true;
                most = std::max(most, std::fabs(acc - gotv));
            }
        }
    }
    // a NaN must reach report(): std::max keeps its first argument against a NaN, so without this the
    // difference stays a number, report() calls it ok, and a NaN output exits 0 with one printed line
    if (nan) most = std::numeric_limits<double>::quiet_NaN();
    std::snprintf(name, sizeof(name), "attend   %d/%d %s win %d pos %lld rows %d%s", n_head, n_kv,
                  swa ? "swa" : "global", swa ? window : 0, pos, rows, with_at ? " (pos on card)" : "");
    report(name, most, 2e-3);

    cudaFree(dsrc); cudaFree(dq_d); cudaFree(dk_d); cudaFree(dv_d); cudaFree(dkc); cudaFree(dvc);
    cudaFree(dsinks); cudaFree(dout_d); cudaFree(at_d);
}

// CUDA Graph: what the engine's recorded launches really are. The attend is captured on a stream with the
// position in a card buffer, replayed for two different positions, and every replay must equal a direct
// launch at the same position to the bit - same kernel, same arguments, no atomics. room is the engine's
// room(n), and the direct comparison passes the identical buffer as a plain at, so the two paths differ
// only in the graph around them.
void test_graph() {
    constexpr int NH = 64, NKV = 4, ROWS = 96;
    const long long pos_a = 900, pos_b = 1400;
    const int cache_rows = 2000;
    std::vector<float> q(size_t(ROWS) * NH * kDimK);
    for (float& x : q) x = frand();
    std::vector<uint16_t> kc(size_t(cache_rows) * NKV * kDimK), vc(size_t(cache_rows) * NKV * kDimV);
    for (uint16_t& x : kc) x = __half_as_ushort(__float2half_rn(frand()));
    for (uint16_t& x : vc) x = __half_as_ushort(__float2half_rn(frand()));
    float* dq = upload(q);
    uint16_t* dkc = upload(kc);
    uint16_t* dvc = upload(vc);
    long long* dat = nullptr;
    cudaMalloc(&dat, sizeof(long long));
    float *d_ref = nullptr, *d_got = nullptr;
    cudaMalloc(&d_ref, size_t(ROWS) * NH * kDimV * sizeof(float));
    cudaMalloc(&d_got, size_t(ROWS) * NH * kDimV * sizeof(float));
    const float scale = 1.f / std::sqrt((float) kDimK);
    const size_t out_n = size_t(ROWS) * NH * kDimV;
    const auto direct = [&](long long p, float* dst) {
        cudaMemcpy(dat, &p, sizeof(long long), cudaMemcpyHostToDevice);
        mimo2_attend(dq, dkc, dvc, p, ROWS, dst, kDimK, kDimV, NH, NKV, scale, nullptr, 0, 0, dat, 1024,
                     nullptr);
        cudaDeviceSynchronize();
    };
    // capture one launch with the position on the card, as the engine records it
    cudaMemcpy(dat, &pos_a, sizeof(long long), cudaMemcpyHostToDevice);
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    mimo2_attend(dq, dkc, dvc, 0, ROWS, d_got, kDimK, kDimV, NH, NKV, scale, (void*) s, 0, 0, dat, 1024,
                 nullptr);
    cudaGraph_t graph = nullptr;
    const bool captured = cudaStreamEndCapture(s, &graph) == cudaSuccess && graph;
    cudaGraphExec_t exec = nullptr;
    if (captured) cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    double worst = 0.;
    if (exec) {
        for (const long long p : {pos_a, pos_b, pos_a}) {   // the same exec, replayed at shifting positions
            cudaMemcpy(dat, &p, sizeof(long long), cudaMemcpyHostToDevice);
            cudaGraphLaunch(exec, s);
            cudaStreamSynchronize(s);
            direct(p, d_ref);
            const std::vector<float> got = download(d_got, out_n), ref = download(d_ref, out_n);
            for (size_t i = 0; i < out_n; ++i)
                if (!std::isnan(got[i]) && !std::isinf(got[i]))
                    worst = std::max(worst, double(std::fabs(got[i] - ref[i])));
        }
    } else
        worst = 1.;
    report(captured ? "graph replay 64/4 rows 96, two positions" : "graph capture FAILED", worst, 0.0);
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(dq); cudaFree(dkc); cudaFree(dvc); cudaFree(dat); cudaFree(d_ref); cudaFree(d_got);
}

}  // namespace

int main() {
    // a global layer: 4 kv heads, no window, rows from 0; then one deep in a long conversation.
    // 8 query heads: cheap cases where only the split, the ring and the window edges are under test
    test_layer(8, 4, false, 0, 1, false);
    test_layer(8, 4, false, 7, 4, false);
    test_layer(8, 4, false, 333, 1, false);
    test_layer(8, 4, false, 333, 1, true);       // the recorded launch's position, on the card
    // a window layer: the window's edges 127/128/129 (kWindow 64 here), the ring's wrap and a window of guesses
    test_layer(8, 8, true, 62, 1, false);
    test_layer(8, 8, true, 63, 1, false);
    test_layer(8, 8, true, 64, 1, false);        // the first key hidden
    test_layer(8, 8, true, 65, 3, false);        // tokens of one call see each other's rows
    test_layer(8, 8, true, 71, 1, false);        // the ring has wrapped once (ring = window + rows)
    test_layer(8, 8, true, 72, 8, false);
    test_layer(8, 8, true, 150, 2, true);        // wrapped again, position on the card
    test_layer(8, 4, true, 140, 1, false);
    // the model's own window of 128 and its exact edges 127 / 128 / 129, and a ring that has wrapped
    test_layer(8, 8, true, 126, 1, false, 128);
    test_layer(8, 8, true, 127, 1, false, 128);
    test_layer(8, 8, true, 128, 1, false, 128);
    test_layer(8, 8, true, 129, 1, false, 128);
    test_layer(8, 8, true, 260, 2, false, 128);
    test_layer(8, 8, true, 260, 8, true, 128);
    // the batch read's rows: from kMmaRows the entry switches to the tensor-core tile (a block of 8 tokens
    // and 8 heads, m16n8k16 scores and weights), so those cases need row counts the reference cases above
    // never reach. Positions stay moderate where the host sums the whole global history - seconds, not minutes.
    test_layer(64, 4, false, 70, 64, true);             // the switch's own row count, position on the card
    test_layer(64, 4, false, 333, 32, false);           // one row short of the tile: the split kernel's own
    test_layer(64, 4, false, 333, 63, false);           // the edge from below
    test_layer(64, 4, false, 333, 64, false);           // and from above: the tile's first launch
    test_layer(64, 4, false, 333, 65, false);           // a tile and a stray token
    test_layer(64, 4, false, 300, 96, false);           // the model's global geometry, a batch of rows
    test_layer(64, 4, false, 333, 128, false);          // two token tiles
    test_layer(64, 4, false, 333, 512, false);          // many tiles, tokens seeing each other's rows
    test_layer(64, 8, true, 400, 96, false, 128);       // and the window one, rows across the window's edge
    test_layer(8, 8, true, 260, 80, false, 128);        // the ring wrapping while a batch of rows reads
    test_layer(8, 4, false, 10, 70, false);             // the tile's guards with the cheap head ratios -
    //   8 heads over 2 kv groups is not the model's geometry: the entry must hand it to the split kernel
    test_layer(64, 4, false, 32767, 1, false);          // the 32k read chunk's own edges, as the speed
    test_layer(64, 4, false, 32768, 1, false);          //   bench's prompts stand on them
    test_layer(64, 4, false, 50000, 1, false);          // and the depth a 50000-token prompt reaches
    test_layer(64, 8, true, 50000, 65, false, 128);     // a batch at that depth, the ring wrapped many times
    test_layer(64, 8, true, 100000, 65, true, 128);     // the model's full 131072 context, position on card
    // the model's real geometry: 64 query heads over 4 key heads global (16 to 1) and 8 window (8 to
    // 1), read at a position a long conversation actually stands at - the ratios the engine runs and
    // the only head counts its files give.  The host sums over the whole global history, so these two
    // are the slow cases of this test, a few seconds each.
    test_layer(64, 8, true, 100000, 1, false, 128);
    test_layer(64, 4, false, 100000, 1, false);
    // the grid's own regression: a recorded launch (the position on the card) at the room a global
    // layer reaches past 65535 tokens - grid.y of n_head x room is past CUDA's ceiling there, and the
    // launch of n_head x rows the kernel keeps is the whole point
    test_layer(64, 4, false, 100000, 1, true, 128, 131072);
    test_layer(64, 8, true, 100000, 1, true, 128, 128);
    test_graph();
    std::printf(fails ? "FAIL %d cases\n" : "ok  all cases passed\n", fails);
    return fails ? 1 : 0;
}
