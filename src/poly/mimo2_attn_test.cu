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

// MiMo's own numbers, kept small where the test does not need the whole model
constexpr int kDimK = 192, kDimV = 128, kHeads = 8, kRot = 64;
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
void test_layer(int n_kv, bool swa, long long pos, int rows, bool with_at, int window = 64) {
    const int kWindow = window;
    const int row = kDimK * (kHeads + n_kv) + kDimV * n_kv;
    const int ring = swa ? kWindow + 8 : 0;
    std::vector<float> src(size_t(rows) * row);
    for (float& x : src) x = frand();

    float* dsrc = upload(src);
    std::vector<float> dq(size_t(rows) * kHeads * kDimK, -777.f), dk(size_t(rows) * n_kv * kDimK, -777.f),
        dv(size_t(rows) * n_kv * kDimV, -777.f);
    float *dq_d = nullptr, *dk_d = nullptr, *dv_d = nullptr;
    cudaMalloc(&dq_d, dq.size() * sizeof(float));
    cudaMalloc(&dk_d, dk.size() * sizeof(float));
    cudaMalloc(&dv_d, dv.size() * sizeof(float));

    long long* at_d = nullptr;
    if (with_at) {
        cudaMalloc(&at_d, sizeof(long long));
        cudaMemcpy(at_d, &pos, sizeof(long long), cudaMemcpyHostToDevice);
        mimo2_rope_split(dsrc, row, rows, dq_d, dk_d, dv_d, kDimK, kDimV, kHeads, n_kv, kRot,
                         swa ? kThetaSwa : kThetaGlobal, 0, nullptr, at_d);
    } else {
        mimo2_rope_split(dsrc, row, rows, dq_d, dk_d, dv_d, kDimK, kDimV, kHeads, n_kv, kRot,
                         swa ? kThetaSwa : kThetaGlobal, pos, nullptr, nullptr);
    }
    dq = download(dq_d, dq.size());
    dk = download(dk_d, dk.size());
    dv = download(dv_d, dv.size());

    // the host split and turn, in double over the same floats
    double most = 0.;
    std::vector<float> y;
    for (int t = 0; t < rows; ++t) {
        const float* src_row = src.data() + size_t(t) * row;
        for (int h = 0; h < kHeads; ++h) {
            ref_rope(src_row + size_t(h) * kDimK, kDimK, kRot, swa ? kThetaSwa : kThetaGlobal, pos + t, y);
            for (int i = 0; i < kDimK; ++i) {
                const double d = std::fabs(y[i] - dq[size_t(t) * kHeads * kDimK + size_t(h) * kDimK + i]);
                if (d == most && most > 3e-5)
                    std::printf("  first diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, y[i],
                                dq[size_t(t) * kHeads * kDimK + size_t(h) * kDimK + i]);
                most = std::max(most, d);
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            ref_rope(src_row + size_t(kHeads + h) * kDimK, kDimK, kRot, swa ? kThetaSwa : kThetaGlobal, pos + t, y);
            for (int i = 0; i < kDimK; ++i) {
                const double d = std::fabs(y[i] - dk[size_t(t) * n_kv * kDimK + size_t(h) * kDimK + i]);
                if (d > 3e-5)
                    std::printf("  k diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, y[i],
                                dk[size_t(t) * n_kv * kDimK + size_t(h) * kDimK + i]);
                most = std::max(most, d);
            }
            for (int i = 0; i < kDimV; ++i) {   // the rows v is packed dim_v a head, after the keys dim_k a head
                const double want = src_row[size_t(kHeads + n_kv) * kDimK + size_t(h) * kDimV + i];
                const double d = std::fabs(want - dv[size_t(t) * n_kv * kDimV + size_t(h) * kDimV + i]);
                if (d > 3e-5)
                    std::printf("  v diff: t %d h %d i %d want %.6g got %.6g\n", t, h, i, want,
                                dv[size_t(t) * n_kv * kDimV + size_t(h) * kDimV + i]);
                most = std::max(most, d);
            }
        }
    }
    char name[160];
    std::snprintf(name, sizeof(name), "rope_split n_kv %d %s win %d pos %lld rows %d%s", n_kv, swa ? "swa" : "global",
                  swa ? window : 0, pos, rows, with_at ? " (pos on card)" : "");
    report(name, most, 3e-5);

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

    std::vector<float> sinks(kHeads);
    if (swa) for (float& x : sinks) x = frand() * 3.f;
    uint16_t* dkc = upload(kcache);
    uint16_t* dvc = upload(vcache);
    float* dsinks = swa ? upload(sinks) : nullptr;
    float* dout_d = nullptr;
    cudaMalloc(&dout_d, size_t(rows) * kHeads * kDimV * sizeof(float));
    if (with_at) {
        cudaMemcpy(at_d, &pos, sizeof(long long), cudaMemcpyHostToDevice);
        mimo2_attend(dq_d, (uint16_t*) dkc, (uint16_t*) dvc, 0, rows, dout_d, kDimK, kDimV, kHeads, n_kv,
                     1.f / std::sqrt((float) kDimK), nullptr, swa ? kWindow : 0, ring, at_d,
                     std::max(rows, 8), swa ? dsinks : nullptr);
    } else {
        mimo2_attend(dq_d, (uint16_t*) dkc, (uint16_t*) dvc, pos, rows, dout_d, kDimK, kDimV, kHeads, n_kv,
                     1.f / std::sqrt((float) kDimK), nullptr, swa ? kWindow : 0, ring, nullptr, 0,
                     swa ? dsinks : nullptr);
    }
    const std::vector<float> got = download(dout_d, size_t(rows) * kHeads * kDimV);

    const double scale = 1.0 / std::sqrt((double) kDimK);
    most = 0.;
    bool nan = false;
    for (int t = 0; t < rows; ++t) {
        const long long p = pos + t;
        const long long lo = swa && p >= kWindow ? p - kWindow + 1 : 0;
        for (int h = 0; h < kHeads; ++h) {
            const int kvh = h / (kHeads / n_kv);
            const float* q = qh.data() + (size_t(t) * kHeads + h) * kDimK;
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
                const double gotv = got[size_t(t) * kHeads * kDimV + size_t(h) * kDimV + i];
                if (std::isnan(gotv) || std::isinf(gotv)) nan = true;
                most = std::max(most, std::fabs(acc - gotv));
            }
        }
    }
    // a NaN must reach report(): std::max keeps its first argument against a NaN, so without this the
    // difference stays a number, report() calls it ok, and a NaN output exits 0 with one printed line
    if (nan) most = std::numeric_limits<double>::quiet_NaN();
    std::snprintf(name, sizeof(name), "attend   n_kv %d %s win %d pos %lld rows %d%s", n_kv, swa ? "swa" : "global",
                  swa ? window : 0, pos, rows, with_at ? " (pos on card)" : "");
    report(name, most, 2e-3);

    cudaFree(dsrc); cudaFree(dq_d); cudaFree(dk_d); cudaFree(dv_d); cudaFree(dkc); cudaFree(dvc);
    cudaFree(dsinks); cudaFree(dout_d); cudaFree(at_d);
}

}  // namespace

int main() {
    // a global layer: 4 kv heads, no window, rows from 0; then one deep in a long conversation
    test_layer(4, false, 0, 1, false);
    test_layer(4, false, 7, 4, false);
    test_layer(4, false, 333, 1, false);
    test_layer(4, false, 333, 1, true);          // the recorded launch's position, on the card
    // a window layer: the window's edges 127/128/129 (kWindow 64 here), the ring's wrap and a window of guesses
    test_layer(8, true, 62, 1, false);
    test_layer(8, true, 63, 1, false);
    test_layer(8, true, 64, 1, false);           // the first key hidden
    test_layer(8, true, 65, 3, false);           // tokens of one call see each other's rows
    test_layer(8, true, 71, 1, false);           // the ring has wrapped once (ring = window + 8)
    test_layer(8, true, 72, 8, false);
    test_layer(8, true, 150, 2, true);           // wrapped again, position on the card
    test_layer(4, true, 140, 1, false);
    // the model's own window of 128 and its exact edges 127 / 128 / 129, and a ring that has wrapped (ring 136)
    test_layer(8, true, 126, 1, false, 128);
    test_layer(8, true, 127, 1, false, 128);
    test_layer(8, true, 128, 1, false, 128);
    test_layer(8, true, 129, 1, false, 128);
    test_layer(8, true, 260, 2, false, 128);
    test_layer(8, true, 260, 8, true, 128);
    std::printf(fails ? "FAIL %d cases\n" : "ok  all cases passed\n", fails);
    return fails ? 1 : 0;
}
