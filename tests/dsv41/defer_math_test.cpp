// Strata. The STRATA_DS4_DEFER late add's arithmetic (ds4_defer_math.hpp - the same functions the device
// defer_add_kernel / defer_mix_kernel run) against a CPU reference for the on-time add (hostile review 7):
// the part is rank one (the vector y with per-stream weights post), so a part that lands layers late, after
// the streams went through the mixings, must land with exactly the coefficients the on-time add would have
// become. Every value is an exact binary fraction (multiples of 1/4, mixing columns of absolute weight sum
// at most 1, y at most 1/4), so every product, sum and composition is EXACT in float32 and the kernel's
// operation order and the reference's are bit comparable. No device, no model.
#include "ds4_defer_math.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "  FAIL: %s\n", what); }
}

struct Rng {
    unsigned s = 12345u;
    int next(int m) {
        s = s * 1103515245u + 12345u;
        return (int) ((s >> 16) % (unsigned) (2 * m + 1)) - m;
    }
    float q() { return (float) next(4) / 4.0f; }             // a multiple of 1/4 in [-1, 1]
    float y() { return (float) next(4) / 16.0f; }            // a multiple of 1/16 in [-1/4, 1/4]
    int nextu(int n) { s = s * 1103515245u + 12345u; return (int) ((s >> 16) % (unsigned) n); }
    // a mixing matrix: 4 signed units of 1/4 spread over the 4*hc cells - every column's absolute sum <= 1,
    // so a stream of magnitude m stays at m mixed and 1/1024-grid values stay exact three layers deep
    void mix4(int hc, float* C) {
        for (int x = 0; x < hc * hc; ++x) C[x] = 0.0f;   // (the whole hc x hc matrix - rows past 4*hc/hc must not be stack junk)
        for (int u = 0; u < 4; ++u) C[(size_t) nextu(4 * hc)] = next(1) < 0 ? -0.25f : 0.25f;
    }
};

// the mixing a stream goes through: out_k = sum_j C[j*hc + k] * in_j (the layout the kernels use). The
// reference accumulates in double and rounds once - exact under the bounds above, so it carries no
// dependence on summation order.
static void mix_row(const float* in, const float* C, int hc, float* out) {
    for (int k = 0; k < hc; ++k) {
        double acc = 0.0;
        for (int j = 0; j < hc; ++j) acc += (double) C[(size_t) j * hc + k] * in[j];
        out[k] = (float) acc;
    }
}
static bool same(const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

int main() {
    for (const int hc : {1, 2, 4, 8}) {
        for (int seed = 0; seed < 64; ++seed) {
            Rng r;
            r.s = 0x9e3779b9u * (unsigned) (seed + 1) + (unsigned) hc * 7919u;
            float post[8], yv = r.y(), A[64], F[64], s[8], s_nop[8];
            for (int j = 0; j < hc; ++j) post[j] = r.q();
            r.mix4(hc, A);
            r.mix4(hc, F);
            // s: the streams right after the producing layer, WITH the part's post[j]*y on top; s_nop: the
            // streams had the part never been added. Built, not subtracted, so both are exact.
            for (int j = 0; j < hc; ++j) {
                s_nop[j] = r.q();
                s[j] = s_nop[j] + post[j] * yv;
            }
            float tmp[8], ontime[8], late[8];

            // 1. the plain landing (comb null): nothing mixed in between since the producing layer
            std::memcpy(late, s_nop, sizeof late);
            defer_add_elem(late, 1, yv, post, nullptr, nullptr, hc);
            check(same(late, s, hc), "a part landing on the layer after its add gains exactly post * y");

            // 2. a lone comb2 changes no branch: the kernel composes only when comb is there too
            std::memcpy(late, s_nop, sizeof late);
            defer_add_elem(late, 1, yv, post, nullptr, F, hc);
            check(same(late, s, hc), "comb2 without comb stays the plain add (the kernel's branch order)");

            // 3. one mixing in between (DEFER=2,4 one layer late): the on-time add mixed by A == the old
            // streams mixed by A and the part added with the composed weights
            mix_row(s, A, hc, ontime);          // what the on-time add would have become
            mix_row(s_nop, A, hc, late);        // what the streams really are: the same mix, no part
            defer_add_elem(late, 1, yv, post, A, nullptr, hc);
            check(same(late, ontime, hc), "one mixing since: the composed late add equals the on-time add mixed");

            // 4. two mixings (DEFER=3): through the next layer's attention (A) and FFN (F) mixing
            mix_row(s, A, hc, tmp);
            mix_row(tmp, F, hc, ontime);
            mix_row(s_nop, A, hc, tmp);
            mix_row(tmp, F, hc, late);
            defer_add_elem(late, 1, yv, post, A, F, hc);
            check(same(late, ontime, hc), "two mixings since: comb-then-comb2 lands the part exactly");

            // 5. the DEFER=4 weight remap, repeated over three layers: the carried weights land the part
            // exactly where the on-time add would have gone, layer after layer - and the remap reads its row
            // through a snapshot (in place: a k=0 write must not poison a k=1 read)
            float c[8], old[8];
            std::memcpy(c, post, sizeof c);
            std::memcpy(old, s_nop, sizeof old);
            bool all_ok = true;
            for (int layer = 0; layer < 3; ++layer) {
                const float* M = layer % 2 == 0 ? A : F;
                defer_mix_row(c, M, hc);
                float nx[8];
                mix_row(old, M, hc, nx);
                std::memcpy(old, nx, sizeof old);
                // the on-time streams, rebuilt from s through layers 0..layer
                float ot[8];
                std::memcpy(ot, s, sizeof ot);
                for (int m = 0; m <= layer; ++m) {
                    const float* MM = m % 2 == 0 ? A : F;
                    float q[8];
                    mix_row(ot, MM, hc, q);
                    std::memcpy(ot, q, sizeof ot);
                }
                float with_part[8];
                std::memcpy(with_part, old, sizeof old);
                defer_add_elem(with_part, 1, yv, c, nullptr, nullptr, hc);
                if (!same(with_part, ot, hc)) all_ok = false;
            }
            check(all_ok, "DEFER=4: weights remapped per layer land the part exactly, layer after layer");

            // 6. the remap composition equals the two-mixing add's coefficients bit for bit (the same fma
            // sequence), so a mode-4 part deferred two layers folds y exactly as a mode-3 add would
            float c2[8], zero[8] = {};
            std::memcpy(c2, post, sizeof c2);
            defer_mix_row(c2, A, hc);
            defer_mix_row(c2, F, hc);
            defer_add_elem(zero, 1, yv, post, A, F, hc);   // zero streams: the result is c_k * y
            bool coef = true;
            for (int k = 0; k < hc; ++k)
                if (!(zero[k] == c2[k] * yv)) coef = false;
            check(coef, "the per-layer remap composes to the DEFER=3 coefficients (mode 3 and 4 agree)");

            // 7. stride: the kernel hands streams `dim` apart; the add must write only its own element of
            // each stream
            {
                const int dim = 40;
                std::vector<float> h((size_t) hc * dim, 0.0f);
                mix_row(s_nop, A, hc, tmp);   // (the streams as they really stand: mixed already, no part)
                for (int k = 0; k < hc; ++k) h[(size_t) k * dim] = tmp[k];
                defer_add_elem(h.data(), dim, yv, post, A, nullptr, hc);
                mix_row(s, A, hc, ontime);
                bool ok = true;
                for (int k = 0; k < hc; ++k)
                    if (!(h[(size_t) k * dim] == ontime[k])) ok = false;
                for (size_t x = 1; x < (size_t) dim; ++x)
                    for (int k = 0; k < hc; ++k)
                        if (h[(size_t) k * dim + x] != 0.0f) ok = false;
                check(ok, "strided (device-shaped) adds write only their own element of each stream");
            }
        }
    }
    std::printf("defer_math_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
