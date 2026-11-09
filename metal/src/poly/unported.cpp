// metal/src/poly/unported.cpp - the entry points of the shared strata-poly host code that have no Metal kernel yet.
// None of them is on a Mac family's default path: llama.cpp's flash attention (which falls back to attend() and
// attend_batch() by its own contract), Gemma 4's opt-in int8 attention caches, Clover's block guesses and coreset
// routing, and the half-sum guard of Flint's MMQ.  A call that would need a kernel says what is missing instead of
// computing something else.
#include "fattn.hpp"
#include "mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

[[noreturn]] void unported(const char* what) { throw std::invalid_argument(std::string("Metal has no ") + what); }

}  // namespace

bool fattn_fits(int, int, int, int, size_t) { return false; }
bool fdec_fits(int, int, int, int, size_t) { return false; }
bool fattn_q8(const float*, Kvq8, Kvq8, int64_t, int, float*, int, int, int, float, float*, size_t, void*) {
    return false;
}
bool fattn_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, float*, int, int, int, float, float*,
               size_t, void*) {
    return false;
}
bool fdec_q8(const float*, Kvq8, Kvq8, int64_t, int, const long long*, int, float*, int, int, int, float, float*,
             size_t, void*) {
    return false;
}
bool fdec_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, const long long*, int, float*, int, int,
              int, float, float*, size_t, void*) {
    return false;
}

void store_int8(const float*, int8_t*, uint16_t*, int, int, int64_t, int, void*) {
    unported("int8 attention cache (--kv int8 / int8k, Gemma 4)");
}
void store_int8_at(const float*, int8_t*, uint16_t*, int, int, int, const long long*, int, void*) {
    unported("int8 attention cache (--kv int8 / int8k, Gemma 4)");
}
void attend_int8(const float*, KvInt8, KvInt8, KvInt8, int64_t, int, float*, int, float*, int, int, int, float, void*,
                 int, int, bool, const long long*, int) {
    unported("int8 attention cache (--kv int8 / int8k, Gemma 4)");
}
void attend_int8_mixed(const float*, KvInt8, KvInt8, const uint16_t*, int64_t, int, float*, int, float*, int, int,
                       int, float, void*, int, int, bool, const long long*, int) {
    unported("int8 attention cache (--kv int8 / int8k, Gemma 4)");
}
void dequant_int8_to_f16(const int8_t*, const uint16_t*, uint16_t*, int64_t, void*) {
    unported("int8 attention cache (--kv int8 / int8k, Gemma 4)");
}

void conv4_block_silu(const float*, const float*, const float*, const float*, float*, int, int, void*) {
    unported("block guesses (Clover)");
}
void gdn_block(float*, float*, float*, const float*, const float*, const float*, const float*, const float*,
               const float*, const float*, const float*, float*, float*, int, int, int, int, float, void*) {
    unported("block guesses (Clover)");
}
void coreset_mask(float*, int, int, int, float, void*) { unported("coreset routing (Clover)"); }

// the guard is Flint's (Flint::Flint() turns it on), and Flint is no Mac family: false keeps it from being asked for
bool Mmq::half_sums(int) { return false; }
void guard_half_sums(float*, int64_t, int, int, float*, void*) { unported("half-sum guard (Flint)"); }
void scale_rows(float*, int64_t, int, int, const int32_t*, const float*, bool, void*) {
    unported("half-sum guard (Flint)");
}

}  // namespace strata::poly

namespace strata::kernels {

// the 32-expert launches of STRATA_SLOTS_WARP: Metal's slots kernel takes 8 cached experts a launch
bool native_mmvq_slots_wide() { return false; }

}  // namespace strata::kernels
