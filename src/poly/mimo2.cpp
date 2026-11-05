// src/poly/mimo2.cpp - the layers of MiMo-V2.6-Flash.  The reference for the math is llama.cpp's mimo2 graph
// (src/models/mimo2.cpp), and the geometry this file checks is the MOPD/RL Q2_K GGUF's own metadata.
#include "mimo2.hpp"
#include "mimo2_attn.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void Mimo2::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
}

void Mimo2::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };

    n_head_ = shape_i("attention.head_count");
    dim_k_ = shape_i("attention.key_length");
    dim_v_ = shape_i("attention.value_length");
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    if (!(dim_k_ > 0 && dim_v_ > 0 && dim_k_ % 32 == 0 && dim_v_ % 32 == 0 && dim_k_ <= 256 && dim_v_ <= 256))
        throw std::runtime_error("unsupported model: MiMo's key and value widths must be multiples of 32 of at most "
                                 "256: this model gives " + std::to_string(dim_k_) + " and " + std::to_string(dim_v_));
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : dim_k_;
    need(n_rot_ % 2 == 0 && n_rot_ <= dim_k_, "bad rope.dimension_count");

    const MetaValue& pattern = must("attention.sliding_window_pattern");
    const MetaValue& heads_kv = must("attention.head_count_kv");
    swa_.assign((size_t) n_layer_, 0);
    kv_.assign((size_t) n_layer_, 0);
    n_kv_max_ = 0;
    for (int il = 0; il < n_layer_; ++il) {
        swa_[(size_t) il] = of_layer(pattern, il) != 0 ? 1 : 0;
        const int h = (int) of_layer(heads_kv, il);
        need(h > 0 && n_head_ % h == 0, "a layer's key heads do not divide its query heads");
        kv_[(size_t) il] = h;
        n_kv_max_ = std::max(n_kv_max_, h);
    }
    // the block that guesses ahead, when it comes from another file: its geometry is that file's, one layer
    // after the model's (the MTP sidecars give its pattern and heads over all 51 layers of their own count)
    if (block_ && block_ != gg_.get()) {
        const MetaValue* bp = meta_in(*block_, "attention.sliding_window_pattern");
        const MetaValue* bh = meta_in(*block_, "attention.head_count_kv");
        need(bp && bh && bp->type == MetaType::ARRAY && bh->type == MetaType::ARRAY,
             "the draft model gives no per-layer window pattern or key heads");
        swa_.push_back(of_layer(*bp, n_layer_) != 0 ? 1 : 0);
        const int h = (int) of_layer(*bh, n_layer_);
        need(h > 0 && n_head_ % h == 0, "the draft layer's key heads do not divide its query heads");
        kv_.push_back(h);
        n_kv_max_ = std::max(n_kv_max_, h);
    }
    qkv_wide_ = dim_k_ * (n_head_ + n_kv_max_) + dim_v_ * n_kv_max_;

    window_ = shape_i("attention.sliding_window");
    need(window_ > 0, "bad attention.sliding_window");
    const MetaValue* vs = find("attention.value_scale");
    value_scale_ = vs ? (float) vs->num() : 1.f;
    need(value_scale_ > 0.f && std::isfinite(value_scale_), "bad attention.value_scale");

    theta_global_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);
    const MetaValue* base_swa = find("rope.freq_base_swa");
    theta_swa_ = base_swa ? std::pow((float) base_swa->num(), -2.f / (float) n_rot_) : theta_global_;

    need(shape_i("expert_gating_func") == 2, "the expert gate is not a sigmoid");
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    n_ff_ = shape_i("feed_forward_length");   // the dense first layer's FFN
    need(n_ff_ > 0, "bad feed_forward_length");

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Mimo2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    B.swa = swa_[(size_t) il] != 0;
    B.n_kv = kv_[(size_t) il];
    B.qkv_rows = dim_k_ * (n_head_ + B.n_kv) + dim_v_ * B.n_kv;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded

    const TensorInfo* qkv = tensor(il, "attn_qkv.weight");
    if (!qkv || qkv->shape.size() != 2 || (int64_t) qkv->shape[0] != n_embd_ ||
        (int64_t) qkv->shape[1] != B.qkv_rows)
        throw std::runtime_error(blk(il, "attn_qkv.weight") + " is " + (qkv ? shape_of(*qkv) : std::string("missing")) +
                                 ", expected [" + std::to_string(n_embd_) + ", " + std::to_string(B.qkv_rows) + "]");
    B.wqkv = mat("attn_qkv.weight", n_embd_, B.qkv_rows);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * dim_v_, n_embd_);
    if (B.swa) {
        const TensorInfo* s = tensor(il, "attn_sinks.weight");
        if (!s || s->type != kTypeF32 || (int64_t) s->elements() != n_head_)
            throw std::runtime_error(blk(il, "attn_sinks.weight") + " is " +
                                     (s ? shape_of(*s) : std::string("missing")) + ", expected F32 [" +
                                     std::to_string(n_head_) + "] on a window layer");
        B.sinks = f32("attn_sinks.weight", n_head_);
    } else if (tensor(il, "attn_sinks.weight"))
        throw std::runtime_error(blk(il, "attn_sinks.weight") + ": a sink on a layer that attends the whole "
                                 "conversation, which the mimo2 reference does not read");

    if (!tensor(il, "ffn_gate_exps.weight") || il == n_layer_) {   // the reference reads the dense FFN where
        // there are no experts - and the block that guesses ahead must be such a dense layer, which is what
        // draft_step() runs; a block with routed experts is not one this draft path runs.
        if (il == n_layer_ && !tensor(il, "ffn_gate.weight"))
            throw std::runtime_error("the block that guesses ahead is not a dense mimo2 layer: no " +
                                     blk(n_layer_, "ffn_gate.weight") + "; --no-draft loads the model without it");
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, false);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (!b || b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + (b ? shape_of(*b) : std::string("missing")) +
                                 ", expected F32 [" + std::to_string(n_expert_) + "]");
    home(*b).payload(*b);
    PA_MEMCPY(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
}

int Mimo2::widest() const { return n_head_ * std::max(dim_k_, dim_v_); }

// The caches: a global layer keeps a row a position, a window layer a ring of the rows a call can read.  The
// two sides differ in width - dim_k values a key head, dim_v a value head - so they are sized apart.
void Mimo2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t rings = 0;   // a window layer's rows are a ring, whatever the sequence's length
    for (int il = 0; il < (int) blocks_.size(); ++il)
        if (runs(il) && blocks_[(size_t) il].swa)
            rings += (size_t) ring_ * blocks_[(size_t) il].n_kv * (dim_k_ + dim_v_);
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            B.keys = kv;
            kv += (size_t) ring_ * B.n_kv * dim_k_;
            B.vals = kv;
            kv += (size_t) ring_ * B.n_kv * dim_v_;
        } else {
            B.keys = (uint16_t*) kv_alloc((size_t) B.n_kv * dim_k_ * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc((size_t) B.n_kv * dim_v_ * sizeof(uint16_t), 1);
        }
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Mimo2::assign_scratch() {
    const size_t n = (size_t) rows_;
    qkv_ = fbuf(n * qkv_wide_);
    q_ = fbuf(n * n_head_ * dim_k_);
    k_ = fbuf(n * n_kv_max_ * dim_k_);
    v_ = fbuf(n * n_kv_max_ * dim_v_);
    att_ = fbuf(n * n_head_ * dim_v_);
}

// A sequence's rows are the global layers'; a window layer's are part of its state (save_layer()).
size_t Mimo2::card_row_bytes() const {
    size_t bytes = 0;
    for (const Block& B : blocks_)
        if (B.keys && !B.swa) bytes += (size_t) B.n_kv * (dim_k_ + dim_v_) * sizeof(uint16_t);
    return bytes;
}

void Mimo2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        if (!B.keys || B.swa || n == 0) continue;
        const size_t kbytes = (size_t) n * B.n_kv * dim_k_ * sizeof(uint16_t);
        const size_t vbytes = (size_t) n * B.n_kv * dim_v_ * sizeof(uint16_t);
        ck(cudaMemcpy(dst, (const uint8_t*) B.keys + (size_t) first * B.n_kv * dim_k_ * sizeof(uint16_t), kbytes,
                      cudaMemcpyDeviceToHost),
           "save rows");
        dst += kbytes;
        ck(cudaMemcpy(dst, (const uint8_t*) B.vals + (size_t) first * B.n_kv * dim_v_ * sizeof(uint16_t), vbytes,
                      cudaMemcpyDeviceToHost),
           "save rows");
        dst += vbytes;
    }
}

void Mimo2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        if (!B.keys || B.swa || n == 0) continue;
        const size_t krow = (size_t) B.n_kv * dim_k_ * sizeof(uint16_t);
        const size_t vrow = (size_t) B.n_kv * dim_v_ * sizeof(uint16_t);
        const size_t kbytes = (size_t) n * krow, vbytes = (size_t) n * vrow;
        ck(cudaMemcpy((uint8_t*) B.keys + (size_t) first * krow, src + (size_t) (first - from) * krow, kbytes,
                      cudaMemcpyHostToDevice),
           "load rows");
        src += (size_t) count * krow;
        ck(cudaMemcpy((uint8_t*) B.vals + (size_t) first * vrow, src + (size_t) (first - from) * vrow, vbytes,
                      cudaMemcpyHostToDevice),
           "load rows");
        src += (size_t) count * vrow;
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Mimo2::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends.
size_t Mimo2::extra_bytes() const {
    size_t bytes = 0;
    for (const Block& B : blocks_)
        if (B.swa) bytes += (size_t) keep_ * B.n_kv * (dim_k_ + dim_v_) * sizeof(uint16_t);
    return bytes;
}

// A window layer's rows, behind the window layers before it (there is no recurrent state).
void Mimo2::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    const Block& B = blocks_[(size_t) il];
    const size_t krow = (size_t) B.n_kv * dim_k_ * sizeof(uint16_t);
    const size_t vrow = (size_t) B.n_kv * dim_v_ * sizeof(uint16_t);
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa) dst += (size_t) keep_ * (krow + vrow);
    const uint16_t* cache[2] = {B.keys, B.vals};
    const size_t row[2] = {krow, vrow};
    for (int side = 0; side < 2; ++side) {
        const size_t side_bytes = (size_t) keep_ * row[side];
        std::memset(dst, 0, side_bytes);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * row[side], (const uint8_t*) cache[side] + at * row[side], m * row[side],
                          cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += side_bytes;
    }
}

void Mimo2::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        const size_t krow = (size_t) B.n_kv * dim_k_ * sizeof(uint16_t);
        const size_t vrow = (size_t) B.n_kv * dim_v_ * sizeof(uint16_t);
        const size_t row[2] = {krow, vrow};
        uint16_t* cache[2] = {B.keys, B.vals};
        for (int side = 0; side < 2; ++side) {
            const size_t side_bytes = (size_t) keep_ * row[side];
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy((uint8_t*) cache[side] + at * row[side], src + slot * row[side], m * row[side],
                              cudaMemcpyHostToDevice),
                   "load state");
            });
            src += side_bytes;
        }
    }
}

// The recordings of a global layer's head are numbered by the rows they are sized for; a window layer's never
// reads more than its window.
int Mimo2::head_key(int il, int n) const { return swa_[(size_t) il] ? 0 : room_key(n); }

// A side of a layer's cache: a global layer's at the positions, a window layer's at their places in the ring.
// The keys' rows are n_kv * dim_k and the values' n_kv * dim_v halves.
void Mimo2::store(const float* x, const Block& B, int64_t pos, int n, bool value) {
    const int width = B.n_kv * (value ? dim_v_ : dim_k_);
    uint16_t* cache = value ? B.vals : B.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// The attention: the fused projection split and turned by the position, the keys and values into the caches,
// the scores q.k / sqrt(dim_k) with the layer's sink in the sum where it has them, and the output times the
// model's value_scale on its way to the residual.
void Mimo2::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const float theta = B.swa ? theta_swa_ : theta_global_;
    const int sized = !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n);
    lin(B.wqkv, xn_, qkv_, n, xq_);
    mimo2_rope_split(qkv_, B.qkv_rows, n, q_, k_, v_, dim_k_, dim_v_, n_head_, B.n_kv, n_rot_, theta, pos, st_,
                     at ? at + 1 : nullptr);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    const float scale = 1.f / std::sqrt((float) dim_k_);
    mimo2_attend(q_, B.keys, B.vals, pos, n, att_, dim_k_, dim_v_, n_head_, B.n_kv, scale, st_,
                 B.swa ? window_ : 0, B.swa ? ring_ : 0, at, sized, B.swa ? fp(B.sinks) : nullptr);
    lin(B.wo, att_, ao_, n);
    if (value_scale_ != 1.f) axpy(ao_, value_scale_, ao_, n * n_embd_, false, st_);
}

// step()'s rows, each of its own sequence: the projection for all of them at once, and a row at a time the
// turn (it takes one position and counts on from it), the stored keys and values and the attention, in the
// row's sequence.
void Mimo2::attn_rows(Block& B, int n) {
    const float theta = B.swa ? theta_swa_ : theta_global_;
    lin(B.wqkv, xn_, qkv_, n, xq_);
    const float scale = 1.f / std::sqrt((float) dim_k_);
    const size_t hq = (size_t) n_head_ * dim_k_, hk = (size_t) B.n_kv * dim_k_, hv = (size_t) B.n_kv * dim_v_;
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        mimo2_rope_split(qkv_ + (size_t) i * B.qkv_rows, B.qkv_rows, 1, q_ + i * hq, k_ + i * hk, v_ + i * hv, dim_k_,
                         dim_v_, n_head_, B.n_kv, n_rot_, theta, 0, st_, ai + 1);
        uint16_t* kc = (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vc = (uint16_t*) seq_buf(r.seq, B.vals);
        store_f16_at(k_ + i * hk, kc, (int) hk, ai, 1, ring, st_);
        store_f16_at(v_ + i * hv, vc, (int) hv, ai, 1, ring, st_);
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
        pos_ = keep;
        mimo2_attend(q_ + i * hq, kc, vc, r.pos, 1, att_ + i * (size_t) n_head_ * dim_v_, dim_k_, dim_v_, n_head_,
                     B.n_kv, scale, st_, window, ring, ai, sized, B.swa ? fp(B.sinks) : nullptr);
    }
    lin(B.wo, att_, ao_, n);
    if (value_scale_ != 1.f) axpy(ao_, value_scale_, ao_, n * n_embd_, false, st_);
}

// The block's parts: the shared ones - and, for this model, the norm its own layer carries.  The reference graph
// answers for the head what the block has, in its order: nextn.shared_head_norm, the block's layer_output_norm,
// the model's output_norm - and a MiMo MTP file gives every block a layer_output_norm while the main model has
// none at all, so the block's norm is the one its guesses were trained with.  Other architectures keep the
// shared choice: only this file knows its block.
void Mimo2::load_drafter() {
    Engine::load_drafter();
    const TensorInfo* n = tensor(n_layer_, "layer_output_norm.weight");
    if (n) d_head_ = upload(n, blk(n_layer_, "layer_output_norm.weight"), n_embd_, 1, kF32);
}

// The block that guesses ahead: its projection, its turn and its caches, with the positions from the host - the
// guess runs outside the recorded launches, as every layer's store() already takes the position as a number.
void Mimo2::block_qkv(Block& B, int64_t pos, int n) {
    const float theta = B.swa ? theta_swa_ : theta_global_;
    const bool was_recording = recording_;   // read()'s draft turn runs between recorded rows: a guess's rows
    recording_ = false;                      // stand where the host says, never where a recorded row would go
    lin(B.wqkv, xn_, qkv_, n, xq_);
    mimo2_rope_split(qkv_, B.qkv_rows, n, q_, k_, v_, dim_k_, dim_v_, n_head_, B.n_kv, n_rot_, theta, pos, st_,
                     nullptr);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    recording_ = was_recording;
}

// The rows the real tokens leave in the block's ring: the block reads nothing of them but the caches.
void Mimo2::block_rows(int64_t pos, int n) { block_qkv(blocks_[(size_t) n_layer_], pos, n); }

// A guess's attention: its own row in the ring, then the window it attends, the sink among it, the output.
void Mimo2::block_attend(int64_t pos, int n) {
    Block& B = blocks_[(size_t) n_layer_];
    block_qkv(B, pos, n);
    const float scale = 1.f / std::sqrt((float) dim_k_);
    mimo2_attend(q_, B.keys, B.vals, pos, n, att_, dim_k_, dim_v_, n_head_, B.n_kv, scale, st_,
                 B.swa ? window_ : 0, B.swa ? ring_ : 0, nullptr, 0, B.swa ? fp(B.sinks) : nullptr);
    lin(B.wo, att_, ao_, n);
    if (value_scale_ != 1.f) axpy(ao_, value_scale_, ao_, n * n_embd_, false, st_);
}

// One token through the MTP block: its embedding and the model's hidden state, each turned by its own norm and
// joined by the projection, then the block's dense window layer as the model's own layers run, then the
// model's head - the block has no head of its own, and its file's token table and output matrix go unread.
void Mimo2::draft_step(int32_t token, int i) {
    Layer& L = layers_[(size_t) n_layer_];
    Block& B = blocks_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, pos_ + i, i == 0 ? hn_ : own, 1);
    block_attend(pos_ + i, 1);
    axpy(ao_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, 1, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_, 1, st_);
    dense_ffn(B, n_layer_, 1);
    axpy(fo_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(d_head_), own, n_embd_, 1, rms_eps_, st_);
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    lin(head, own, logits_, 1);
    finish_logits(logits_, 1);
}

// The dense first layer's FFN: a SwiGLU of 16384 hidden values.
void Mimo2::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (the dense layer: to its end).  A prompt batch's MoE (moe_rows)
// does its own routing.  A norm and its quantization are one launch, and so are the residual's sum and the
// next norm; the attention arrives already scaled by the model's value_scale.
void Mimo2::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void Mimo2::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Mimo2::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights
// from the bare gates, made to sum to one.
void Mimo2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Mimo2::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// No expert is shared: the routed experts' sum starts from nothing.
void Mimo2::moe_shared(const Layer&, int, int n) {
    ck(PA_ZERO(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

}  // namespace strata::poly
