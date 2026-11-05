// src/poly/mimo2.hpp - Xiaomi's MiMo-V2.6-Flash (llama.cpp architecture "mimo2"): MOPD and RL, 309B-A15B.
//
// One residual stream.  Every layer's attention reads one fused projection: 64 query heads of 192 key values
// over 4 (global layers) or 8 (window layers) key heads, whose keys are 192 values and values 128 - the two
// sides a shared attend() cannot index, hence mimo2_attn.hpp's kernels.  All layers turn q and k by the
// position on 64 of their 192 values, by a base of 1e7 on the global layers and 1e4 on the window ones; the
// window layers (39 of 48) attend the last 128 tokens from a ring and add a learned sink logit a head to their
// softmaxes.  The attention's output joins the residual times the model's value_scale.  The first layer's FFN
// is dense, the others' routed experts: 256 of 2048 wide, gates and ups Q2_K, downs MXFP4, picked by a sigmoid
// plus a bias and weighed by the bare sigmoids made to sum to one; no shared expert.
//
// The reference for the math is llama.cpp's mimo2 graph (src/models/mimo2.cpp).
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Mimo2 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        bool swa = false;    ///< attends a sliding window; its caches are rings
        int n_kv = 0;        ///< the layer's key heads: 4 on a global layer, 8 on a window one
        int qkv_rows = 0;    ///< the fused projection's output rows: q, k and v of this layer's heads
        Dev wqkv, wo, sinks;
        uint16_t* keys = nullptr;   ///< rows of n_kv * dim_k, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;   ///< rows of n_kv * dim_v
        Dev ffn_gate, ffn_up, ffn_down;   ///< a dense layer's FFN
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the sigmoids
        Dev probs_b_dev;
    };

    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    void dense_ffn(Block& B, int il, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<uint8_t> swa_;    ///< per layer, from the GGUF's sliding_window_pattern
    std::vector<int> kv_;         ///< per layer, from its head_count_kv

    int n_head_ = 0, dim_k_ = 0, dim_v_ = 0, n_rot_ = 0, n_kv_max_ = 0, qkv_wide_ = 0;
    float theta_global_ = 0;   ///< a global layer's rotary pair turns by position * theta_global_^pair
    float theta_swa_ = 0;      ///< and a window layer's by its own base
    float value_scale_ = 1.f;  ///< the attention's output joins the residual times this
    bool w_norm_ = true;       ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;        ///< and then scaled by this, when it is not 0
    int window_ = 0;           ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;             ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;             ///< rows of a window layer a sequence's state keeps

    // device scratch
    float *qkv_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr;
};

}  // namespace strata::poly
