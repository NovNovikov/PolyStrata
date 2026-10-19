#include "runtime.hpp"
#include "resident_memory.hpp"
#include "resident_dma.hpp"
#include "no_disk.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include <numeric>

namespace strata::poly {
Layer::Host Engine::host_expert(const Layer& l, size_t e) const {
    if (opt_.no_disk) {
        const auto h = l.resident_host.at(e);
        if (!h.gate || !h.up || !h.down) throw std::logic_error("no-disk: a VRAM expert was requested from RAM");
        return h;
    }
    return {l.exp_gate + e * l.gu_stride, l.exp_up + e * l.gu_stride, l.exp_down + e * l.d_bytes};
}

void Engine::pin_host_weights(void* p, size_t bytes) {
    if (opt_.no_disk && dense_) resident_->add_external(p, bytes);
}

Dev Engine::upload_resident(const TensorInfo& t, const Dev& d, size_t bytes, size_t count, Want want,
                            float (*fix)(float)) {
    const size_t rows = (size_t) d.n_out, cols = (size_t) d.n_in, in_row = bytes / rows;
    if (count != rows * cols) throw std::logic_error("no-disk: upload shape changed for " + t.name);
    const bool raw = d.type == t.type && !fix && want != kGuessMat;
    const size_t out_row = raw ? in_row : d.type == kTypeF32 ? cols * 4 : d.type == kTypeBF16 ? cols * 2
                                                      : ggml_row_size((ggml_type) d.type, (int64_t) cols);
    const size_t block = std::max<size_t>(1, (8u << 20) / std::max({in_row, out_row, cols * 4}));
    std::vector<uint8_t> source(block * in_row), converted(raw ? 0 : block * out_row);
    std::vector<float> floats(raw ? 0 : block * cols);
    const auto* traits = ggml_get_type_traits((ggml_type) t.type);
    for (size_t first = 0; first < rows; first += block) {
        const size_t n = std::min(block, rows - first), values = n * cols;
        home(t).read(t, first * in_row, source.data(), n * in_row);
        const void* src = source.data();
        if (!raw) {
            if (t.type == kTypeF32) std::memcpy(floats.data(), src, values * 4);
            else if (traits && traits->to_float) traits->to_float(src, floats.data(), (int64_t) values);
            else throw std::runtime_error("no-disk: cannot convert " + t.name);
            if (fix) for (size_t i = 0; i < values; ++i) floats[i] = fix(floats[i]);
            if (d.type == kTypeF32) src = floats.data();
            else if (d.type == kTypeBF16) {
                ggml_fp32_to_bf16_row(floats.data(), (ggml_bf16_t*) converted.data(), (int64_t) values);
                src = converted.data();
            } else {
                ggml_quantize_chunk((ggml_type) d.type, floats.data(), converted.data(), 0, (int64_t) n,
                                    (int64_t) cols, nullptr);
                src = converted.data();
            }
        }
        h2d((uint8_t*) d.p + first * out_row, src, n * out_row);
        ck(cudaStreamSynchronize(cs(st_)), "no-disk: loading a tensor");
    }
    return d;
}

void Engine::load_resident_experts() {
    resident_->lock_external();
    std::vector<ResidentExpert> experts;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& l = layers_[(size_t) il];
        if (!l.moe) continue;
        const size_t raw = 2 * l.gu_bytes + l.d_bytes;
        const bool native = opt_.gpu_experts && kernels::native_mmvq_supported((int) l.gu_type) &&
                            kernels::native_mmvq_supported((int) l.d_type);
        if (native) {
            const size_t unit = std::lcm(std::lcm(ggml_type_size((ggml_type) l.gu_type),
                                                  ggml_type_size((ggml_type) l.d_type)), size_t(8));
            l.slot_bytes = (raw + kTail + unit - 1) / unit * unit;
            slot_max_ = std::max(slot_max_, l.slot_bytes);
        }
        l.resident_host.resize((size_t) n_expert_);
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            const double uses = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            experts.push_back({il, e, raw, native ? l.slot_bytes : 0, uses});
        }
    }
    if (opt_.no_disk_vram_margin_mib < 0)
        throw std::runtime_error("--no-disk-vram-margin-mib cannot be negative");
    reserve_mib_ = opt_.vram_reserve_mib >= 0 ? std::max(opt_.vram_reserve_mib, 256) : 1024;
    const uint64_t reserve = (uint64_t) reserve_mib_ << 20;
    const uint64_t load_margin = (uint64_t) opt_.no_disk_vram_margin_mib << 20;
    const uint64_t headroom = resident_add(reserve, load_margin);
    // Whole-device deltas include driver allocations and other processes. The engine's requested bytes are
    // reported separately; a change in free memory alone does not establish spill or its owner.
    size_t free = 0, total = 0, first_free = 0, previous_free = 0;
    bool first_sample = true;
    const auto sample_vram = [&](const char* phase) {
        ck(cudaStreamSynchronize(cs(st_)), "no-disk: VRAM diagnostic fence");
        ck(cudaMemGetInfo(&free, &total), "no-disk: VRAM diagnostic");
        if (first_sample) { first_free = previous_free = free; first_sample = false; }
        uint64_t tracked = 0;
        for (const auto& allocation : dev_sizes_) tracked = resident_add(tracked, allocation.second);
        constexpr double mib = 1048576.0;
        std::fprintf(stderr, "strata-poly: no-disk VRAM[%s]: free=%.2f MiB (%llu bytes), "
                             "reserve=%llu MiB, load_margin=%llu MiB, used_since_previous=%+.2f MiB, "
                             "used_since_plan=%+.2f MiB, engine_allocated=%.2f MiB\n",
                     phase, double(free) / mib, (unsigned long long) free,
                     (unsigned long long) (reserve >> 20), (unsigned long long) (load_margin >> 20),
                     (double(previous_free) - double(free)) / mib,
                     (double(first_free) - double(free)) / mib, double(tracked) / mib);
        previous_free = free;
    };
    // Dense, scratch, full-context KV, staging and optional VRAM rows already exist.
    sample_vram("before-plan");
    uint64_t vram = free > headroom ? free - headroom : 0;
    // Future host read rows are counted now; no 1 GiB file-staging buffers or swap buffers exist in this mode.
    const uint64_t future_rows = opt_.rows_vram ? 0 : (uint64_t) read_rows_ * streams() * n_embd_ * sizeof(float);
    const auto ram_budget = [&]() {
        uint64_t available = 0;
#ifdef _WIN32
        MEMORYSTATUSEX status{}; status.dwLength = sizeof status;
        if (GlobalMemoryStatusEx(&status)) available = status.ullAvailPhys;
        constexpr bool attempt_lock = true;
#else
        constexpr bool attempt_lock = false;
        if (FILE* f = std::fopen("/proc/meminfo", "r")) {
            char line[128]; unsigned long long kb = 0;
            while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            std::fclose(f); available = kb << 10;
        }
#endif
        const uint64_t limit = (uint64_t) opt_.ram_limit_mib << 20;
        const uint64_t held = resident_add(resident_->bytes(), service_host_bytes_);
        std::fprintf(stderr, "strata-poly: no-disk RAM: policy=%s, available_now=%llu MiB, "
                             "held=%llu MiB, future_rows=%llu MiB, explicit_limit=%llu MiB (0=none)\n",
                     attempt_lock ? "allocate-and-lock" : "available-physical",
                     (unsigned long long) (available >> 20), (unsigned long long) (held >> 20),
                     (unsigned long long) (future_rows >> 20), (unsigned long long) (limit >> 20));
        return resident_ram_budget(available, held, future_rows, limit, attempt_lock);
    };
    ResidentPlan plan;
    uint8_t* host = nullptr;
    bool placement_ready = false;
    // Preparing locked RAM and DMA can change the device budget. Rebuild a smaller plan from the live
    // budget instead of passing an obsolete plan to dmalloc(). No expert payload has been read yet.
    for (int attempt = 1; attempt <= 4; ++attempt) {
        const uint64_t ram = ram_budget();
        try { plan = plan_resident(experts, ram, vram); }
        catch (const std::runtime_error&) { print_memory("placement-refused"); throw; }
        const std::string ram_text = ram == UINT64_MAX ? "allocation/lock" : std::to_string(ram >> 20) + " MiB";
        std::fprintf(stderr, "strata-poly: no-disk plan: attempt=%d, %zu experts in RAM (%llu MiB), %zu permanent in VRAM "
                         "(%llu MiB); expert budgets RAM=%s VRAM=%llu MiB; staging=%zu MiB\n",
                 attempt, experts.size() - plan.card_count, (unsigned long long) (plan.ram >> 20), plan.card_count,
                 (unsigned long long) (plan.vram >> 20), ram_text.c_str(),
                 (unsigned long long) (vram >> 20), stage_bytes_ >> 20);
        host = (uint8_t*) resident_->allocate((size_t) plan.ram, "routed experts");
        sample_vram("after-ram-arena");
        if (plan.ram && stage_) {
            // WDDM refuses registration of the whole 110+ GiB arena. A working window registers original
            // source pages only, bounded by staging plus page rounding; no second RAM arena is allocated.
            const uint64_t limit = resident_add(stage_bytes_, uint64_t(2) * n_expert_ * ResidentDma::page_size());
            if (limit > SIZE_MAX) throw std::runtime_error("no-disk: DMA window size overflow");
            expert_dma_ = std::make_unique<ResidentDma>(host, (size_t) plan.ram, (size_t) limit);
            expert_dma_->prepare({{0, std::min((size_t) plan.ram, stage_bytes_)}});
            std::fprintf(stderr, "strata-poly: no-disk DMA verified: CUDA/HIP-registered source window=%zu MiB, "
                             "limit=%zu MiB; original resident RAM, no host staging copy\n",
                     expert_dma_->bytes() >> 20, (size_t) limit >> 20);
        }
        sample_vram("after-registration");
        const uint64_t live_vram = free > headroom ? free - headroom : 0;
        if (plan.vram <= live_vram) { placement_ready = true; break; }
        std::fprintf(stderr, "strata-poly: no-disk replan: expert arena=%llu bytes exceeds live budget=%llu bytes "
                             "after RAM/DMA setup (free=%llu, reserve=%llu, load_margin=%llu); retrying before payload load\n",
                     (unsigned long long) plan.vram, (unsigned long long) live_vram, (unsigned long long) free,
                     (unsigned long long) reserve, (unsigned long long) load_margin);
        // Unregister before releasing its source. Do not hold two model-sized RAM arenas during a retry.
        expert_dma_.reset();
        if (host) resident_->release_last(host);
        host = nullptr;
        vram = std::min(vram, live_vram);   // never oscillate back to the optimistic pre-setup budget
    }
    if (!placement_ready)
        throw std::runtime_error("no-disk: VRAM budget did not stabilize after four RAM/DMA placement attempts");
    if (plan.vram) {
        exp_ = (uint8_t*) dmalloc((size_t) plan.vram, "no-disk permanent experts");
    }
    sample_vram("after-expert-allocation");
    if (plan.vram) zero_vram(exp_, (size_t) plan.vram);
    sample_vram("after-expert-zero");
    std::vector<uint8_t> buffer(8u << 20);
    const auto card_read = [&](const TensorInfo& t, size_t off, uint8_t* dst, size_t n) {
        for (size_t at = 0; at < n;) {
            const size_t m = std::min(buffer.size(), n - at);
            home(t).read(t, off + at, buffer.data(), m);
            h2d(dst + at, buffer.data(), m);
            ck(cudaStreamSynchronize(cs(st_)), "no-disk: loading a permanent expert");
            at += m;
        }
    };
    size_t host_at = 0;
    for (size_t i = 0; i < experts.size(); ++i) {
        const auto& e = experts[i];
        Layer& l = layers_[(size_t) e.layer];
        const size_t gu_off = (size_t) e.expert * l.gu_stride;
        const size_t up_off = gu_off + (l.gate_tensor == l.up_tensor ? l.gu_bytes : 0);
        const size_t down_off = (size_t) e.expert * l.d_bytes;
        if (plan.on_card[i]) {
            uint8_t* dst = exp_ + exp_at_;
            l.slot[(size_t) e.expert] = dst;
            card_read(*l.gate_tensor, gu_off, dst, l.gu_bytes);
            card_read(*l.up_tensor, up_off, dst + l.gu_bytes, l.gu_bytes);
            card_read(*l.down_tensor, down_off, dst + 2 * l.gu_bytes, l.d_bytes);
            exp_at_ += l.slot_bytes;
            ++cached_;
        } else {
            uint8_t* dst = host + host_at;
            l.resident_host[(size_t) e.expert] = {dst, dst + l.gu_bytes, dst + 2 * l.gu_bytes};
            home(*l.gate_tensor).read(*l.gate_tensor, gu_off, dst, l.gu_bytes);
            home(*l.up_tensor).read(*l.up_tensor, up_off, dst + l.gu_bytes, l.gu_bytes);
            home(*l.down_tensor).read(*l.down_tensor, down_off, dst + 2 * l.gu_bytes, l.d_bytes);
            host_at += (size_t) e.ram;
        }
        if (e.expert == n_expert_ - 1)
            std::fprintf(stderr, "strata-poly: no-disk loaded experts of layer %d\n", e.layer);
    }
    if (host_at != plan.ram || exp_at_ != plan.vram) throw std::logic_error("no-disk: placement byte mismatch");
    full_bytes_ = exp_at_; full_cached_ = cached_;
    resident_host_bytes_ += (size_t) plan.ram;
    ck(cudaStreamSynchronize(cs(st_)), "no-disk load");
    sample_vram("after-expert-copy");
    const uint64_t read = gg_->read_bytes() + (dg_ ? dg_->read_bytes() : 0);
    gg_->close_data(); gg_->release_small();
    if (dg_) { dg_->close_data(); dg_->release_small(); }
    sample_vram("after-file-close");
    if (free < reserve) {
        print_memory("load-reserve-refused");
        throw std::runtime_error("no-disk: free VRAM below requested reserve after load: free=" +
            std::to_string(free) + " bytes, reserve=" + std::to_string(reserve) +
            " bytes, load_margin=" + std::to_string(load_margin) + " bytes");
    }
    std::fprintf(stderr, "strata-poly: no-disk ready: %zu experts total, %zu in locked RAM, %d direct in VRAM; "
                         "locked=%llu MiB, read=%llu bytes, model files closed; VRAM free=%zu MiB, "
                         "host buffers=%zu MiB\n", experts.size(), experts.size() - plan.card_count, cached_,
                 (unsigned long long) (resident_->bytes() >> 20), (unsigned long long) read, free >> 20,
                 service_host_bytes_ >> 20);
}
} // namespace strata::poly
