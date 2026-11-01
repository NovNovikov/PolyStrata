#include "runtime.hpp"
#include "resident_dma.hpp"
#include "exchange_transfer.hpp"
#include "no_disk.hpp"

namespace strata::poly {
void Engine::exchange_available() const {
    if (!exchange_error_.empty()) throw std::runtime_error(exchange_error_);
}

void Engine::init_exchange() {
    if (!exchange_until_.empty()) return;
    recent_.assign(use_.size(), 0.f);
    coming_.assign(use_.size(), 0);
    exchange_until_.assign(use_.size(), 0);
    cudaStream_t stream = nullptr;
    ck(PA_CREATE_STREAM(&stream, cudaStreamNonBlocking), "exchange stream");
    swap_st_ = stream;
    std::fprintf(stderr, "strata-poly: expert exchange enabled: main_gpu=%d, staging copies disabled, "
                         "temporary_bytes=%zu inside staging, max=%d per eval, cooldown=%d evaluated rows, "
                         "budget=%.1f%% of preceding eval; no extra weight arena\n",
                 full_cached_, slot_max_, opt_.exchange_max, opt_.exchange_cooldown, opt_.exchange_budget_pct);
}

void Engine::exchange_pair(int il, int hot, int cold, int fault, bool forced_verify) {
    audit::Scope audit_layer(nullptr, il);
    exchange_available();
    if (il < 0 || il >= (int) layers_.size() || hot < 0 || hot >= n_expert_ || cold < 0 || cold >= n_expert_)
        throw std::invalid_argument("exchange: pair outside expert table");
    Layer& l = layers_[(size_t) il];
    const auto h = l.resident_host.at((size_t) hot);
    uint8_t* const v = l.slot.at((size_t) cold);
    const ResidentLayout lay = resident_layout(l.gu_bytes, l.u_bytes, l.d_bytes);
    const size_t raw = lay.raw;
    if (!expert_dma_ || !stage_ || !swap_st_ || !h.gate || l.slot[(size_t) hot] || !v ||
        l.resident_host[(size_t) cold].gate || raw > stage_bytes_ || l.slot_bytes < raw ||
        h.up != h.gate + lay.up || h.down != h.gate + lay.down ||
        v < exp_ || (size_t) (v - exp_) > exp_at_ || l.slot_bytes > exp_at_ - (size_t) (v - exp_))
        throw std::logic_error("exchange: invalid ownership, layout or slot capacity");
    auto* r = const_cast<uint8_t*>(h.gate);
    const size_t at = (size_t) (h.gate - expert_dma_->base());
    const bool verify = forced_verify || fault || std::getenv("STRATA_VERIFY_EXCHANGE");
    const auto started = Clock::now();
    auto tick = Clock::now();
    // Workers::run() returns only after every CPU reader finishes. All GPU users of the shared
    // staging/source pages finish here, before any payload can be overwritten.
    ck(PA_WAIT_STREAM(cs(st_)), "exchange readers");
    if (up_) ck(PA_WAIT_STREAM(cs(up_)), "exchange prefill DMA");
    ck(PA_WAIT_STREAM(cs(swap_st_)), "exchange prior transaction");
    exchange_stats_.wait_ms += ms_since(tick);
    tick = Clock::now();
    const DmaCost before = expert_dma_->cost();
    expert_dma_->retain({{at, raw}});
    exchange_stats_.registration_ms += ms_since(tick);
    if (stage_trace_) add_cost(stage_stats_.reg_decode, expert_dma_->cost(), before);
    // Poison before touching the only weight copies. Unexpected exceptions (including a failed
    // end event after successful DMA) must not allow a later request to use unpublished owners.
    exchange_error_ = "expert exchange did not publish consistent owners; engine disabled";
    expert_dma_->begin(cs(swap_st_));
    tick = Clock::now();
    try {
        exchange_transfer(v, r, stage_, raw, cs(swap_st_), verify, fault);
    } catch (const ExchangeFailure& e) {
        ++exchange_stats_.failed;
        if (e.restored) ++exchange_stats_.restored;
        else exchange_error_ = std::string("exchange cannot restore weights; engine disabled: ") + e.what();
        expert_dma_->end(cs(swap_st_));
        if (e.restored) exchange_error_.clear();
        exchange_stats_.copy_ms += ms_since(tick);
        exchange_stats_.wall_ms += ms_since(started);
        throw;
    }
    expert_dma_->end(cs(swap_st_));
    exchange_stats_.copy_ms += ms_since(tick);
    // The mixed expert path reads the host slot table on every call. GPU-only replay reads
    // d_slots_; invalidate its host mirror so card_moe() refreshes it before any later replay.
    l.resident_host[(size_t) cold] = {r, r + lay.up, r + lay.down};
    l.resident_host[(size_t) hot] = {};
    l.slot[(size_t) cold] = nullptr;
    l.slot[(size_t) hot] = v;
    h_slots_.clear();
    staged_il_ = -1;
    const size_t hi = (size_t) il * n_expert_ + hot, ci = (size_t) il * n_expert_ + cold;
    exchange_until_[hi] = exchange_until_[ci] = exchange_clock_ + opt_.exchange_cooldown;
    exchange_error_.clear(); // both payloads and every published owner are now consistent
    audit::exchange(il, hot, cold, raw);
    ++exchange_stats_.completed;
    ++stats_.swapped_experts; // the promoted RAM expert arrived in its GPU slot
    exchange_stats_.verified += verify;
    exchange_stats_.h2d += raw;
    exchange_stats_.d2h += raw;
    exchange_stats_.d2d += raw;
    exchange_stats_.wall_ms += ms_since(started);
    if (std::getenv("STRATA_TRACE_EXCHANGE"))
        std::fprintf(stderr, "strata-poly: exchanged layer=%d hot=%d cold=%d raw=%zu "
                             "hot_recent=%.3f cold_recent=%.3f verified=%d; GPU<->RAM owners committed\n",
                     il, hot, cold, raw, recent_[hi], recent_[ci], verify ? 1 : 0);
}

void Engine::follow_exchange(int written) {
    exchange_available();
    if (written <= 0 || freeze_experts_ || !opt_.expert_swaps || !expert_dma_ || !full_cached_) return;
    exchange_clock_ += written;
    const float keep = std::exp2(-(float) written / (float) kExchangeHalf);
    for (float& r : recent_) r *= keep;
    const auto started = Clock::now();
    std::vector<ExchangeExpert> experts;
    std::vector<ExchangeLayer> geometry(layers_.size());
    experts.reserve(use_.size());
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& l = layers_[(size_t) il];
        if (!l.moe || !l.slot_bytes) continue;
        const size_t raw = resident_layout(l.gu_bytes, l.u_bytes, l.d_bytes).raw;
        geometry[(size_t) il] = {raw, l.slot_bytes, raw};
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            ExchangeExpert x;
            x.layer = il; x.expert = e;
            x.home = l.resident_host[(size_t) e].gate ? ExchangeHome::ram : ExchangeHome::vram;
            x.payload_bytes = raw; x.slot_bytes = l.slot_bytes; x.recent = recent_[i];
            x.prior = prior_.empty() ? 0 : prior_[i];
            x.locked = exchange_clock_ < exchange_until_[i];
            experts.push_back(x);
        }
    }
    const auto plan = plan_exchanges(experts, geometry, (size_t) std::min(opt_.exchange_max, opt_.expert_swaps));
    exchange_stats_.plan_ms += ms_since(started);
    ++exchange_stats_.calls;
    exchange_stats_.proposed += plan.pairs.size();
    for (const auto reason : plan.skipped)
        if (reason != ExchangeSkip::none && reason != ExchangeSkip::selected)
            ++exchange_stats_.skipped[(size_t) reason];
    const double budget = exchange_eval_ms_ * opt_.exchange_budget_pct / 100.;
    // Soft budget: complete the first transaction, then stop additional work at this boundary.
    // A cold registration or diagnostic checksum must never disable adaptation permanently.
    const auto work = Clock::now();
    size_t completed = 0;
    for (const auto& p : plan.pairs) {
        if (completed && ms_since(work) >= budget) { exchange_stats_.budget_skips += plan.pairs.size() - completed; break; }
        const auto& h = experts[(size_t) p.hot];
        const auto& c = experts[(size_t) p.cold];
        exchange_pair(h.layer, h.expert, c.expert);
        ++completed;
    }
}

void Engine::exchange_test(int fault) {
    if (!opt_.no_disk_expert_exchange) throw std::runtime_error("EXCHANGE requires --no-disk-expert-exchange");
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const auto& l = layers_[(size_t) il];
        if (!l.moe || !l.slot_bytes) continue;
        int h = -1, c = -1;
        for (int e = 0; e < n_expert_; ++e) {
            if (l.resident_host[(size_t) e].gate && h < 0) h = e;
            if (l.slot[(size_t) e] && c < 0) c = e;
        }
        if (h >= 0 && c >= 0) { ++exchange_stats_.proposed; exchange_pair(il, h, c, fault, true); return; }
    }
    throw std::runtime_error("EXCHANGE: no same-layer pair exists");
}

void Engine::print_exchange_stats(const ExchangeStats& b) const {
    if (!opt_.no_disk_expert_exchange) return;
    const auto& a = exchange_stats_;
    std::fprintf(stderr, "strata-poly: exchange: planned=%llu completed=%llu verified=%llu failed=%llu restored=%llu "
                         "committed_H2D=%llu committed_D2H=%llu committed_D2D=%llu bytes; plan_ms=%.3f registration_ms=%.3f "
                         "copy_ms=%.3f reader_wait_ms=%.3f transaction_ms=%.3f budget_skips=%llu; "
                         "no_cold=%llu below_threshold=%llu locked=%llu taken=%llu no_room=%llu batch_limit=%llu; "
                         "registered=%zu peak_registered=%zu reg_added_total=%llu reg_removed_total=%llu bytes\n",
                 (unsigned long long) (a.proposed-b.proposed), (unsigned long long) (a.completed-b.completed),
                 (unsigned long long) (a.verified-b.verified), (unsigned long long) (a.failed-b.failed),
                 (unsigned long long) (a.restored-b.restored), (unsigned long long) (a.h2d-b.h2d),
                 (unsigned long long) (a.d2h-b.d2h), (unsigned long long) (a.d2d-b.d2d),
                 a.plan_ms-b.plan_ms, a.registration_ms-b.registration_ms, a.copy_ms-b.copy_ms,
                 a.wait_ms-b.wait_ms, a.wall_ms-b.wall_ms, (unsigned long long) (a.budget_skips-b.budget_skips),
                 (unsigned long long) (a.skipped[1]-b.skipped[1]), (unsigned long long) (a.skipped[2]-b.skipped[2]),
                 (unsigned long long) (a.skipped[3]-b.skipped[3]), (unsigned long long) (a.skipped[4]-b.skipped[4]),
                 (unsigned long long) (a.skipped[5]-b.skipped[5]), (unsigned long long) (a.skipped[6]-b.skipped[6]),
                 expert_dma_ ? expert_dma_->bytes() : 0, expert_dma_ ? expert_dma_->peak() : 0,
                 (unsigned long long) (expert_dma_ ? expert_dma_->registered_bytes() : 0),
                 (unsigned long long) (expert_dma_ ? expert_dma_->unregistered_bytes() : 0));
}
} // namespace strata::poly
