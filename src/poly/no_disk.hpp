// Pure placement arithmetic for --no-disk; no GPU, files or allocation side effects.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {
struct ResidentExpert {
    int layer = 0, expert = 0;
    uint64_t ram = 0, vram = 0;
    double use = 0;
};
struct ResidentPlan {
    std::vector<bool> on_card;
    uint64_t ram = 0, vram = 0;
    size_t card_count = 0;
};
inline uint64_t resident_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::runtime_error("no-disk: byte count overflow");
    return a + b;
}
// On Windows an available-physical snapshot excludes pages the OS can reclaim from other processes.
// Try the actual allocation and full lock instead; only an explicit host limit is a hard RAM budget.
// Other platforms retain the existing available-memory admission check. UINT64_MAX means no explicit cap.
inline uint64_t resident_ram_budget(uint64_t available, uint64_t held, uint64_t future_rows,
                                    uint64_t limit, bool attempt_lock) {
    uint64_t ram = attempt_lock ? UINT64_MAX : (available > future_rows ? available - future_rows : 0);
    if (limit) {
        const uint64_t used = resident_add(held, future_rows);
        if (used > limit) throw std::runtime_error("no-disk: host weights and buffers need " +
            std::to_string(used) + " RAM bytes before experts; --ram-limit-mib provides " + std::to_string(limit));
        ram = std::min(ram, limit - used);
    }
    return ram;
}
inline ResidentPlan plan_resident(const std::vector<ResidentExpert>& experts, uint64_t ram, uint64_t vram) {
    ResidentPlan p;
    p.on_card.assign(experts.size(), false);
    std::vector<size_t> order;
    for (size_t i = 0; i < experts.size(); ++i) {
        const auto& e = experts[i];
        if (e.ram == 0) throw std::runtime_error("no-disk: an expert has zero bytes");
        p.ram = resident_add(p.ram, e.ram);
        if (e.vram) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const auto& x = experts[a]; const auto& y = experts[b];
        const double xv = x.use / (double) x.vram, yv = y.use / (double) y.vram;
        return xv != yv ? xv > yv : x.expert < y.expert;
    });
    // The card fills first, with the experts used most per byte of VRAM; RAM holds whatever is left.
    for (const size_t i : order) {
        const auto& e = experts[i];
        if (e.vram > vram - p.vram) continue;      // one that does not fit is left in RAM, not a reason to stop
        p.on_card[i] = true;
        p.ram -= e.ram;
        p.vram += e.vram;
        ++p.card_count;
    }
    if (p.ram > ram)
        throw std::runtime_error("no-disk: model does not fit: experts need " + std::to_string(p.ram) +
            " RAM bytes after placing " + std::to_string(p.vram) + " in VRAM; RAM budget " +
            std::to_string(ram) + ", available expert VRAM " + std::to_string(vram));
    return p;
}
} // namespace strata::poly
