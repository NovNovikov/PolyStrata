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
    // Only the prefix needed to fit RAM becomes permanent VRAM; do not fill the rest of the card.
    for (const size_t i : order) {
        if (p.ram <= ram) break;
        const auto& e = experts[i];
        if (e.vram > vram - p.vram) continue;
        p.on_card[i] = true;
        p.ram -= e.ram;
        p.vram += e.vram;
        ++p.card_count;
    }
    if (p.ram > ram)
        throw std::runtime_error("no-disk: model does not fit: experts need " + std::to_string(p.ram) +
            " RAM bytes after placing " + std::to_string(p.vram) + " in VRAM; available RAM " +
            std::to_string(ram) + ", available expert VRAM " + std::to_string(vram));
    return p;
}
} // namespace strata::poly
