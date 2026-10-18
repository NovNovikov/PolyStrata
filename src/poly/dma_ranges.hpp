// Page ranges registered for a bounded DMA working window, inside one owned RAM arena.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::poly {
using DmaRange = std::pair<size_t, size_t>;   // offset, bytes
inline std::vector<DmaRange> dma_ranges(std::vector<DmaRange> ranges, size_t arena, size_t limit, size_t page) {
    if (!page || (page & (page - 1)) || arena > SIZE_MAX - (page - 1))
        throw std::runtime_error("no-disk: invalid DMA page geometry");
    std::vector<DmaRange> pages;
    for (const auto& r : ranges) {
        if (r.first > arena || r.second > arena - r.first)
            throw std::runtime_error("no-disk: DMA source outside the resident arena");
        if (!r.second) continue;
        const size_t first = r.first & ~(page - 1), end = (r.first + r.second + page - 1) & ~(page - 1);
        pages.emplace_back(first, end - first);
    }
    std::sort(pages.begin(), pages.end());
    std::vector<DmaRange> merged;
    for (const auto& r : pages) {
        if (!merged.empty() && r.first <= merged.back().first + merged.back().second) {
            auto& last = merged.back();
            last.second = std::max(last.first + last.second, r.first + r.second) - last.first;
        } else merged.push_back(r);
    }
    size_t total = 0;
    for (const auto& r : merged) {
        if (r.second > limit - total) throw std::runtime_error("no-disk: DMA registration window exceeds its budget");
        total += r.second;
    }
    return merged;
}
} // namespace strata::poly
