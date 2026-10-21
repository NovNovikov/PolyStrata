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
// Both inputs are sorted, disjoint page ranges. Return only uncovered pages, never overlapping
// an existing CUDA registration (a registration can be removed only as a whole).
inline std::vector<DmaRange> dma_missing(const std::vector<DmaRange>& wanted,
                                         const std::vector<DmaRange>& held) {
    std::vector<DmaRange> out;
    for (const auto& w : wanted) {
        size_t at = w.first, end = w.first + w.second;
        for (const auto& h : held) {
            const size_t he = h.first + h.second;
            if (he <= at) continue;
            if (h.first >= end) break;
            if (h.first > at) out.emplace_back(at, h.first - at);
            at = std::max(at, he);
            if (at >= end) break;
        }
        if (at < end) out.emplace_back(at, end - at);
    }
    return out;
}
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
