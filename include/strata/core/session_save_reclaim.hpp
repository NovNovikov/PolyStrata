// Optional SAVE/RESTORE admission: free reconstructible host caches, never the live device state.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

namespace strata::core {

struct SessionSaveReclaimed {
    size_t checkpoints = 0, parked = 0, bytes = 0;
};

// Called BEFORE session_save_checkpoints selects into chain. Probe after each
// eviction: freed capacity is not necessarily physical RAM returned by the allocator.
// No allocation is needed here. Unknown telemetry / sizes cannot justify cache eviction.
// The chain is never touched: every checkpoint in it (the deepest and the pins) goes
// into the file, so freeing its states would break the save it is making room for.
// Only the caches are given up, in order: retained reuse, then parked conversations.
// (For RESTORE see session_restore_reclaim below: the file replaces the outgoing
// session whole, so its checkpoints may go too.)
template<class Probe>
SessionSaveReclaimed session_save_reclaim(std::vector<ConversationCheckpoint>& chain,
                                         ConversationCache& cache, const SessionSaveLive& live,
                                         uint64_t floor, Probe available) {
    SessionSaveReclaimed out;
    const uint64_t need = session_save_peak_bytes(chain, live);
    auto room = available();
    auto done = [&] { return !room || need == UINT64_MAX || conversation_memory_admit(room, need, floor); };
    if (done()) return out;
    if (cache.retained_bytes()) {
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        room = available();
        if (done()) return out;
    }
    while (true) {
        const size_t before = cache.bytes();
        if (!cache.evict_oldest()) break; // pinned conversations stay parked
        ++out.parked;
        out.bytes += before - cache.bytes();
        room = available();
        if (done()) return out;
    }
    return out;
}

// RESTORE: the incoming file replaces the outgoing session whole, so every reconstructible
// cache of the outgoing one may be given up when the parse refuses. The parse's own peak
// arrives as `need`; probe after each eviction as above. Pins still survive (they are
// explicit) and unknown telemetry still stops the search.
template<class Probe>
SessionSaveReclaimed session_restore_reclaim(std::vector<ConversationCheckpoint>& chain,
                                             ConversationCache& cache, uint64_t need,
                                             uint64_t floor, Probe available) {
    SessionSaveReclaimed out;
    auto room = available();
    auto done = [&] { return !room || need == UINT64_MAX || conversation_memory_admit(room, need, floor); };
    if (done()) return out;
    if (cache.retained_bytes()) {
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        room = available();
        if (done()) return out;
    }
    for (bool progress = true; progress;) {
        progress = false;
        const size_t before = cache.bytes();
        if (cache.evict_oldest()) {
            ++out.parked;
            out.bytes += before - cache.bytes();
            room = available();
            if (done()) return out;
            progress = true;
            continue;
        }
        // Erase from the back to preserve the order of what remains.
        for (size_t i = chain.size(); i-- > 0;) {
            if (chain[i].pinned) continue;
            out.bytes += chain[i].bytes();
            chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i));
            ++out.checkpoints;
            room = available();
            if (done()) return out;
            progress = true;
            break;
        }
    }
    return out;
}

} // namespace strata::core
