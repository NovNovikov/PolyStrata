// What the DMA window's own cost was: the wait for the DMA that used the pages, the unregistration, and the
// registration.  Kept apart from resident_dma.hpp so that a header that only needs the numbers does not pull
// the CUDA runtime in with it.
#pragma once
#include <cstdint>

namespace strata::poly {
struct DmaCost {
    int64_t calls = 0;        ///< the times the window was changed
    int64_t added = 0;         ///< the bytes registered
    int64_t removed = 0;       ///< the bytes unregistered
    double drain_ms = 0;       ///< waiting for the DMA that used the pages
    double unregister_ms = 0;  ///< cudaHostUnregister
    double register_ms = 0;    ///< cudaHostRegister
};

/// What a call of prepare() or retain() cost: the difference of the counters around it.
inline DmaCost cost_delta(const DmaCost& now, const DmaCost& before) {
    DmaCost d;
    d.calls = now.calls - before.calls;
    d.added = now.added - before.added;
    d.removed = now.removed - before.removed;
    d.drain_ms = now.drain_ms - before.drain_ms;
    d.unregister_ms = now.unregister_ms - before.unregister_ms;
    d.register_ms = now.register_ms - before.register_ms;
    return d;
}
inline void add_cost(DmaCost& to, const DmaCost& now, const DmaCost& before) {
    const DmaCost d = cost_delta(now, before);
    to.calls += d.calls;
    to.added += d.added;
    to.removed += d.removed;
    to.drain_ms += d.drain_ms;
    to.unregister_ms += d.unregister_ms;
    to.register_ms += d.register_ms;
}
} // namespace strata::poly
