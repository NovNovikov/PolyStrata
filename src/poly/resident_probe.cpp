// Deliberate large-allocation probe, run by hand: poly_resident_probe MIB. Frees the region before exit.
#include "resident_memory.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace {
void release_probe() {
    strata::poly::ResidentMemory memory;
    auto* keep = (unsigned char*) memory.allocate(4096, "replan keeper");
    keep[0] = 37;
    for (int i = 0; i < 3; ++i) {
#ifdef _WIN32
        SIZE_T before_min = 0, before_max = 0; DWORD before_flags = 0;
        if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &before_min, &before_max, &before_flags))
            throw std::runtime_error("release probe: cannot query working-set limits");
#endif
        void* p = memory.allocate(8192, "replan candidate");
        bool refused = false;
        try { memory.release_last(keep); } catch (const std::logic_error&) { refused = true; }
        if (!refused || memory.bytes() != 12288) throw std::runtime_error("release probe: ownership check failed");
        memory.release_last(p);
        if (memory.bytes() != 4096 || keep[0] != 37) throw std::runtime_error("release probe: remaining region changed");
#ifdef _WIN32
        SIZE_T after_min = 0, after_max = 0; DWORD after_flags = 0;
        if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &after_min, &after_max, &after_flags) ||
            before_min != after_min || before_max != after_max || before_flags != after_flags)
            throw std::runtime_error("release probe: working-set limits accumulated across retries");
#endif
    }
    memory.release_last(keep);
    if (memory.bytes() != 0) throw std::runtime_error("release probe: locked-byte count did not return to zero");
    std::puts("resident release probe: three reallocations, ownership, byte counts and working-set restore passed");
}
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: poly_resident_probe MIB [RUNS]\n"); return 2; }
    const auto mib = std::strtoull(argv[1], nullptr, 10);
    if (!mib || mib > SIZE_MAX >> 20) return 2;
    const int runs = argc == 3 ? std::atoi(argv[2]) : 1;
    if (runs < 1 || runs > 64) return 2;
    try {
      release_probe();
      for (int run = 0; run < runs; ++run) {
        const auto t = std::chrono::steady_clock::now();
        strata::poly::ResidentMemory memory;
        memory.allocate((size_t) mib << 20, "probe");
        std::printf("resident probe: requested=%llu MiB locked=%llu MiB in %.2f s\n", mib,
                    (unsigned long long) (memory.bytes() >> 20),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
      }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
