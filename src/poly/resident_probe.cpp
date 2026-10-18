// Deliberate large-allocation probe, run by hand: poly_resident_probe MIB. Frees the region before exit.
#include "resident_memory.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: poly_resident_probe MIB [RUNS]\n"); return 2; }
    const auto mib = std::strtoull(argv[1], nullptr, 10);
    if (!mib || mib > SIZE_MAX >> 20) return 2;
    const int runs = argc == 3 ? std::atoi(argv[2]) : 1;
    if (runs < 1 || runs > 64) return 2;
    try {
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
