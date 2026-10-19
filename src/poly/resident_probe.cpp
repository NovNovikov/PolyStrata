// Deliberate large-allocation probe, run by hand: poly_resident_probe MIB. Frees the region before exit.
#include "resident_memory.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <psapi.h>
#endif
namespace {
#ifdef _WIN32
void verify_locked(void* p, size_t n) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const size_t page = info.dwPageSize;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages(4096);
    for (size_t offset = 0; offset < n;) {
        size_t count = 0;
        while (count < pages.size() && offset < n) {
            pages[count++] = {};
            pages[count - 1].VirtualAddress = (unsigned char*) p + offset;
            offset += page;
        }
        if (!QueryWorkingSetEx(GetCurrentProcess(), pages.data(), (DWORD) (count * sizeof(pages[0]))))
            throw std::runtime_error("resident probe: QueryWorkingSetEx failed");
        for (size_t i = 0; i < count; ++i)
            if (!pages[i].VirtualAttributes.Valid || !pages[i].VirtualAttributes.Locked)
                throw std::runtime_error("resident probe: a page is not resident and locked");
    }
    std::puts("resident probe: every page verified resident and locked via QueryWorkingSetEx");
}
#endif
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
#ifdef _WIN32
      // An impossible address-space request must report the actual Windows allocation error, then clean up.
      bool refused = false;
      try {
          strata::poly::ResidentMemory memory;
          memory.allocate(SIZE_MAX, "impossible allocation probe");
      } catch (const std::runtime_error& e) {
          const std::string message = e.what();
          refused = message.find("VirtualAlloc failed") != std::string::npos &&
                    message.find("(error ") != std::string::npos;
      }
      if (!refused) throw std::runtime_error("resident probe: missing Windows allocation error");
      std::puts("resident probe: impossible allocation reports Windows error and releases cleanly");
#endif
      for (int run = 0; run < runs; ++run) {
        const auto t = std::chrono::steady_clock::now();
        strata::poly::ResidentMemory memory;
        void* p = memory.allocate((size_t) mib << 20, "probe");
#ifdef _WIN32
        verify_locked(p, (size_t) mib << 20);
#endif
        std::printf("resident probe: requested=%llu MiB locked=%llu MiB in %.2f s\n", mib,
                    (unsigned long long) (memory.bytes() >> 20),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
      }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
