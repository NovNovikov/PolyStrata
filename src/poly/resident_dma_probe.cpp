// By hand: poly_resident_dma_probe RAM_MIB [WINDOW_MIB] [RUNS]. No model or private text is used.
#include "resident_memory.hpp"
#include "resident_dma.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::poly;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: poly_resident_dma_probe RAM_MIB [WINDOW_MIB] [RUNS]\n"); return 2;
    }
    const auto mib = std::strtoull(argv[1], nullptr, 10), window_mib = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 3072;
    const int runs = argc >= 4 ? std::atoi(argv[3]) : 1;
    if (!mib || mib > SIZE_MAX >> 20 || !window_mib || window_mib > mib || runs < 1 || runs > 64) return 2;
    cudaStream_t streams[2]{};
    void* dev[2]{};
    int result = 0;
    try {
        for (int j = 0; j < 2; ++j) { ck(cudaStreamCreateWithFlags(&streams[j], cudaStreamNonBlocking)); ck(cudaMalloc(&dev[j], 4096)); }
        for (int run = 0; run < runs; ++run) {
            const auto start = std::chrono::steady_clock::now();
            ResidentMemory memory;
            const size_t bytes = (size_t) mib << 20, window = (size_t) window_mib << 20;
            auto* base = (uint8_t*) memory.allocate(bytes, "DMA probe");
            {
                ResidentDma dma(base, bytes, window);
                for (int pass = 0; pass < 4; ++pass) {
                    const size_t at = pass % 2 ? bytes - window : 0;
                    std::memset(base + at, 0x31 + pass, 4096);
                    std::memset(base + at + window - 4096, 0x61 + pass, 4096);
                    dma.prepare({{at, window}});
                    check(dma.bytes() == window, "wrong registered byte count");
                    // Reusing a window must keep it registered; rejected geometry must not evict it.
                    dma.prepare({{at, window}});
                    bool rejected = false;
                    try { dma.prepare({{bytes, 1}}); } catch (const std::runtime_error&) { rejected = true; }
                    check(rejected && dma.bytes() == window, "invalid source changed a valid window");
                    for (int j = 0; j < 2; ++j) {
                        dma.begin(streams[j]);
                        ck(cudaMemcpyAsync(dev[j], base + at + (j ? window - 4096 : 0), 4096,
                                           cudaMemcpyHostToDevice, streams[j]));
                        dma.end(streams[j]);
                    }
                    // Changing registration must fence BOTH independent streams before the old pages are unpinned.
                    if (bytes == window) dma.clear();
                    dma.prepare({{pass % 2 ? 0 : bytes - window, window}});
                    std::array<uint8_t, 4096> got{};
                    for (int j = 0; j < 2; ++j) {
                        ck(cudaMemcpy(got.data(), dev[j], got.size(), cudaMemcpyDeviceToHost));
                        for (uint8_t v : got) check(v == (j ? 0x61 : 0x31) + pass, "DMA payload mismatch");
                    }
                }
                dma.clear();
                // A real partial registration failure must roll back OUR first range, retaining an external one.
                ck(cudaHostRegister(base + bytes - 4096, 4096, 0));
                bool refused = false;
                try { dma.prepare({{0, 4096}, {bytes - 4096, 4096}}); }
                catch (const std::runtime_error&) { refused = true; }
                const bool rolled_back = refused && dma.bytes() == 0;
                ck(cudaHostUnregister(base + bytes - 4096));
                check(rolled_back, "partial registration failure did not roll back");
                // Disjoint original pages are supported as well as contiguous source windows.
                dma.prepare({{0, 4096}, {bytes - 4096, 4096}});
                check(dma.bytes() == 8192, "disjoint source pages were not registered");
                for (int j = 0; j < 2; ++j) {
                    auto* source = base + (j ? bytes - 4096 : 0);
                    std::memset(source, 0x41 + j, 4096);
                    dma.begin(streams[j]);
                    ck(cudaMemcpyAsync(dev[j], source, 4096, cudaMemcpyHostToDevice, streams[j]));
                    dma.end(streams[j]);
                }
                dma.clear();
                for (int j = 0; j < 2; ++j) {
                    std::array<uint8_t, 4096> got{};
                    ck(cudaMemcpy(got.data(), dev[j], got.size(), cudaMemcpyDeviceToHost));
                    for (uint8_t v : got) check(v == 0x41 + j, "disjoint DMA payload mismatch");
                }
                dma.prepare({{0, window}});
                // Destruction must also fence a partially issued operation without end().
                dma.begin(streams[0]);
                ck(cudaMemcpyAsync(dev[0], base, 4096, cudaMemcpyHostToDevice, streams[0]));
            }
#ifndef HIP_VERSION
            for (const size_t at : {size_t(0), bytes - 1}) {
                cudaPointerAttributes a{}; ck(cudaPointerGetAttributes(&a, base + at));
                check(a.type == cudaMemoryTypeUnregistered, "DMA source still registered after destruction");
            }
#endif
            std::printf("DMA probe: locked=%llu MiB, registered window=%llu MiB, two streams/4 switches/data/rollback/cleanup OK, %.2f s\n",
                         mib, window_mib, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "DMA probe: %s\n", e.what()); result = 1; }
    for (int j = 0; j < 2; ++j) { if (streams[j]) cudaStreamSynchronize(streams[j]); if (dev[j]) cudaFree(dev[j]); if (streams[j]) cudaStreamDestroy(streams[j]); }
    return result;
}
