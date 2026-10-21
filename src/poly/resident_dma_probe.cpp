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
                dma.clear();
                dma.prepare_stage({{0, 4096}, {8192, 4096}});
                check(dma.bytes() == 12288, "stage did not merge the bounded gap");
                const auto registered = dma.registered_bytes(), unregistered = dma.unregistered_bytes();
                dma.prepare_stage({{8192 + 17, 1}});
                check(dma.bytes() == 12288 && dma.registered_bytes() == registered &&
                      dma.unregistered_bytes() == unregistered, "covered stage subset changed registration");
                bool invalid = false;
                try { dma.prepare_stage({{bytes, 1}}); } catch (const std::runtime_error&) { invalid = true; }
                check(invalid && dma.bytes() == 12288, "invalid stage range evicted valid pages");
                // The exact same [G,U,D] bytes and staging padding as stage(), with an unaligned GU size.
                constexpr size_t gu = 333, down = 777, count = 2;
                constexpr size_t down_at = (count * 2 * gu + 255) & ~size_t(255);
                constexpr size_t output_bytes = down_at + count * down + 256;
                static_assert(output_bytes <= 4096);
                for (size_t e = 0; e < count; ++e) {
                    auto* src = base + e * 8192;
                    for (size_t i = 0; i < 2 * gu + down; ++i) src[i] = (uint8_t) (i * 31 + e * 17);
                }
                // streams[0] is the original three-copy layout, streams[1] the merged two-copy layout.
                for (int j = 0; j < 2; ++j) {
                    ck(cudaMemsetAsync(dev[j], 0, 4096, streams[j]));
                    dma.begin(streams[j]);
                    for (size_t e = 0; e < count; ++e) {
                        const auto* src = base + e * 8192;
                        auto* dst = (uint8_t*) dev[j] + e * 2 * gu;
                        ck(cudaMemcpyAsync(dst, src, j ? 2 * gu : gu, cudaMemcpyHostToDevice, streams[j]));
                        if (!j) ck(cudaMemcpyAsync(dst + gu, src + gu, gu, cudaMemcpyHostToDevice, streams[j]));
                        ck(cudaMemcpyAsync((uint8_t*) dev[j] + down_at + e * down, src + 2 * gu, down,
                                           cudaMemcpyHostToDevice, streams[j]));
                    }
                    dma.end(streams[j]);
                }
                // Eviction fences both outstanding source readers, including the merged copy.
                dma.clear();
                std::array<uint8_t, output_bytes> original{}, merged{};
                ck(cudaMemcpy(original.data(), dev[0], original.size(), cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(merged.data(), dev[1], merged.size(), cudaMemcpyDeviceToHost));
                check(original == merged, "merged gate/up changed staging bytes/padding/tail");
                for (size_t e = 0; e < count; ++e) {
                    check(std::memcmp(merged.data() + e * 2 * gu, base + e * 8192, 2 * gu) == 0, "gate/up data");
                    check(std::memcmp(merged.data() + down_at + e * down, base + e * 8192 + 2 * gu, down) == 0, "down data");
                }
                if (bytes >= 4u * 1024 * 1024) {
                    // The decode cache's own policy: the pages that are registered stay registered, only the
                    // missing ones are added, and an old whole registration goes only when the budget needs it.
                    const size_t page = ResidentDma::page_size();
                    {
                        ResidentDma kept(base, bytes, 4 * page);
                        // A fully covered batch must not touch the CUDA registration at all.
                        kept.prepare({{0, page}});
                        const uint64_t reg0 = kept.registered_bytes(), unreg0 = kept.unregistered_bytes();
                        kept.retain({{0, page}});
                        check(kept.registered_bytes() == reg0 && kept.unregistered_bytes() == unreg0,
                              "a covered retain changed the registration");
                        check(kept.bytes() == page, "a covered retain changed the window");
                        // A partial intersection keeps the old range and adds what is missing.
                        kept.retain({{page, 2 * page}});
                        check(kept.bytes() == 3 * page, "a partial intersection did not add the missing page");
                        // The page the retain added must carry its data.
                        std::memset(base + page, 0x5a, page);
                        kept.begin(streams[0]);
                        ck(cudaMemcpyAsync(dev[0], base + page, page, cudaMemcpyHostToDevice, streams[0]));
                        kept.end(streams[0]);
                        std::array<uint8_t, 4096> got{};
                        ck(cudaMemcpy(got.data(), dev[0], got.size(), cudaMemcpyDeviceToHost));
                        for (uint8_t v : got) check(v == 0x5a, "the page a retain added has the wrong data");
                        // An invalid source must not change the window.
                        bool invalid = false;
                        try { kept.retain({{bytes + 1, 1}}); } catch (const std::runtime_error&) { invalid = true; }
                        check(invalid && kept.bytes() == 3 * page, "an invalid source changed the window");
                    }
                    // The budget: a new source that does not fit evicts the oldest registration, and the
                    // wanted pages stay.
                    {
                        ResidentDma tight(base, bytes, 3 * page);
                        tight.prepare({{0, page}});
                        tight.retain({{0, 3 * page}});
                        check(tight.bytes() == 3 * page, "the retained window is not the budget");
                        tight.retain({{3 * page, page}});
                        check(tight.bytes() == 3 * page, "the budget was exceeded");
#ifndef HIP_VERSION
                        {
                            cudaPointerAttributes a{};
                            ck(cudaPointerGetAttributes(&a, base));
                            check(a.type == cudaMemoryTypeUnregistered, "the evicted page is still registered");
                            ck(cudaPointerGetAttributes(&a, base + 3 * page));
                            check(a.type == cudaMemoryTypeHost, "the wanted page is not registered");
                        }
#endif
                    }
                    // An eviction must wait for the readers of the pages it removes.
                    {
                        ResidentDma fenced(base, bytes, page);
                        std::array<uint8_t, 4096> got{};
                        fenced.prepare({{0, page}});
                        std::memset(base, 0x6b, page);
                        fenced.begin(streams[0]);
                        ck(cudaMemcpyAsync(dev[0], base, page, cudaMemcpyHostToDevice, streams[0]));
                        fenced.retain({{page, page}});
                        ck(cudaMemcpy(got.data(), dev[0], got.size(), cudaMemcpyDeviceToHost));
                        for (uint8_t v : got) check(v == 0x6b, "the fenced reader's data mismatch");
                    }
                    // A real partial registration failure must roll back and leave nothing registered.
                    {
                        ResidentDma roll(base, bytes, 4 * page);
                        roll.prepare({{0, page}});
                        ck(cudaHostRegister(base + bytes - page, page, 0));
                        bool refused = false;
                        try { roll.retain({{0, page}, {bytes - page, page}}); }
                        catch (const std::runtime_error&) { refused = true; }
                        const bool rolled_back = refused && roll.bytes() == page;
                        ck(cudaHostUnregister(base + bytes - page));
                        check(rolled_back, "a partial registration failure did not roll back the window");
                    }
                    // Cleanup, and the same policy applied again.
                    {
                        ResidentDma again(base, bytes, 4 * page);
                        again.retain({{0, page}});
                        check(again.bytes() == page, "the policy did not apply to a fresh window");
                        again.clear();
                        check(again.bytes() == 0, "the cleanup left pages registered");
                        again.retain({{0, page}});
                        check(again.bytes() == page, "the policy did not apply after a cleanup");
                        again.clear();
                    }
                    // The engine's fallback: the gate and the up are not adjacent in RAM, so they are copied
                    // separately. The staging bytes must be the same as the original three-copy layout.
                    {
                        constexpr size_t fgu = 333, fdown = 777, fcount = 2;
                        constexpr size_t fdown_at = (fcount * 2 * fgu + 255) & ~size_t(255);
                        constexpr size_t fout = fdown_at + fcount * fdown + 256;
                        static_assert(fout <= 4096);
                        const size_t gate_at = 0, up_at = 8192, down_src = 16384;
                        for (size_t e = 0; e < fcount; ++e) {
                            uint8_t* gate = base + gate_at + e * 1024;
                            uint8_t* up = base + up_at + e * 1024;
                            uint8_t* dn = base + down_src + e * 1024;
                            for (size_t i = 0; i < fgu; ++i) { gate[i] = (uint8_t) (i * 31 + e * 17); up[i] = (uint8_t) (i * 7 + e * 3); }
                            for (size_t i = 0; i < fdown; ++i) dn[i] = (uint8_t) (i * 13 + e * 5);
                        }
                        ck(cudaMemsetAsync(dev[0], 0, fout, streams[0]));
                        ResidentDma split(base, bytes, 64 * 1024);
                        split.prepare({{gate_at, 1024}, {up_at, 1024}, {down_src, 1024}});
                        split.begin(streams[0]);
                        for (size_t e = 0; e < fcount; ++e) {
                            uint8_t* dst = (uint8_t*) dev[0] + e * 2 * fgu;
                            ck(cudaMemcpyAsync(dst, base + gate_at + e * 1024, fgu, cudaMemcpyHostToDevice, streams[0]));
                            ck(cudaMemcpyAsync(dst + fgu, base + up_at + e * 1024, fgu, cudaMemcpyHostToDevice, streams[0]));
                            ck(cudaMemcpyAsync((uint8_t*) dev[0] + fdown_at + e * fdown, base + down_src + e * 1024,
                                               fdown, cudaMemcpyHostToDevice, streams[0]));
                        }
                        split.end(streams[0]);
                        std::array<uint8_t, fout> apart{};
                        ck(cudaMemcpy(apart.data(), dev[0], apart.size(), cudaMemcpyDeviceToHost));
                        for (size_t e = 0; e < fcount; ++e) {
                            check(std::memcmp(apart.data() + e * 2 * fgu, base + gate_at + e * 1024, fgu) == 0, "the split gate");
                            check(std::memcmp(apart.data() + e * 2 * fgu + fgu, base + up_at + e * 1024, fgu) == 0, "the split up");
                            check(std::memcmp(apart.data() + fdown_at + e * fdown, base + down_src + e * 1024, fdown) == 0,
                                  "the split down");
                        }
                        split.clear();
                    }
                    // The last expert at the end of the arena: the registered range must cover both full GU
                    // halves of it, and the copy of them must be exact.
                    {
                        constexpr size_t lgu = 333, ldown = 777;
                        const size_t at = bytes - (2 * lgu + ldown);
                        uint8_t* src = base + at;
                        for (size_t i = 0; i < 2 * lgu + ldown; ++i) src[i] = (uint8_t) (i * 11 + 3);
                        ResidentDma last(base, bytes, 64 * 1024);
                        last.prepare({{at, 2 * lgu + ldown}});
                        last.begin(streams[0]);
                        ck(cudaMemcpyAsync(dev[0], src, 2 * lgu, cudaMemcpyHostToDevice, streams[0]));
                        last.end(streams[0]);
                        std::array<uint8_t, 2 * lgu> tail{};
                        ck(cudaMemcpy(tail.data(), dev[0], tail.size(), cudaMemcpyDeviceToHost));
                        check(std::memcmp(tail.data(), src, 2 * lgu) == 0, "the merged GU at the arena's end changed the bytes");
                        last.clear();
                    }
                }
                if (bytes > window) {
                    dma.prepare_stage({{0, 4096}, {bytes - 4096, 4096}});
                    check(dma.bytes() == 8192, "stage span exceeded budget rather than falling back");
                    dma.clear();
                }
                dma.prepare_stage({{0, 4096}, {8192, 4096}}, true);
                check(dma.bytes() == 12288, "retained stage did not merge bounded gap");
                dma.clear();
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
