#include "resident_memory.hpp"
#include "resident_dma.hpp"
#include "exchange_transfer.hpp"
#include <cstdio>
#include <cstring>

using namespace strata::poly;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b) { if (!b) throw std::runtime_error("exchange GPU invariant failed"); }
}
int main() {
    cudaStream_t stream = nullptr;
    uint8_t *allocation = nullptr, *temporary = nullptr;
    try {
        constexpr size_t raw = 65536, guard = 256;
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ck(cudaMalloc((void**) &allocation, raw + 3 * guard));
        ck(cudaMalloc((void**) &temporary, raw + 2 * guard));
        ck(cudaMemset(allocation, 0xA7, raw + 3 * guard));
        ck(cudaMemset(temporary, 0xB8, raw + 2 * guard));
        uint8_t* v = allocation + guard;
        uint8_t* t = temporary + guard;
        ck(cudaMemset(v, 0x11, raw));
        // The GPU tail remains zero and surrounding slot canaries remain unchanged.
        ck(cudaMemset(v + raw, 0, guard));
        ResidentMemory memory;
        auto* r = (uint8_t*) memory.allocate(16 * raw, "exchange GPU test");
        std::memset(r, 0x5A, 16 * raw);
        ResidentDma dma(r, 16 * raw, 2 * raw);
        dma.prepare({{0, 2 * raw}});
        for (int i = 0; i < 1000; ++i) {
            dma.retain({{0, raw}});
            dma.begin(stream);
            exchange_transfer(v, r, t, raw, stream, true);
            dma.end(stream);
        }
        // An even number of exchanges restores both starting payloads.
        std::vector<uint8_t> expected(raw, 0x11);
        require(exchange_hash(v, raw, true) == exchange_hash(expected.data(), raw, false));
        expected.assign(raw, 0x5A);
        require(exchange_hash(r, raw, false) == exchange_hash(expected.data(), raw, false));
        for (int fault : {-1, 1, 2, 3}) {
            const auto a = exchange_hash(v, raw, true), b = exchange_hash(r, raw, false);
            bool caught = false;
            try { exchange_transfer(v, r, t, raw, stream, true, fault); }
            catch (const ExchangeFailure& e) { caught = true; require(e.restored); }
            require(caught && exchange_hash(v, raw, true) == a && exchange_hash(r, raw, false) == b);
        }
        std::vector<uint8_t> gpu(raw + 3 * guard);
        ck(cudaMemcpy(gpu.data(), allocation, gpu.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) {
            require(gpu[i] == 0xA7 && gpu[guard + raw + i] == 0 && gpu[2 * guard + raw + i] == 0xA7);
        }
        std::vector<uint8_t> temp(raw + 2 * guard);
        ck(cudaMemcpy(temp.data(), temporary, temp.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) require(temp[i] == 0xB8 && temp[guard + raw + i] == 0xB8);
        // Reuse, overlap, eviction and re-registration of original pages, never above the bound.
        for (size_t i = 0; i < 16; ++i) {
            dma.retain({{i * raw + 1, raw - 2}});
            require(dma.bytes() <= 2 * raw && dma.peak() <= 2 * raw);
            const auto registered = dma.registered_bytes();
            dma.retain({{i * raw + 3, raw - 6}});
            require(dma.registered_bytes() == registered);
        }
        std::printf("exchange_gpu_test: 1000 verified in-place exchanges, four rollback points, "
                    "payloads/tails/canaries and bounded retained DMA registrations OK\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exchange_gpu_test: %s\n", e.what());
        if (stream) cudaStreamSynchronize(stream);
        if (allocation) cudaFree(allocation);
        if (temporary) cudaFree(temporary);
        if (stream) cudaStreamDestroy(stream);
        return 1;
    }
    ck(cudaFree(allocation)); ck(cudaFree(temporary)); ck(cudaStreamDestroy(stream));
    return 0;
}
