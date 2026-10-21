// One in-place exchange. Readers must have finished and RAM must be CUDA/HIP registered.
#pragma once
#include "transfer_audit_cuda.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {
struct ExchangeFailure : std::runtime_error {
    bool restored;
    ExchangeFailure(const std::string& why, bool ok) : std::runtime_error(why), restored(ok) {}
};
inline uint64_t exchange_hash(const uint8_t* p, size_t bytes, bool device) {
    uint64_t h = 14695981039346656037ull;
    std::vector<uint8_t> block(device ? std::min(bytes, size_t(65536)) : 0);
    for (size_t at = 0; at < bytes;) {
        const size_t n = std::min(bytes - at, size_t(65536));
        const uint8_t* source = p + at;
        if (device) {
            const auto e = PA_COPY_SYNC(block.data(), source, n, cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            source = block.data();
        }
        for (size_t i = 0; i < n; ++i) { h ^= source[i]; h *= 1099511628211ull; }
        at += n;
    }
    return h;
}
inline void exchange_transfer(uint8_t* gpu, uint8_t* ram, uint8_t* temporary, size_t bytes,
                              cudaStream_t stream, bool verify = false, int fail_after = 0) {
    const auto check = [](cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    };
    const auto copy = [&](void* to, const void* from, cudaMemcpyKind kind) {
        check(PA_COPY(to, from, bytes, kind, stream));
        check(PA_WAIT_STREAM(stream));
    };
    const uint64_t cold = verify ? exchange_hash(gpu, bytes, true) : 0;
    const uint64_t hot = verify ? exchange_hash(ram, bytes, false) : 0;
    int stage = 0;
    const auto fault = [&] {
        if (fail_after == stage) throw std::runtime_error("injected exchange failure");
    };
    if (fail_after == -1) throw ExchangeFailure("injected exchange failure before copy", true);
    try {
        copy(temporary, gpu, cudaMemcpyDeviceToDevice);
        stage = 1; fault();
        copy(gpu, ram, cudaMemcpyHostToDevice);
        stage = 2; fault();
        copy(ram, temporary, cudaMemcpyDeviceToHost);
        stage = 3; fault();
        if (verify && (exchange_hash(gpu, bytes, true) != hot || exchange_hash(ram, bytes, false) != cold))
            throw std::runtime_error("exchange payload checksum mismatch");
    } catch (const std::exception& e) {
        bool restored = true;
        try {
            check(PA_WAIT_STREAM(stream));
            // After step 2 the hot payload survives in GPU, even if step 3 partly wrote RAM.
            // Before step 2 completed the original hot payload is still intact in RAM.
            if (stage >= 2) copy(ram, gpu, cudaMemcpyDeviceToHost);
            if (stage >= 1) copy(gpu, temporary, cudaMemcpyDeviceToDevice);
            if (verify && (exchange_hash(gpu, bytes, true) != cold || exchange_hash(ram, bytes, false) != hot))
                restored = false;
        } catch (...) { restored = false; }
        throw ExchangeFailure(std::string("expert exchange: ") + e.what(), restored);
    }
}
} // namespace strata::poly
