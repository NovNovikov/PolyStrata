// Real-weight CPU decode microbenchmark. No server, private prompts, or expert profiles.
// Usage: poly_cpu_decode_bench MODEL.gguf [sweeps=16] [exclusive-reference-output.bin]
#include "cpu_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#ifdef STRATA_VNNI_PROBE
#include <intrin.h>
#endif

using Clock = std::chrono::steady_clock;
using namespace strata;
struct Layer {
    int id, gu_type, d_type, embd, ff;
    size_t gu_bytes, d_bytes;
    std::vector<uint8_t> weights;
    std::vector<float> reference;
};
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    try {
#ifdef STRATA_VNNI_PROBE
        int cpuid[4]; __cpuidex(cpuid, 7, 1);
        if (!(cpuid[0] & (1 << 4))) { std::fprintf(stderr, "AVX-VNNI is not supported by this CPU\n"); return 77; }
        std::fprintf(stderr, "CPUID.7.1:EAX AVX-VNNI supported; ggml x86 dot products use AVX-VNNI\n");
#endif
#ifdef _WIN32
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0x1ul | 0x4ul, 0};
        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &off, sizeof(off));
#endif
        const int sweeps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 16;
        std::vector<Layer> layers;
        size_t bytes = 0;
        constexpr int experts = 4;
        {
            GgufFile file(argv[1]);
            for (const auto& gate : file.tensors()) {
                const auto at = gate.name.find(".ffn_gate_exps.weight");
                if (at == std::string::npos || gate.shape.size() != 3) continue;
                const auto prefix = gate.name.substr(0, at);
                const auto* up = file.find(prefix + ".ffn_up_exps.weight");
                const auto* down = file.find(prefix + ".ffn_down_exps.weight");
                if (!up || !down || up->type != gate.type || gate.shape[2] < experts) continue;
                Layer l{};
                l.id = std::stoi(prefix.substr(4)); l.gu_type = gate.type; l.d_type = down->type;
                l.embd = (int) gate.shape[0]; l.ff = (int) gate.shape[1];
                l.gu_bytes = tensor_payload_bytes(gate) / gate.shape[2];
                l.d_bytes = tensor_payload_bytes(*down) / down->shape[2];
                std::string err;
                if (!poly::cpu_expert_types_ok(l.gu_type, l.d_type, l.embd, l.ff, err)) throw std::runtime_error(err);
                l.weights.resize(experts * (2 * l.gu_bytes + l.d_bytes));
                for (int e = 0; e < experts; ++e) {
                    auto* dst = l.weights.data() + e * (2 * l.gu_bytes + l.d_bytes);
                    std::memcpy(dst, file.tensor_data(gate) + e * l.gu_bytes, l.gu_bytes);
                    std::memcpy(dst + l.gu_bytes, file.tensor_data(*up) + e * l.gu_bytes, l.gu_bytes);
                    std::memcpy(dst + 2 * l.gu_bytes, file.tensor_data(*down) + e * l.d_bytes, l.d_bytes);
                }
                bytes += l.weights.size();
                std::fprintf(stderr, "layer=%d gu=%s down=%s embd=%d ff=%d bytes=%zu\n", l.id,
                             ggml_type_name(l.gu_type), ggml_type_name(l.d_type), l.embd, l.ff, l.weights.size());
                layers.push_back(std::move(l));
            }
        }
        if (layers.empty()) throw std::runtime_error("no routed layers");
        std::vector<float> input(layers.front().embd), output(input.size());
        std::mt19937 rng(1234);
        // Exact binary fractions: identical inputs across compiler/ISA variants, without libm RNG transforms.
        for (auto& v : input) v = float(int(rng() & 1023) - 512) / 8192.f;
        std::printf("# layers=%zu experts_per_layer=%d weights_bytes=%zu sweeps=%d\n", layers.size(), experts, bytes, sweeps);
        std::puts("round,threads,mode,ms_per_sweep,logical_weight_GiB_s,bitwise_reference,checksum,prepare_ms,gate_up_ms,quantize_ms,down_ms");
        for (int round = 0; round < 3; ++round) {
            std::vector<int> counts = {1, 3, 6, 8, 12};
            std::shuffle(counts.begin(), counts.end(), rng);
            // The first reference is computed outside every timed interval.
            for (const int threads : counts) {
                poly::Workers pool(threads);
                auto call = [&](Layer& l) {
                    poly::CpuExpert jobs[experts];
                    for (int e = 0; e < experts; ++e) {
                        const auto* p = l.weights.data() + e * (2 * l.gu_bytes + l.d_bytes);
                        jobs[e] = {p, p + l.gu_bytes, p + 2 * l.gu_bytes, .25f};
                    }
                    poly::cpu_experts(pool, jobs, experts, 1, l.gu_type, l.gu_type, l.d_type, l.embd, l.ff,
                                      input.data(), 10.f, false, output.data());
                };
                bool same = true;
                for (auto& l : layers) {
                    call(l);
                    if (l.reference.empty()) l.reference = output;
                    else same &= std::memcmp(l.reference.data(), output.data(), output.size() * sizeof(float)) == 0;
                }
                for (int mode = 0; mode < 2; ++mode) {
                    poly::cpu_stage_timing() = {};
                    const auto start = Clock::now();
                    double sum = 0;
                    for (int s = 0; s < sweeps; ++s) for (size_t j = 0; j < layers.size(); ++j) {
                        call(layers[mode == 0 ? j : 0]);
                        sum += output[(s + j) % output.size()];
                    }
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / sweeps;
                    const size_t logical_bytes = mode == 0 ? bytes : layers.front().weights.size() * layers.size();
                    const auto times = poly::cpu_stage_timing();
                    std::printf("%d,%d,%s,%.6f,%.6f,%d,%.9g,%.6f,%.6f,%.6f,%.6f\n", round, threads,
                                mode == 0 ? "stream_layers" : "repeat_layer", ms,
                                logical_bytes / (1024.0 * 1024 * 1024) / (ms / 1000), same, sum,
                                times.prepare / sweeps, times.gate_up / sweeps,
                                times.quantize / sweeps, times.down / sweeps);
                    std::fflush(stdout);
                }
                const auto start = Clock::now();
                std::atomic<int> sink{0};
                for (int i = 0; i < 1000; ++i) pool.run(256, [&](int j) { if (j == 0) sink.fetch_add(1); });
                std::printf("%d,%d,dispatch256,%.6f,0,1,%d,0,0,0,0\n", round, threads,
                            std::chrono::duration<double, std::milli>(Clock::now() - start).count() / 1000, sink.load());
                std::fflush(stdout);
            }
        }
        if (argc > 3) {
#ifdef _WIN32
            HANDLE reference = CreateFileA(argv[3], GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (reference == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot exclusively create reference output");
            auto write = [&](const std::vector<float>& data) {
                DWORD done = 0, bytes = (DWORD) (data.size() * sizeof(float));
                return WriteFile(reference, data.data(), bytes, &done, nullptr) && done == bytes;
            };
            bool ok = write(input);
            for (const auto& l : layers) ok &= write(l.reference);
            ok &= CloseHandle(reference) != 0;
#else
            FILE* reference = std::fopen(argv[3], "wbx");
            if (!reference) throw std::runtime_error("cannot exclusively create reference output");
            bool ok = std::fwrite(input.data(), sizeof(float), input.size(), reference) == input.size();
            for (const auto& l : layers)
                ok &= std::fwrite(l.reference.data(), sizeof(float), l.reference.size(), reference) == l.reference.size();
            ok &= std::fclose(reference) == 0;
#endif
            if (!ok) throw std::runtime_error("failed to write reference output");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
