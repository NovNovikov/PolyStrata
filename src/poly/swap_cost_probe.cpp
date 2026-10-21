// By hand: poly_swap_cost_probe [SLOTS] [WINDOW_MIB] [EXPERT_MIB].
// The price of one exchange, with every exchange independent of the others.
//
// One exchange moves one expert: the cold expert VRAM->staging (D2D), the hot expert RAM->VRAM (H2D), the cold
// expert staging->RAM (D2H).  Each exchange here has its own VRAM source, its own VRAM target, its own staging
// slot, its own RAM source and its own RAM target, so a batch of them is a batch of independent exchanges and
// not several copies fighting over one buffer.
#include "resident_memory.hpp"
#include "resident_dma.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::poly;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}

int main(int argc, char** argv) {
    const int slots = argc >= 2 ? std::atoi(argv[1]) : 8;
    const size_t window_mib = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 2048;
    const size_t expert_mib = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 8;
    const size_t expert_bytes = expert_mib << 20;
    if (slots < 1 || slots > 64) { std::fprintf(stderr, "usage: poly_swap_cost_probe [SLOTS] [WINDOW_MIB] [EXPERT_MIB]\n"); return 2; }

    cudaStream_t streams[4]{};
    void* vram_cold = nullptr;      // the cold experts, where they live on the card
    void* vram_target = nullptr;    // where the hot experts go
    void* staging = nullptr;        // the temporary slots
    int result = 0;
    try {
        for (int j = 0; j < 4; ++j) ck(cudaStreamCreateWithFlags(&streams[j], cudaStreamNonBlocking));
        ck(cudaMalloc(&vram_cold, expert_bytes * slots));
        ck(cudaMalloc(&vram_target, expert_bytes * slots));
        ck(cudaMalloc(&staging, expert_bytes * slots));
        ck(cudaMemset(vram_cold, 0x11, expert_bytes * slots));
        ck(cudaMemset(vram_target, 0x22, expert_bytes * slots));

        // the RAM arena: a source and a target per exchange, in one locked buffer
        ResidentMemory memory;
        const size_t ram_bytes = expert_bytes * slots * 2;
        auto* ram = (uint8_t*) memory.allocate(ram_bytes, "swap cost probe");
        std::memset(ram, 0x5A, ram_bytes);
        // the pages of the arena are registered, as the engine's are: the D2H writes into a registered page
        ResidentDma dma(ram, ram_bytes, ram_bytes);
        auto t = Clock::now();
        dma.prepare({{0, ram_bytes}});
        const double register_ms = ms_since(t);

        const int rounds = 20;
        // one exchange, on one stream, into its own buffers
        const auto one_exchange = [&](cudaStream_t s) {
            ck(cudaMemcpyAsync((uint8_t*) staging, vram_cold, expert_bytes, cudaMemcpyDeviceToDevice, s));
            ck(cudaMemcpyAsync(vram_target, ram, expert_bytes, cudaMemcpyHostToDevice, s));
            ck(cudaMemcpyAsync(ram + expert_bytes, staging, expert_bytes, cudaMemcpyDeviceToHost, s));
        };
        // a warm-up, so the first use of each copy kind is not in the numbers
        one_exchange(streams[0]);
        ck(cudaStreamSynchronize(streams[0]));

        // 1. the three copies, one exchange at a time
        t = Clock::now();
        for (int i = 0; i < rounds; ++i) {
            ck(cudaMemcpyAsync((uint8_t*) staging, vram_cold, expert_bytes, cudaMemcpyDeviceToDevice, streams[0]));
        }
        ck(cudaStreamSynchronize(streams[0]));
        const double d2d = ms_since(t) / rounds;
        t = Clock::now();
        for (int i = 0; i < rounds; ++i) {
            ck(cudaMemcpyAsync(vram_target, ram, expert_bytes, cudaMemcpyHostToDevice, streams[0]));
        }
        ck(cudaStreamSynchronize(streams[0]));
        const double h2d = ms_since(t) / rounds;
        t = Clock::now();
        for (int i = 0; i < rounds; ++i) {
            ck(cudaMemcpyAsync(ram + expert_bytes, (uint8_t*) staging, expert_bytes, cudaMemcpyDeviceToHost, streams[0]));
        }
        ck(cudaStreamSynchronize(streams[0]));
        const double d2h = ms_since(t) / rounds;
        std::printf("эксперт %zu МиБ, страницы RAM зарегистрированы: D2D %.3f мс, H2D %.3f мс, D2H %.3f мс; "
                    "обмен %.3f мс\n", expert_mib, d2d, h2d, d2h, d2d + h2d + d2h);
        std::printf("   полоса: D2D %.1f ГБ/с, H2D %.1f ГБ/с, D2H %.1f ГБ/с\n",
                    expert_bytes / d2d / 1e6, expert_bytes / h2d / 1e6, expert_bytes / d2h / 1e6);
        std::printf("   через PCIe за обмен идёт 2 × %zu МиБ = %.1f ГБ/с\n", expert_mib,
                    2.0 * expert_bytes / (d2d + h2d + d2h) / 1e6);

        // 2. a batch of independent exchanges, on one stream and on several
        for (int nstreams : {1, 2, 4}) {
            for (int n : {1, 2, 4, 8}) {
                if (n > slots) continue;
                t = Clock::now();
                for (int round = 0; round < 5; ++round) {
                    for (int i = 0; i < n; ++i) {
                        cudaStream_t s = streams[i % nstreams];
                        ck(cudaMemcpyAsync((uint8_t*) staging + (size_t) i * expert_bytes,
                                           (uint8_t*) vram_cold + (size_t) i * expert_bytes, expert_bytes,
                                           cudaMemcpyDeviceToDevice, s));
                        ck(cudaMemcpyAsync((uint8_t*) vram_target + (size_t) i * expert_bytes,
                                           ram + (size_t) i * 2 * expert_bytes, expert_bytes,
                                           cudaMemcpyHostToDevice, s));
                        ck(cudaMemcpyAsync(ram + (size_t) i * 2 * expert_bytes + expert_bytes,
                                           (uint8_t*) staging + (size_t) i * expert_bytes, expert_bytes,
                                           cudaMemcpyDeviceToHost, s));
                    }
                    for (int j = 0; j < nstreams; ++j) ck(cudaStreamSynchronize(streams[j]));
                }
                const double batch = ms_since(t) / 5;
                std::printf("пачка из %d независимых обменов на %d потоках: %.3f мс (%.3f мс на обмен; "
                            "PCIe несёт %.1f ГБ/с)\n",
                            n, nstreams, batch, batch / n, 2.0 * n * expert_bytes / batch / 1e6);
            }
        }

        // 3. the registration: adding a range to a live window, and dropping it
        {
            ResidentMemory other;
            void* spare = other.allocate(ram_bytes, "swap cost probe spare");
            t = Clock::now();
            ck(cudaHostRegister(spare, expert_bytes, 0));
            const double add_ms = ms_since(t);
            t = Clock::now();
            ck(cudaHostUnregister(spare));
            const double drop_ms = ms_since(t);
            std::printf("регистрация %zu МиБ: добавить %.3f мс, снять %.3f мс\n", expert_mib, add_ms, drop_ms);
        }
        t = Clock::now();
        dma.clear();
        const double clear_ms = ms_since(t);
        std::printf("снятие окна %zu МиБ: %.3f мс (%.0f МиБ/с)\n", ram_bytes / (1 << 20), clear_ms,
                    ram_bytes / (1 << 20) / (clear_ms / 1000.0));
        t = Clock::now();
        dma.prepare({{0, ram_bytes}});
        const double again = ms_since(t);
        std::printf("регистрация окна %zu МиБ заново: %.3f мс (%.0f МиБ/с); первая регистрация была %.3f мс\n",
                    ram_bytes / (1 << 20), again, ram_bytes / (1 << 20) / (again / 1000.0), register_ms);
    } catch (const std::exception& e) { std::fprintf(stderr, "swap cost probe: %s\n", e.what()); result = 1; }
    for (int j = 0; j < 4; ++j) { if (streams[j]) cudaStreamDestroy(streams[j]); }
    if (vram_cold) cudaFree(vram_cold);
    if (vram_target) cudaFree(vram_target);
    if (staging) cudaFree(staging);
    return result;
}
