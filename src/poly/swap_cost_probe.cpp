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

        // the RAM arena, as big as the window: a source and a target per exchange at its start, the rest of the
        // window is the room the engine's arena has around them
        ResidentMemory memory;
        const size_t ram_bytes = window_mib << 20;
        const size_t used_bytes = expert_bytes * slots * 2;
        if (used_bytes > ram_bytes) { std::fprintf(stderr, "the window of %llu MiB is smaller than %d exchanges\n", (unsigned long long) window_mib, slots); return 2; }
        auto* ram = (uint8_t*) memory.allocate(ram_bytes, "swap cost probe");
        std::memset(ram, 0x5A, ram_bytes);
        // the window's pages are registered, as the engine's are: the D2H writes into a registered page
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

        // 3. the full transaction: the three copies between the same two places, the wait for the readers, and the
        //    publication of the owners.  This is what the planner's budget is made of, not the copies alone.
        {
            // the two places of every exchange: a VRAM place and a RAM place.  The places do not move; what moves
            // is what lives in them, so the addresses stay the same from round to round.
            std::vector<void*> vram_place((size_t) slots), ram_place((size_t) slots);
            std::vector<int> owner_vram((size_t) slots), owner_ram((size_t) slots);
            for (int i = 0; i < slots; ++i) {
                vram_place[(size_t) i] = (uint8_t*) vram_cold + (size_t) i * expert_bytes;
                ram_place[(size_t) i] = ram + (size_t) i * 2 * expert_bytes;
                owner_vram[(size_t) i] = 1000 + i;   // the cold expert
                owner_ram[(size_t) i] = 2000 + i;    // the hot expert
            }
            const int rounds_tx = 20;
            t = Clock::now();
            for (int round = 0; round < rounds_tx; ++round) {
                for (int i = 0; i < slots; ++i) {
                    cudaStream_t s = streams[i % 4];
                    // the readers of the old contents are finished before the swap starts
                    ck(cudaStreamSynchronize(s));
                    ck(cudaMemcpyAsync((uint8_t*) staging + (size_t) i * expert_bytes, vram_place[(size_t) i],
                                       expert_bytes, cudaMemcpyDeviceToDevice, s));
                    ck(cudaMemcpyAsync(vram_place[(size_t) i], ram_place[(size_t) i], expert_bytes,
                                       cudaMemcpyHostToDevice, s));
                    ck(cudaMemcpyAsync(ram_place[(size_t) i], (uint8_t*) staging + (size_t) i * expert_bytes,
                                       expert_bytes, cudaMemcpyDeviceToHost, s));
                }
                for (int j = 0; j < 4; ++j) ck(cudaStreamSynchronize(streams[j]));
                // the publication of the owners: the experts swap places, the places stay
                for (int i = 0; i < slots; ++i) std::swap(owner_vram[(size_t) i], owner_ram[(size_t) i]);
            }
            const double tx = ms_since(t) / rounds_tx;
            std::printf("полная транзакция %d обменов (те же два места, ожидание читателей, публикация "
                        "владельцев): %.3f мс на обмен\n", slots, tx / slots);
        }

        // 4. the registration: adding a range to a live window, and dropping it
        {
            // a second arena, so the range measured is not one the live window already holds
            ResidentMemory other;
            void* spare = other.allocate(expert_bytes, "swap cost probe spare");
            t = Clock::now();
            ck(cudaHostRegister(spare, expert_bytes, 0));
            const double add_ms = ms_since(t);
            t = Clock::now();
            ck(cudaHostUnregister(spare));
            const double drop_ms = ms_since(t);
            std::printf("регистрация %zu МиБ: добавить %.3f мс, снять %.3f мс\n", expert_mib, add_ms, drop_ms);
        }        t = Clock::now();
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
    catch (...) { std::fprintf(stderr, "swap cost probe: unknown error\n"); result = 1; }
    for (int j = 0; j < 4; ++j) { if (streams[j]) cudaStreamDestroy(streams[j]); }
    if (vram_cold) cudaFree(vram_cold);
    if (vram_target) cudaFree(vram_target);
    if (staging) cudaFree(staging);
    return result;
}
