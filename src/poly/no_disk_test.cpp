#include "no_disk.hpp"
#include "dma_ranges.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace strata::poly;
namespace {
int checks = 0;
void check(bool b) { ++checks; if (!b) { std::fprintf(stderr, "no_disk_test: failure at check %d\n", checks); std::exit(1); } }
template<class F> bool fails(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
template<class T> void put(std::ofstream& f, T n) { f.write((const char*) &n, sizeof n); }
}
int main() {
    // Only original source pages are registered. Adjacent/overlapping pages merge, including a partial last page.
    check(dma_ranges({}, 8193, 16384, 4096).empty());
    check(dma_ranges({{1, 1}, {4095, 2}, {8192, 1}}, 8193, 12288, 4096) ==
          std::vector<DmaRange>{{0, 12288}});
    check(dma_ranges({{8192, 10}, {1, 10}, {1, 10}}, 16384, 8192, 4096) ==
          std::vector<DmaRange>{{0, 4096}, {8192, 4096}});
    check(fails([] { dma_ranges({{8192, 2}}, 8193, 16384, 4096); }));
    check(fails([] { dma_ranges({{0, 8192}}, 8192, 4096, 4096); }));
    check(fails([] { dma_ranges({}, SIZE_MAX, SIZE_MAX, 4096); }));
    check(fails([] { dma_ranges({}, 8192, 8192, 3); }));
    for (size_t offset = 0; offset < 8192; offset += 31) {
        const auto ranges = dma_ranges({{offset, 1}}, 8192, 4096, 4096);
        check(ranges.size() == 1 && ranges[0].first == offset / 4096 * 4096 && ranges[0].second == 4096);
    }
    std::vector<ResidentExpert> e{{0,0,10,12,1}, {0,1,10,12,30}, {1,0,10,12,10}};
    // The card fills first: with room for all three, nothing stays in RAM even though RAM would hold them.
    auto p = plan_resident(e, 30, 1000);
    check(p.card_count == 3 && p.ram == 0 && p.vram == 36);
    p = plan_resident(e, 0, 1000);
    check(p.card_count == 3 && p.ram == 0);
    p = plan_resident(e, 10, 24);
    check(p.card_count == 2 && p.on_card[1] && p.on_card[2] && p.ram == 10 && p.vram == 24);
    // Room for one slot goes to the expert used most per byte (index 1), whatever the order of the vector.
    p = plan_resident(e, 20, 12);
    check(p.card_count == 1 && p.on_card[1] && p.ram == 20 && p.vram == 12);
    check(fails([&] { plan_resident(e, 9, 24); }));
    check(fails([&] { plan_resident(e, 20, 11); }));
    check(fails([&] { plan_resident({{0,0,0,1,0}}, 0, 0); }));
    check(fails([&] { resident_add(UINT64_MAX, 1); }));
    check(plan_resident({}, 0, 0).on_card.empty());
    // Unsupported GPU types stay on the host, while a lower ranked supported expert can make room for them.
    p = plan_resident({{0,0,10,0,100}, {1,0,10,12,1}}, 10, 12);
    check(!p.on_card[0] && p.on_card[1]);
    check(fails([&] { plan_resident({{0,0,10,0,100}}, 9, 100); }));
    for (uint64_t r = 0; r <= 30; ++r) for (uint64_t v = 0; v <= 40; ++v) {
        const uint64_t on_card = std::min<uint64_t>(3, v / 12), need = (3 - on_card) * 10;
        if (need > r) check(fails([&] { plan_resident(e, r, v); }));
        else { const auto q = plan_resident(e, r, v); check(q.card_count == on_card && q.ram == need &&
                                                             q.vram == on_card * 12 && q.ram <= r && q.vram <= v); }
    }
    // A real tiny GGUF: the new reader and the unchanged mapping see the same metadata and payload.
    const auto file = std::filesystem::temp_directory_path() /
        ("poly-nodisk-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    const float values[] = {1, -2, 3, 4};
    {
        std::ofstream f(file, std::ios::binary);
        put<uint32_t>(f, 0x46554747); put<uint32_t>(f, 3); put<uint64_t>(f, 1); put<uint64_t>(f, 0);
        put<uint64_t>(f, 6); f.write("weight", 6); put<uint32_t>(f, 1); put<uint64_t>(f, 4);
        put<uint32_t>(f, 0); put<uint64_t>(f, 0);
        while ((size_t) f.tellp() % 32) f.put(0);
        f.write((const char*) values, sizeof values);
    }
    {
        strata::GgufFile mapped(file.string()), direct(file.string(), false);
        check(mapped.mapped() && !direct.mapped());
        check(fails([&] { direct.tensor_data(*direct.find("weight")); }));
        check(mapped.data_start() == direct.data_start() && direct.tensors().size() == 1);
        check(mapped.find("weight")->shape == direct.find("weight")->shape);
        float read[4] = {};
        direct.read_at(direct.data_start(), read, sizeof read);
        check(std::memcmp(read, values, sizeof read) == 0);
        check(std::memcmp(mapped.tensor_data(*mapped.find("weight")), read, sizeof read) == 0);
        direct.read_at(direct.data_start() + 4, read, 4);
        check(read[0] == -2);
        check(fails([&] { direct.read_at(direct.file_size() - 1, read, 4); }));
        direct.close_data();
        check(fails([&] { direct.read_at(direct.data_start(), read, 4); }));
        check(direct.find("weight") != nullptr);
        direct.close_data(); // destruction and an explicit close are safe together
    }
    std::filesystem::remove(file);
    std::printf("no_disk_test: %d checks passed\n", checks);
}
