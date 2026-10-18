#include "no_disk.hpp"
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
    std::vector<ResidentExpert> e{{0,0,10,12,1}, {0,1,10,12,30}, {1,0,10,12,10}};
    auto p = plan_resident(e, 30, 1000);
    check(p.card_count == 0 && p.ram == 30 && p.vram == 0);
    p = plan_resident(e, 20, 1000);
    check(p.card_count == 1 && p.on_card[1] && p.ram == 20 && p.vram == 12);
    p = plan_resident(e, 10, 24);
    check(p.card_count == 2 && p.on_card[1] && p.on_card[2]);
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
        const uint64_t need = (30 - r + 9) / 10;
        if (v < need * 12) check(fails([&] { plan_resident(e, r, v); }));
        else { const auto q = plan_resident(e, r, v); check(q.card_count == need && q.ram <= r && q.vram <= v); }
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
