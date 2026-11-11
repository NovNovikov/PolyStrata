// By hand: disk_probe FILE [GiB] [MODE]. How fast the disk reads the model, and what the 4 KiB fault path costs.
//
// The mmap prefill read 129.87 GiB in 203.4 s through 34 506 662 page faults: 3.86 KiB per fault, about
// 1.1 GiB/s.  This reads the same file through the project's own read_at() - the path a pread-based engine
// would use - with big blocks and with 4 KiB blocks, so the difference between "the disk is slow" and "the
// read pattern is slow" is measured rather than guessed.
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace strata;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: disk_probe FILE [GiB] [seq|scattered]\n"); return 2; }
    const std::string path = argv[1];
    const size_t want = (argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 4) << 30;
    const std::string mode = argc >= 4 ? argv[3] : "seq";
    try {
        // The reader with no mapping: read_at() is then the only way in, which is the path being measured.
        GgufFile file(path, false);
        std::printf("файл: %s, %.1f ГиБ\n", path.c_str(), double(file.file_size()) / (1 << 30));
        const size_t block = mode == "seq" ? (8u << 20) : 4096;
        const size_t step = mode == "scattered" ? (1u << 20) : block;
        std::vector<uint8_t> buf(block);
        size_t got = 0, off = (argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 1ull) << 30;
        const auto start = Clock::now();
        while (got < want && off + block < file.file_size()) {
            file.read_at(off, buf.data(), block);
            got += block;
            off += step;
        }
        const double secs = std::chrono::duration<double>(Clock::now() - start).count();
        std::printf("%s блоками по %zu Б: %.2f ГиБ/с (%.2f с, %zu МБ\n", mode == "seq" ? "последовательное" : (mode == "tiny" ? "4 КиБ подряд" : "разбросанное"),
                    block, double(got) / secs / (1 << 30), secs, got >> 20);
    } catch (const std::exception& e) { std::fprintf(stderr, "disk_probe: %s\n", e.what()); return 1; }
    return 0;
}
