// gfile_test.cpp - session file round trip with REALISTIC sizes: a blob part and a fill (engine tail)
// part, row_bytes ~7 KB like the real GLM engine, then read back and compare bit for bit.
//
//   g++ -std=c++20 -I<repo>/src/poly gfile_test.cpp -o gfile_test && ./gfile_test
#include "session_file.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

using namespace strata::poly;

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::fflush(stderr);
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    std::filesystem::path dir = argc > 1 ? argv[1] : std::filesystem::temp_directory_path() / "gfile-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const int64_t rb = 7697;            // real GLM-like row bytes (not a multiple of anything)
    const int64_t sb = 393216;          // state bytes per mark
    const int n_seq = 32768;            // rows the conversation holds
    const int n_marks = 4;
    const int mark_at = 32000;          // the deepest checkpoint

    // the truth payloads: every mark carries its own sb bytes of state
    std::vector<uint8_t> state((size_t) sb * n_marks);
    for (size_t i = 0; i < state.size(); ++i) state[i] = (uint8_t) (i * 7 + (i >> 9));
    std::vector<uint8_t> rows((size_t) mark_at * rb);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (uint8_t) (i * 13 + (i >> 7));

    // marks: each mark carries exactly sb bytes of state, the way the engine's save_state writes it
    std::vector<std::pair<int64_t, SessionFile::SessionPart>> parts;
    const int64_t mark_pos[] = {8192, 16384, 24000, mark_at};
    for (int m = 0; m < n_marks; ++m)
        parts.push_back({mark_pos[m], SessionFile::part_of(state.data() + (size_t) m * sb, (size_t) sb)});

    // rows: a blob part for [0, 100) and the tail [100, mark_at) read straight from the "engine"
    std::vector<SessionFile::StreamRows> stream;
    std::vector<uint8_t> tail = rows;
    (void) tail;
    stream.push_back({0, 100, 100, SessionFile::part_of(rows.data(), (size_t) 100 * rb)});
    SessionFile::SessionPart tail_part;
    tail_part.bytes = (uint64_t) (mark_at - 100) * rb;
    tail_part.align = (uint64_t) rb;
    tail_part.fill = [&](void* dst, size_t off, size_t n) {
        std::memcpy(dst, rows.data() + 100 * rb + off, n);
        return true;
    };
    stream.push_back({100, mark_at - 100, mark_at - 100, tail_part});

    std::vector<int32_t> seq((size_t) mark_at);
    for (int i = 0; i < mark_at; ++i) seq[(size_t) i] = (int32_t) (i * 2654435761u);
    SessionFile::WStat st;
    const std::string name = "gfile-roundtrip.bin";
    std::fprintf(stderr, "TRACE: writing %d rows, rb=%lld, marks=%d\n", n_marks, (long long) rb, n_marks);
    std::fflush(stderr);
    const bool ok = SessionFile::write_stream(dir, name, 5000, sb, rb, seq, parts, stream, &st, {});
    check(ok, "the write succeeded");
    check(st.bytes > 0, "the write made bytes");
    std::fprintf(stderr, "TRACE: written %zu bytes\n", st.bytes);
    std::fflush(stderr);

    const auto view = SessionFile::read(dir / name);
    std::fprintf(stderr, "TRACE: read done, marks=%zu rows=%zu\n", view ? view->marks.size() : (size_t) -1,
                 view ? view->rows.size() : (size_t) -1);
    std::fflush(stderr);
    check(view && !view->seq.empty(), "the read came back");
    check((int) view->seq.size() == mark_at, "the seq came back whole");
    check(view->sb == (size_t) sb && view->rb == (size_t) rb, "the header's sizes came back");
    check((int) view->marks.size() == n_marks, "the marks came back");
    for (int m = 0; m < n_marks; ++m) {
        check(view->marks[(size_t) m].pos == mark_pos[m], "a mark holds its place");
        check((int) view->marks[(size_t) m].state.size() == sb, "a mark holds its state bytes");
        if (view->marks[(size_t) m].state.size() == (size_t) sb)
            check(std::memcmp(view->marks[(size_t) m].state.data(), state.data() + (size_t) m * sb, (size_t) sb) == 0,
                  "a mark's state came back bit for bit");
    }
    check((int) view->rows.size() == 2, "the rows came back in two");
    for (const auto& r : view->rows) {
        const size_t want = (size_t) r.count * rb;
        check(r.data.size() == want, "a rows span holds its bytes");
        check(std::memcmp(r.data.data(), rows.data() + (size_t) r.first * rb, want) == 0,
              "a rows span came back bit for bit");
    }
    std::filesystem::remove_all(dir);
    std::printf("gfile_test: %d checks passed\n", checks);
    return 0;
}
