// src/poly/read_chunk_test.cpp - Unit tests for read chunking, call splitting and buffer boundaries.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

constexpr int kStageRows = 32;
constexpr size_t kReadBytes = size_t(512) << 20;

struct TestEngineMath {
    int max_context = 32768;
    int prompt_batch = 1024;
    int read_chunk_tokens = 0;
    bool rows_vram = true;
    bool dump = false;
    int streams = 4;
    int n_embd = 4096;

    int rows = 1;
    int read_rows = 1;
    size_t v_res_bytes = 0;

    void compute() {
        if (read_chunk_tokens > 0 && prompt_batch > read_chunk_tokens) {
            prompt_batch = read_chunk_tokens;
        }
        rows = dump ? 1 : std::min(prompt_batch, max_context);
        const size_t chunk_limit = read_chunk_tokens > 0
            ? std::min((size_t) read_chunk_tokens, (size_t) max_context)
            : (size_t) max_context;
        const size_t per_token_bytes = (size_t) streams * n_embd * sizeof(float);
        const size_t most = rows_vram ? chunk_limit
            : std::min<size_t>(kReadBytes / per_token_bytes, chunk_limit);
        read_rows = rows < kStageRows ? rows : std::max(rows, (int) most / rows * rows);
        v_res_bytes = rows_vram ? (size_t) read_rows * streams * n_embd * sizeof(float) : 0;
    }
};

struct Call {
    int at, m, mark;
};

std::vector<Call> split_calls(int n, int rows, const std::vector<int>& marks) {
    std::vector<Call> calls;
    int at = 0, k = 0;
    const int n_marks = (int) marks.size();
    while (at < n) {
        const int m = std::min(rows, (k < n_marks ? marks[(size_t) k] : n) - at);
        const bool ends = k < n_marks && at + m == marks[(size_t) k];
        calls.push_back({at, m, ends ? k : -1});
        at += m;
        k += ends ? 1 : 0;
    }
    return calls;
}

void test_alloc_math() {
    // 1. DeepSeek V4 Flash with --read-chunk 4096 on 32k context
    {
        TestEngineMath t;
        t.read_chunk_tokens = 4096;
        t.compute();
        check(t.rows == 1024, "DeepSeek: rows is 1024");
        check(t.read_rows == 4096, "DeepSeek: read_rows is 4096");
        check(t.v_res_bytes == 268435456ULL, "DeepSeek: v_res is exactly 256 MiB");
    }

    // 2. Compatibility: default read_chunk_tokens == 0
    {
        TestEngineMath t;
        t.read_chunk_tokens = 0;
        t.compute();
        check(t.rows == 1024, "Default: rows is 1024");
        check(t.read_rows == 32768, "Default: read_rows is full 32768");
        check(t.v_res_bytes == 2147483648ULL, "Default: v_res is 2048 MiB");
    }

    // 3. User sets read_chunk smaller than prompt_batch: --prompt-batch 2048 --read-chunk 1024
    {
        TestEngineMath t;
        t.prompt_batch = 2048;
        t.read_chunk_tokens = 1024;
        t.compute();
        check(t.prompt_batch == 1024, "Capped prompt_batch at read_chunk");
        check(t.rows == 1024, "Effective rows is 1024");
        check(t.read_rows == 1024, "read_rows is 1024, not 2048");
        check(t.v_res_bytes == 67108864ULL, "v_res sized for 1024 tokens (64 MiB)");
    }

    // 4. Non-multiple read_chunk: 3500 with batch 1024
    {
        TestEngineMath t;
        t.read_chunk_tokens = 3500;
        t.compute();
        check(t.read_rows == 3072, "Rounded down to multiple of batch: 3072");
        check(t.v_res_bytes == 3072ULL * 4 * 4096 * sizeof(float), "v_res sized for 3072 tokens");
    }

    // 5. Small context: max_context 500 with batch 1024
    {
        TestEngineMath t;
        t.max_context = 500;
        t.read_chunk_tokens = 4096;
        t.compute();
        check(t.rows == 500, "Small context rows capped at 500");
        check(t.read_rows == 500, "Small context read_rows is 500");
    }

    // 6. Tiny context: max_context 16 (< kStageRows = 32)
    {
        TestEngineMath t;
        t.max_context = 16;
        t.read_chunk_tokens = 4096;
        t.compute();
        check(t.rows == 16, "Tiny context rows is 16");
        check(t.read_rows == 16, "Tiny context read_rows is 16");
        check(t.read_rows > 0, "Tiny context read_rows is positive");
    }

    // 7. Dump layers mode (rows = 1)
    {
        TestEngineMath t;
        t.dump = true;
        t.read_chunk_tokens = 4096;
        t.compute();
        check(t.rows == 1, "Dump mode rows is 1");
        check(t.read_rows == 1, "Dump mode read_rows is 1");
    }

    // 8. Host RAM path (!rows_vram)
    {
        TestEngineMath t;
        t.rows_vram = false;
        t.read_chunk_tokens = 4096;
        t.compute();
        check(t.read_rows == 4096, "Host RAM read_rows is 4096");
        check(t.v_res_bytes == 0, "No v_res allocation in host mode");
    }
}

void test_chunked_prompt_simulation(int prompt_len, int max_ctx, int prompt_batch, int chunk_tokens,
                                    const std::vector<int>& anchors) {
    TestEngineMath eng;
    eng.max_context = max_ctx;
    eng.prompt_batch = prompt_batch;
    eng.read_chunk_tokens = chunk_tokens;
    eng.compute();

    const size_t r_row = (size_t) eng.streams * eng.n_embd;
    const size_t v_res_floats = eng.v_res_bytes / sizeof(float);

    int i = 0;
    int tokens_covered = 0;

    while (i < prompt_len) {
        const bool last = (i == prompt_len - 1);
        const int room = std::min(prompt_len - 1 - i, eng.read_rows);
        const int m = last ? 1 : room;

        if (!last) {
            check(m > 0 && m <= eng.read_rows, "Chunk size within bounds");

            // Find anchors strictly inside (i, i + m)
            std::vector<int> inside_marks;
            for (int p : anchors) {
                if (p > i && p < i + m) {
                    inside_marks.push_back(p - i);
                }
            }

            std::vector<Call> calls = split_calls(m, eng.rows, inside_marks);
            int sum_m = 0;
            for (const Call& c : calls) {
                check(c.at >= 0, "call.at non-negative");
                check(c.m > 0 && c.m <= eng.rows, "call.m valid");
                check(c.at + c.m <= m, "call within chunk");

                // Check device buffer addressing
                const size_t float_end = (size_t) (c.at + c.m) * r_row;
                check(float_end <= v_res_floats, "No buffer overflow in v_res_");

                sum_m += c.m;
            }
            check(sum_m == m, "Sum of call lengths equals chunk length");
        } else {
            check(m == 1, "Last token is read alone");
        }

        i += m;
        tokens_covered += m;
    }
    check(tokens_covered == prompt_len, "All prompt tokens covered");
}

}  // namespace

int main() {
    test_alloc_math();

    // Verify non-multiple lengths, chunk boundaries, checkpoints
    test_chunked_prompt_simulation(4500, 32768, 1024, 4096, {1000, 2048, 4096, 4200});
    test_chunked_prompt_simulation(4096, 32768, 1024, 4096, {});
    test_chunked_prompt_simulation(4097, 32768, 1024, 4096, {4096});
    test_chunked_prompt_simulation(10000, 32768, 1024, 4096, {4096, 8192});
    test_chunked_prompt_simulation(17, 32768, 1024, 4096, {5, 10});
    test_chunked_prompt_simulation(3500, 32768, 1024, 2000, {1500});

    std::printf("poly_read_chunk_test: all %d checks PASSED\n", checks);
    return 0;
}
