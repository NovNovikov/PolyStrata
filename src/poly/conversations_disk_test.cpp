// src/poly/conversations_disk_test.cpp - the session files: a parked conversation leaves the host memory, comes
// back across an engine restart (a new Conversations over a fresh engine), and its state is bit for bit what a
// prompt reread from the first token would give.
//
//   g++ -std=c++20 src/poly/conversations_disk_test.cpp -o conversations_disk_test && ./conversations_disk_test
#include "conversations.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

using strata::poly::CacheOptions;
using strata::poly::Conversations;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

int files_in(const std::filesystem::path& dir) {
    int files = 0;
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec))
        if (de.path().extension() == ".bin") ++files;
    return files;
}

uint64_t mix(uint64_t h, int32_t t) { return (h ^ (uint64_t) (uint32_t) t) * 0x9E3779B97F4A7C15ull + 0x7F4A7C15ull; }

// The same engine conversations_test.cpp checks with: the recurrent state is a hash of the tokens read, and every
// token leaves two rows.
struct Fake {
    static constexpr int64_t ctx = 1 << 16;
    uint64_t h = 1;
    int64_t at = 0;
    size_t state = 8;
    std::vector<uint64_t> a = std::vector<uint64_t>(ctx), b = std::vector<uint64_t>(ctx);

    void reset() { h = 1, at = 0; }
    void eval(const int32_t* t, int64_t n) {
        for (int64_t i = 0; i < n; ++i, ++at) {
            a[(size_t) at] = h = mix(h, t[i]);
            b[(size_t) at] = (uint64_t) t[i];
        }
    }
    size_t state_bytes() const { return state; }
    void save_state(uint8_t* dst) const { std::memcpy(dst, &h, 8); }
    void load_state(const uint8_t* src, int64_t pos) { std::memcpy(&h, src, 8), at = pos; }
    size_t row_bytes() const { return 16; }
    void save_rows(int64_t first, int64_t n, uint8_t* dst) const {
        std::memcpy(dst, &a[(size_t) first], (size_t) n * 8);
        std::memcpy(dst + n * 8, &b[(size_t) first], (size_t) n * 8);
    }
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
        std::memcpy(&a[(size_t) first], src + (first - from) * 8, (size_t) n * 8);
        std::memcpy(&b[(size_t) first], src + count * 8 + (first - from) * 8, (size_t) n * 8);
    }

    bool holds(const std::vector<int32_t>& ids) const {
        uint64_t x = 1;
        for (size_t i = 0; i < ids.size(); ++i)
            if (a[i] != (x = mix(x, ids[i])) || b[i] != (uint64_t) ids[i]) return false;
        return at == (int64_t) ids.size() && h == x;
    }
};

std::vector<int32_t> text(int seed, int n) {
    std::vector<int32_t> v((size_t) n);
    std::mt19937 r((unsigned) seed);
    for (int32_t& t : v) t = (int32_t) (r() % 50000);
    return v;
}

// Reads `ids` into the engine from where resume() took it up, the way the server drives the reads.
void go_on(Fake& e, Conversations<Fake>& kept, const std::vector<int32_t>& ids, int64_t from) {
    kept.turn_at(-1);
    for (int64_t i = from; i < (int64_t) ids.size();) {
        const bool last = i == (int64_t) ids.size() - 1;
        const int64_t m = last ? 1 : kept.until_checkpoint(i, 100);
        kept.before(i, last);
        e.eval(ids.data() + i, m);
        kept.read(ids.data() + i, m);
        i += m;
    }
}

CacheOptions options(const std::filesystem::path& dir, int64_t park_min = 64) {
    CacheOptions o;
    o.checkpoints = 2;
    o.every = 100;
    o.park_slots = 4;
    o.park_min = park_min;
    o.session_dir = dir;
    o.session_max_mib = 64;
    return o;
}

}  // namespace

int main() {
    // runs of this test leave no files behind: earlier ones carried over (std::rand() seeds the same every run)
    const std::chrono::steady_clock::rep tick = std::chrono::steady_clock::now().time_since_epoch().count();
    for (const auto& de : std::filesystem::directory_iterator(std::filesystem::temp_directory_path())) {
        const std::string n = de.path().filename().string();
        if (de.is_directory() && n.rfind("poly-sessions-", 0) == 0) {
            std::error_code wipe;
            std::filesystem::remove_all(de.path(), wipe);
        }
    }
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("poly-sessions-" + std::to_string(tick));
    std::filesystem::create_directories(dir);

    const std::vector<int32_t> base = text(11, 300);

    // A conversation is parked to a file: the file is there, and no host memory stays with it.
    {
        Fake e;
        CacheOptions o1 = options(dir);
        o1.log = stderr;
        Conversations<Fake> kept(e, o1);
        std::vector<int32_t> ids = base;
        const int64_t from = kept.resume(ids);
        go_on(e, kept, ids, from);
        kept.prompt_was((int64_t) ids.size());
        const std::vector<int32_t> other = text(12, 100);
        kept.resume(other);   // disjoint: the first conversation parks
        go_on(e, kept, other, 0);
        check(files_in(dir) == 1, "the parked conversation left one session file");
        check(kept.parked_bytes() == 0, "a conversation parked to a file holds no host memory");
    }

    // An engine restart: the file comes back as a parked conversation, and the engine holds exactly what its last
    // checkpoint covers.
    {
        Fake e;
        Conversations<Fake> kept(e, options(dir));
        const std::vector<int32_t> ids = base;
        const int64_t from = kept.resume(ids);
        check(from > 0 && from < (int64_t) ids.size(), "the file's conversation is come back to before its last token");
        const std::vector<int32_t> seen(ids.begin(), ids.begin() + (std::ptrdiff_t) from);
        check(e.holds(seen), "the engine holds exactly the tokens the file's checkpoint covers");
    }

    // Going on from there, the result is bit for bit what a full reread would give.
    {
        Fake e;
        Conversations<Fake> kept(e, options(dir));
        std::vector<int32_t> ids = base;
        const int64_t from = kept.resume(ids);
        check(from > 0, "the restarted conversation goes on from the file's state");
        go_on(e, kept, ids, from);
        check(e.holds(ids), "the prompt read on from the file holds bit for bit the reread's result");
    }

    // A fork before the file's deepest checkpoint: the file holds the whole chain, so an earlier checkpoint is
    // come back to too.
    {
        Fake e;
        Conversations<Fake> kept(e, options(dir));
        std::vector<int32_t> ids(base.begin(), base.begin() + 250);
        const auto tail = text(31, 150);
        ids.insert(ids.end(), tail.begin(), tail.end());
        const int64_t from = kept.resume(ids);
        check(from == 200 || from == 100, "the fork lands on one of the file's earlier checkpoints");
        go_on(e, kept, ids, from);
        check(e.holds(ids), "the branch read on from the file's checkpoint is bit for bit the reread's result");
    }

    // Another engine (a different state size): the file is not this engine's any more, and the conversation is
    // reread instead.
    {
        Fake e;
        e.state = 16;
        Conversations<Fake> kept(e, options(dir));
        const std::vector<int32_t> other = text(41, 100);
        const int64_t from = kept.resume(other);
        check(from == 0, "a file of another engine is not come back to");
        check(files_in(dir) == 0, "the file of another engine is taken away");
    }

    // The files take session_max_mib at most: the least recently used goes.  The conversations are long enough
    // that two of them together are past the cap.
    {
        CacheOptions o = options(dir, 8);
        o.session_max_mib = 1;
        Fake e;
        Conversations<Fake> kept(e, o);
        for (int round = 0; round < 3; ++round) {
            std::vector<int32_t> ids = text(50 + round, 40000);
            const int64_t n = (int64_t) ids.size(), from = kept.resume(ids);
            go_on(e, kept, ids, from);
            kept.prompt_was(n);
            const std::vector<int32_t> switch_to = text(60 + round, 50);
            kept.resume(switch_to);   // the prompt parks: its file goes to the dir
        }
        check(files_in(dir) <= 2, "the cap leaves only the newest files");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf("conversations_disk_test: %d checks passed\n", checks);
    return 0;
}
