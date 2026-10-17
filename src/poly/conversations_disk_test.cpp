// src/poly/conversations_disk_test.cpp - the session files: a parked conversation leaves the host memory, comes
// back across an engine restart (a new Conversations over a fresh engine), and its state is bit for bit what a
// prompt reread from the first token would give.
//
//   g++ -std=c++20 src/poly/conversations_disk_test.cpp -o conversations_disk_test && ./conversations_disk_test
#include "conversations.hpp"
#include "session_file.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    static constexpr int64_t ctx = 1 << 19;
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
        const int64_t m = last ? 1 : std::min(kept.until_checkpoint(i, 100), (int64_t) ids.size() - i);
        kept.before(i, last);
        e.eval(ids.data() + i, m);
        kept.read(ids.data() + i, m);
        i += m;
    }
}

// One read can cross several checkpoints, as the server's text-only prefill does.
void go_on_batch(Fake& e, Conversations<Fake>& kept, const std::vector<int32_t>& ids, int64_t from) {
    kept.turn_at(-1);
    std::vector<std::pair<int64_t, uint8_t*>> due;
    const int64_t n = (int64_t) ids.size();
    for (int64_t i = from; i < n;) {
        const bool last = i == n - 1;
        const int64_t m = last ? 1 : std::min<int64_t>(65536, n - 1 - i);
        kept.before(i, last);
        kept.inside(i, m, due);
        int64_t done = 0;
        for (const auto& d : due) {
            e.eval(ids.data() + i + done, d.first - done);
            done = d.first;
            if (d.second) e.save_state(d.second);
        }
        e.eval(ids.data() + i + done, m - done);
        kept.read(ids.data() + i, m);
        i += m;
    }
    kept.prompt_was(n);
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
    // Each run owns one unique directory; do not remove another run's files from the shared temp directory.
    const std::chrono::steady_clock::rep tick = std::chrono::steady_clock::now().time_since_epoch().count();
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

    // A follow-up that starts like the current conversation but diverges before its end: the conversation parks to
    // a file and goes on in a copy, which keeps its own checkpoints (the file-parked slot holds none of them), so
    // the engine has a state to come back to.  What the copy read on gives is bit for bit a reread's result.
    {
        const std::filesystem::path own = dir / ("poly-sessions-frk-" + std::to_string(tick));
        std::filesystem::create_directories(own);
        Fake e;
        CacheOptions o1 = options(own);
        o1.checkpoints = 8;   // an early checkpoint survives, as --prompt-cache 6 keeps one in a real run
        Conversations<Fake> kept(e, o1);
        std::vector<int32_t> ids = base;
        int64_t from = kept.resume(ids);
        check(from == 0, "the first read starts from the first token");
        go_on(e, kept, ids, from);
        kept.prompt_was((int64_t) ids.size());
        const std::vector<int32_t> said = text(55, 20);   // the answer, as the server reads it
        kept.turn_at(-1);
        for (size_t k = 0; k < said.size(); ++k) {
            const bool last = k + 1 == said.size();
            kept.before(300 + (int64_t) k, last);
            e.eval(&said[k], 1);
            kept.read(&said[k], 1);
        }
        std::vector<int32_t> mine = ids;
        mine.insert(mine.end(), said.begin(), said.end());   // the client's next request holds the whole chat
        mine[250] = (int32_t) 49999;   // the re-render differs inside the prompt, so the copy is a branch
        const auto tail = text(56, 15);
        mine.insert(mine.end(), tail.begin(), tail.end());
        from = kept.resume(mine);
        check(files_in(own) == 1, "the branched-away conversation left one session file");
        check(from == 200, "the copy goes on from its last checkpoint, not the diverged tail");
        go_on(e, kept, mine, from);
        check(e.holds(mine), "the copy read on from its own checkpoint is bit for bit the reread's result");
        std::filesystem::remove_all(own);
    }

    // More than the writer's 4 MiB staging buffer. save_rows() is layer-major (all hashes, then all tokens),
    // so concatenating separately saved chunks is NOT the bytes of one save_rows() over the whole span.
    {
        const std::filesystem::path own = dir / "large";
        std::filesystem::create_directories(own);
        CacheOptions o = options(own);
        o.checkpoints = 6;
        o.every = 16384;
        const std::vector<int32_t> original = text(71, 300013);
        Fake e;
        Conversations<Fake> kept(e, o);
        go_on_batch(e, kept, original, kept.resume(original));
        check(e.holds(original), "the large original is a clean full read");
        const std::vector<int32_t> ping = text(72, 101);
        go_on_batch(e, kept, ping, kept.resume(ping));
        const int64_t from = kept.resume(original);
        check(from == (int64_t) original.size() - 1, "a large parked prompt resumes at its last checkpoint");
        check(e.holds(std::vector<int32_t>(original.begin(), original.begin() + from)),
              "all rows of a multi-chunk park restore in their original layer order");
        go_on_batch(e, kept, original, from);
        check(e.holds(original), "a hot engine reuses the large disk session exactly");

        // Branch while the original is active, then return to it through a different parked conversation.
        std::vector<int32_t> fork(original.begin(), original.begin() + 180000);
        const auto tail = text(73, 500);
        fork.insert(fork.end(), tail.begin(), tail.end());
        const int64_t split = kept.resume(fork);
        check(split > 0 && split < 180000, "the large fork restores an interior checkpoint");
        go_on_batch(e, kept, fork, split);
        check(e.holds(fork), "the large branch has the rows and state of a full reread");
        go_on_batch(e, kept, ping, kept.resume(ping));
        go_on_batch(e, kept, original, kept.resume(original));
        check(e.holds(original), "returning from the branch preserves the original's rows");

        // A restart also loads each opaque rows record using its own count, including a partial last record.
        go_on_batch(e, kept, ping, kept.resume(ping));
        Fake cold;
        Conversations<Fake> restarted(cold, o);
        go_on_batch(cold, restarted, fork, restarted.resume(fork));
        check(cold.holds(fork), "a restarted engine forks from the large file exactly");
        go_on_batch(cold, restarted, original, restarted.resume(original));
        check(cold.holds(original), "a restarted engine restores the original after the fork");

        const std::filesystem::path store = own / "save";
        size_t bytes = 0;
        bool published = false;
        check(restarted.snapshot(store, "large.bin", {}, bytes, published) == Conversations<Fake>::SOut::saved,
              "SAVE writes the materialized multi-record session");
        Fake loaded;
        Conversations<Fake> restored(loaded, o);
        std::string err;
        int64_t tokens = 0;
        bool touched = false;
        check(restored.sideload(store / "large.bin", 0, err, tokens, bytes, touched) ==
                  Conversations<Fake>::SLoad::ok && loaded.holds(original),
              "RESTORE mounts every record of the large saved session exactly");
        // Old files do not say whether their opaque rows were captured as one block or wrongly concatenated.
        std::filesystem::copy_file(store / "large.bin", own / "legacy.bin");
        {
            std::fstream legacy(own / "legacy.bin", std::ios::in | std::ios::out | std::ios::binary);
            const uint32_t version = 1;
            legacy.seekp(4);
            legacy.write((const char*) &version, sizeof version);
        }
        check(restored.sideload(own / "legacy.bin", 0, err, tokens, bytes, touched) ==
                  Conversations<Fake>::SLoad::unreadable && !touched && loaded.holds(original),
              "a legacy file is refused before touching the live engine");
        check(!strata::poly::SessionFile::read_head(own / "legacy.bin"),
              "the startup scan does not admit legacy session files");
        go_on_batch(loaded, restored, fork, restored.resume(fork));
        check(loaded.holds(fork), "an interior fork after RESTORE also matches a full reread");
        std::filesystem::remove_all(own);
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

    // SAVE <path>: the whole conversation to a file the server names, rows to the engine's end and the state
    // there; a RESTORE of it in a fresh engine holds the conversation bit for bit and goes on from it.
    {
        Fake e;
        Conversations<Fake> kept(e, options(dir));
        std::vector<int32_t> ids = base;
        const int64_t from = kept.resume(ids);
        go_on(e, kept, ids, from);
        kept.prompt_was((int64_t) ids.size());
        const std::filesystem::path store = dir / "store";
        std::filesystem::create_directories(store);
        const std::string name = "a1b2c3d4e5f60718-300.bin";
        size_t bytes = 0;
        bool published = false;
        bool read_progress = false;
        const auto out = kept.snapshot(store, name,
                                       [&read_progress](uint64_t, uint64_t) { read_progress = true; },
                                       bytes, published);
        check(out == Conversations<Fake>::SOut::saved, "the save of a whole conversation writes its file");
        check(published && bytes > 0 && bytes < (size_t) (1 << 20), "the save published a file of its payloads");
        check(read_progress, "the write called its progress");
        const std::optional<strata::poly::SessionFile::Head> h = strata::poly::SessionFile::read_head(store / name);
        check(h && h->seq == ids, "the file's head carries the whole conversation");
        check(h && h->bytes == bytes, "the scanned size includes tokens, checkpoint payloads and record headers");
        check((size_t) bytes == (size_t) std::filesystem::file_size(store / name), "the save's count is the file's");
        check(e.holds(ids), "the save leaves the engine's conversation alone");
        // a save of an empty cache has nothing to save
        Fake e2;
        Conversations<Fake> fresh(e2, options(dir));
        size_t nothing = 0;
        const auto out2 = fresh.snapshot(store, "empty.bin", {}, nothing, published);
        check(out2 == Conversations<Fake>::SOut::empty, "nothing complete to save is refused");
        // RESTORE: a new engine takes the file whole and continues it
        Fake e3;
        Conversations<Fake> brought(e3, options(dir));   // a fresh engine has none of the conversation
        std::string err;
        int64_t tokens = 0;
        size_t rbytes = 0;
        bool touched = false;
        const auto got = brought.sideload(store / name, 2560, err, tokens, rbytes, touched);
        check(got == Conversations<Fake>::SLoad::ok, "the restore reads the file's conversation in");
        check(tokens == (int64_t) ids.size() && rbytes == bytes, "the restore holds the file's counts");
        check(touched && e3.holds(ids), "after the restore the engine holds the conversation bit for bit");
        // the refusal kinds the server forwards: unknown file, this engine's geometry gone
        const auto gone = brought.sideload(store / "no-such-file.bin", 2560, err, tokens, rbytes, touched);
        check(gone == Conversations<Fake>::SLoad::missing, "an unknown file is refused as missing: the server's invalid");
        Fake e4;
        e4.state = 16;   // another engine: the same file is not its format any more
        Conversations<Fake> stranger(e4, options(dir));
        const auto alien = stranger.sideload(store / name, 2560, err, tokens, rbytes, touched);
        check(alien == Conversations<Fake>::SLoad::unreadable && !err.empty(), "a file of another engine's shapes is refused");
        // A restore replaces the conversation that was in the engine
        Fake e5;
        Conversations<Fake> second(e5, options(dir));
        std::vector<int32_t> other = text(91, 120);
        const int64_t other_from = second.resume(other);
        go_on(e5, second, other, other_from);
        second.prompt_was((int64_t) other.size());
        int64_t second_tokens = 0;
        const auto took = second.sideload(store / name, 2560, err, second_tokens, rbytes, touched);
        check(took == Conversations<Fake>::SLoad::ok && second_tokens == (int64_t) ids.size(),
              "the restore replaces the conversation that was in the engine");
        check(second.seq() == ids && e5.holds(ids), "the engine and the cache both hold the file's conversation");
        // and the conversation continues from the file without rereading a token
        std::vector<int32_t> more = ids;
        const auto extra = text(99, 40);
        more.insert(more.end(), extra.begin(), extra.end());
        const int64_t more_from = second.resume(more);
        check(more_from == (int64_t) ids.size(), "the restored conversation is where it left off");
        go_on(e5, second, more, more_from);
        check(e5.holds(more), "the conversation read on from the restore is bit for bit the reread's result");
        // the file's payloads are in the view and in their blobs at once while the restore mounts: a host that
        // cannot take both gets the refusal before the engine is touched
        {
            Fake e6;
            CacheOptions o2 = options(dir);
            const size_t whole = (size_t) std::filesystem::file_size(store / name);
            o2.admit = [whole](size_t asked) { return asked < 2 * whole; };   // one file, not two
            Conversations<Fake> scarce(e6, o2);
            std::string err2;
            bool freeze = false;
            const auto short_of = scarce.sideload(store / name, 2560, err2, tokens, rbytes, freeze);
            check(short_of == Conversations<Fake>::SLoad::no_ram, "a host two files short is refused before the engine");
            check(!freeze, "a refusal of the mount touches no engine state");
        }
        // A memory that breaks at the third blob of a restore, two states building fine then stopping: the
        // refusal comes before the engine is touched, the file stays whole, and the same file comes in after.
        {
            Fake e6;
            CacheOptions o2 = options(dir);
            o2.fail_blob_at = 3;
            Conversations<Fake> scarce(e6, o2);
            std::string err2;
            bool freeze = false;
            const auto mid = scarce.sideload(store / name, 2560, err2, tokens, rbytes, freeze);
            check(mid == Conversations<Fake>::SLoad::no_ram, "a memory breaking inside the blobs is refused");
            check(!freeze, "a refusal inside the blobs touches no engine state");
            Conversations<Fake> again(e6, options(dir));
            std::string err3;
            bool freeze2 = false;
            const auto retried = again.sideload(store / name, 2560, err3, tokens, rbytes, freeze2);
            check(retried == Conversations<Fake>::SLoad::ok && tokens == (int64_t) ids.size() && freeze2,
                  "with whole memory the same file comes in whole");
        }
        // A memory that breaks inside the materialize of a scanned file: the slot goes whole, the engine holds
        // nothing of a half come-in, and the prompt is reread.
        {
            // a file of its own, at the top of the session dir for the scan to take
            std::vector<int32_t> mine = base;
            {
                Fake w;
                Conversations<Fake> warden(w, options(dir));
                const int64_t w_from = warden.resume(mine);
                go_on(w, warden, mine, w_from);
                warden.prompt_was((int64_t) mine.size());
                size_t wbytes = 0;
                bool wpublic = false;
                check(warden.snapshot(dir, "materialize-me.bin", {}, wbytes, wpublic) ==
                          Conversations<Fake>::SOut::saved,
                      "the materialize's own file is written");
            }
            Fake e7;
            CacheOptions o3 = options(dir);
            o3.fail_blob_at = 2;
            Conversations<Fake> scarce(e7, o3);   // the scan put the newest file into a slot
            check(scarce.parked() == 1, "the file rests in a slot as before");
            const std::vector<int32_t> ids = text(7, 100);
            const int64_t from = scarce.resume(ids);
            check(from == 0, "the memory breaking inside the materialize sends the engine to the reread");
            check(!e7.holds(ids) && !e7.holds(base), "the engine holds nothing of a half materialize");
            std::error_code wipe3;   // gone for the later tests' file counts
            std::filesystem::remove(dir / "materialize-me.bin", wipe3);
        }
        std::error_code wipe;
        std::filesystem::remove_all(store, wipe);
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

    // The disk limit includes the states, not only KV rows, both for newly parked and scanned sessions.
    {
        const std::filesystem::path own = dir / "state-budget";
        std::filesystem::create_directories(own);
        CacheOptions o = options(own);
        o.session_max_mib = 1;
        Fake e;
        e.state = 300 << 10;
        Conversations<Fake> kept(e, o);
        const auto ping = text(85, 31);
        for (int round = 0; round < 3; ++round) {
            const auto ids = text(81 + round, 301);
            go_on_batch(e, kept, ids, kept.resume(ids));
            go_on_batch(e, kept, ping, kept.resume(ping));
            check(files_in(own) == 1, "the disk cap counts checkpoint bytes when parking");
        }
        Fake cold;
        cold.state = e.state;
        Conversations<Fake> restarted(cold, o);
        const auto ids = text(90, 301);
        go_on_batch(cold, restarted, ids, restarted.resume(ids));
        go_on_batch(cold, restarted, ping, restarted.resume(ping));
        check(files_in(own) == 1, "the disk cap counts all bytes of scanned files too");
        std::filesystem::remove_all(own);
    }

    // A part that refuses to fill its bytes (the engine is not there) fails the whole write and leaves no file.
    {
        std::error_code wipe;
        std::filesystem::remove_all(dir, wipe);
        std::filesystem::create_directories(dir);
        strata::poly::SessionFile::SessionPart bad;
        bad.bytes = 64;
        bad.fill = [](void*, size_t, size_t) { return false; };
        std::vector<std::pair<int64_t, strata::poly::SessionFile::SessionPart>> marks(1, {8, bad});
        strata::poly::SessionFile::WStat st;
        const bool ok = strata::poly::SessionFile::write_stream(dir, "refused.bin", 8, 8, 8, base, marks, {}, &st);
        check(!ok, "a refusing part fails the write");
        check(!st.published, "a failed write publishes nothing");
        check(!std::filesystem::exists(dir / "refused.bin") &&
                  !std::filesystem::exists(dir / "refused.bin.tmp"),
              "a failed write leaves no file behind");
    }

    // Unfriendly directories: a junk .bin (what else can happen: whatever file a client leaves over, or a
    // truncated one) in the session dir only has to be ignored, and a truncated session file is refused with
    // nothing half-read.  The constructor scans, so this must also be given a pass without crashing at startup.
    {
        std::error_code wipe;
        std::filesystem::remove_all(dir, wipe);
        std::filesystem::create_directories(dir);
        // a junk file of huge lies, its magic and version fine
        const std::filesystem::path junk = dir / "junk.bin";
        {
            FILE* f = std::fopen(junk.string().c_str(), "wb");
            const uint32_t magic = strata::poly::SessionFile::kMagic, version = strata::poly::SessionFile::kVersion;
            std::fwrite(&magic, 4, 1, f);
            std::fwrite(&version, 4, 1, f);
            const uint64_t n_seq = ~0ull / 2, nonsense = 9;
            std::fwrite(&n_seq, 8, 1, f);          // its place in the layout: right after the version
            for (int field = 0; field < 5; ++field) std::fwrite(&nonsense, 8, 1, f);   // prompt, sb, rb, marks, rows
            std::fclose(f);
        }
        const std::filesystem::path junk2 = dir / "junk2.bin";
        std::ofstream(junk2, std::ios::binary) << std::string(3, 'x');   // not a session file at all
        // another kind of lies: honest up to the counts, then a checkpoint count the file cannot hold whole
        const std::filesystem::path junk3 = dir / "junk3.bin";
        {
            FILE* f = std::fopen(junk3.string().c_str(), "wb");
            const uint32_t magic = strata::poly::SessionFile::kMagic, version = strata::poly::SessionFile::kVersion;
            std::fwrite(&magic, 4, 1, f);
            std::fwrite(&version, 4, 1, f);
            const uint64_t zero = 0, many = ~0ull / 2, none = 0;
            std::fwrite(&zero, 8, 1, f);   // the seq
            std::fwrite(&zero, 8, 1, f);   // the prompt
            std::fwrite(&zero, 8, 1, f);   // the state's bytes
            std::fwrite(&zero, 8, 1, f);   // the row's bytes
            std::fwrite(&many, 8, 1, f);   // the checkpoints' count: a resize this file cannot carry
            std::fwrite(&none, 8, 1, f);
            std::fclose(f);
        }
        {
            Fake e;
            Conversations<Fake> kept(e, options(dir));   // the scan of the junk must not crash
            check(kept.parked() == 0, "junk files in the session dir are not conversations");
            const std::vector<int32_t> ids = base;
            const int64_t from = kept.resume(ids);
            check(from == 0, "junk in the dir does not stand in for a conversation");
            // a RESTORE of the lying file asks for no memory: the server's invalid, not its memory kind
            std::string err;
            bool freeze = false;
            int64_t tokens = 0;
            size_t rbytes = 0;
            const auto lies = kept.sideload(junk3, 2560, err, tokens, rbytes, freeze);
            check(lies == Conversations<Fake>::SLoad::unreadable && !freeze,
                  "the file's lying checkpoint count is refused, not asked for");
        }
        // a truncated but real file: cut a session file in half and try to bring it in
        const std::filesystem::path store = dir / "store";
        std::filesystem::create_directories(store);
        Fake w;
        Conversations<Fake> warden(w, options(dir));
        std::vector<int32_t> ids = base;
        const int64_t w_from = warden.resume(ids);
        go_on(w, warden, ids, w_from);
        warden.prompt_was((int64_t) ids.size());
        size_t bytes = 0;
        bool published = false;
        const std::string name = "deadbeefdeadbeef-300.bin";
        const auto out = warden.snapshot(store, name, {}, bytes, published);
        check(out == Conversations<Fake>::SOut::saved, "the reference file for the truncation is written");
        {
            std::ifstream in(store / name, std::ios::binary);
            std::string all{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            all.resize(all.size() / 2);           // the seq readable, the payloads not there
            std::ofstream half(store / "short.bin", std::ios::binary);
            half.write(all.data(), (std::streamsize) all.size());
        }
        Fake b;
        Conversations<Fake> brought(b, options(dir));
        std::string err;
        int64_t tokens = 0;
        size_t rbytes = 0;
        bool touched = false;
        const auto half = brought.sideload(store / "short.bin", 2560, err, tokens, rbytes, touched);
        check(half == Conversations<Fake>::SLoad::unreadable, "a cut-off file is refused, not half-read");
        check(!touched, "a refusal of a file touches no engine state");

        std::error_code wipe_final;
        std::filesystem::remove_all(store, wipe_final);
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf("conversations_disk_test: %d checks passed\n", checks);
    return 0;
}
