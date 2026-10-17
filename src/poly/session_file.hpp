// src/poly/session_file.hpp - a parked conversation's bytes in a file, not in RAM.
//
// Stage 1 of the disk-session port for the poly engine (the Qwen engine's is the strata-core format).  A
// Conversations<> slot parks by writing one file here and freeing its RAM blobs; a restore reads a file back.
// The layout is engine-agnostic: the payloads are the engine's own save_state()/save_rows() bytes, opaque here.
// A file holds the WHOLE checkpoint chain (the Qwen round's lesson: the deepest checkpoint alone cannot mount a
// prompt that forks before it).
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace strata::poly {

struct SessionView {
    std::vector<int32_t> seq;
    int64_t prompt = 0;
    uint64_t sb = 0, rb = 0;   // the engine's state_bytes() and row_bytes() that wrote the payloads
    struct Mark {
        int64_t pos;
        std::vector<uint8_t> state;
    };
    std::vector<Mark> marks;   // by position, as Conversations keeps them
    struct Rows {
        int64_t first, count, keep;
        std::vector<uint8_t> data;
    };
    std::vector<Rows> rows;    // back to back from row 0
};

class SessionFile {
public:
    static constexpr uint32_t kMagic = 0x53505353u;   // "SSPS" little-endian
    static constexpr uint32_t kVersion = 1;

    // A parked conversation's name: stable for the same token stream, so a repark overwrites its own file.
    static std::string name_for(const std::vector<int32_t>& seq) {
        uint64_t h = 1469598103934665603ull;   // FNV-1a
        for (int32_t t : seq) {
            const uint32_t u = (uint32_t) t;
            h ^= u;
            h *= 1099511628211ull;
            h ^= u << 16;   // fold the sign bits in too
        }
        char out[48];
        std::snprintf(out, sizeof out, "%016llx-%lld.bin", (unsigned long long) h, (long long) seq.size());
        return out;
    }

    // Writes the whole slot; false on any I/O failure (the caller then forgets the conversation, as a RAM park
    // that does not fit is forgotten).
    static bool write(const std::filesystem::path& dir, const std::string& name, const SessionView& v) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
        const std::filesystem::path tmp = dir / (name + ".tmp");
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, tmp.c_str(), L"wb");
#else
        f = std::fopen(tmp.c_str(), "wb");
#endif
        if (!f) return false;
        bool ok = put(f, kMagic) && put(f, kVersion) && put(f, (uint64_t) v.seq.size()) &&
                  put(f, (uint64_t) v.prompt) && put(f, v.sb) && put(f, v.rb) &&
                  put(f, (uint64_t) v.marks.size()) && put(f, (uint64_t) v.rows.size());
        if (!v.seq.empty()) ok = std::fwrite(v.seq.data(), 4, (size_t) v.seq.size(), f) == (size_t) v.seq.size();
        for (const SessionView::Mark& m : v.marks) {
            ok = ok && put(f, (uint64_t) m.pos) && put(f, (uint64_t) m.state.size());
            if (ok && !m.state.empty()) ok = std::fwrite(m.state.data(), 1, m.state.size(), f) == m.state.size();
        }
        for (const SessionView::Rows& r : v.rows) {
            ok = ok && put(f, (uint64_t) r.first) && put(f, (uint64_t) r.count) && put(f, (uint64_t) r.keep) &&
                 put(f, (uint64_t) r.data.size());
            if (ok && !r.data.empty()) ok = std::fwrite(r.data.data(), 1, r.data.size(), f) == r.data.size();
        }
        ok = ok && std::fflush(f) == 0;
#ifdef POLY_DEBUG_SESSION
        std::fprintf(stderr, "write: seq=%zu marks=%zu rows=%zu ok=%d ret=%d\n", v.seq.size(), v.marks.size(),
                     v.rows.size(), (int) ok, (int) (size_t) (v.seq.empty() ? 1 : 0));
#endif
        std::fclose(f);
        if (!ok) {
            std::filesystem::remove(tmp, ec);
            return false;
        }
        const std::filesystem::path dst = dir / name;
        std::filesystem::remove(dst, ec);   // Windows rename does not replace, so take the old one first
        std::filesystem::rename(tmp, dst, ec);
        return !ec;
    }

    // The header without the payloads: enough for resume()'s scan (the tokens and where the checkpoints sit).
    struct Head {
        std::vector<int32_t> seq;
        int64_t prompt = 0;
        uint64_t sb = 0, rb = 0;
        std::vector<int64_t> mark_pos;
        int64_t rows_last = 0;   // the last row the conversation keeps, 0 when none
        size_t bytes = 0;        // the payloads plus the header, as the file is sized
    };

    static std::optional<Head> read_head(const std::filesystem::path& path) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return std::nullopt;
        Head h;
        uint32_t magic = 0, version = 0;
        uint64_t n_seq = 0, n_marks = 0, n_rows = 0;
        const bool ok = get(f, magic) && get(f, version) && magic == kMagic && version == kVersion &&
                        get(f, n_seq) && get(f, (uint64_t&) h.prompt) && get(f, h.sb) && get(f, h.rb) &&
                        get(f, n_marks) && get(f, n_rows);
        if (!ok) {
            std::fclose(f);
            return std::nullopt;
        }
        h.seq.resize((size_t) n_seq);
        if (n_seq && std::fread(h.seq.data(), 4, (size_t) n_seq, f) != (size_t) n_seq) {
            std::fclose(f);
            return std::nullopt;
        }
        size_t payloads = 0;
        for (uint64_t i = 0; i < n_marks; ++i) {
            uint64_t pos = 0, bytes = 0;
            if (!get(f, pos) || !get(f, bytes) || std::fseek(f, (long) bytes, SEEK_CUR)) {
                std::fclose(f);
                return std::nullopt;
            }
            h.mark_pos.push_back((int64_t) pos);
            payloads += (size_t) bytes;
        }
        for (uint64_t i = 0; i < n_rows; ++i) {
            uint64_t first = 0, count = 0, keep = 0, bytes = 0;
            if (!get(f, first) || !get(f, count) || !get(f, keep) || !get(f, bytes) ||
                std::fseek(f, (long) bytes, SEEK_CUR)) {
                std::fclose(f);
                return std::nullopt;
            }
            h.rows_last = (int64_t) (first + keep);
            payloads += (size_t) bytes;
        }
        std::fclose(f);
        h.bytes = payloads + 64;   // the header's own fields, the size a write's reader sees
        return h;
    }

    static std::optional<SessionView> read(const std::filesystem::path& path) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return std::nullopt;
        SessionView v;
        uint32_t magic = 0, version = 0;
        uint64_t n_seq = 0, n_marks = 0, n_rows = 0;
        const bool head = get(f, magic) && get(f, version) && magic == kMagic && version == kVersion &&
                          get(f, n_seq) && get(f, (uint64_t&) v.prompt) && get(f, v.sb) && get(f, v.rb) &&
                          get(f, n_marks) && get(f, n_rows);
        if (!head) {
            std::fclose(f);
            return std::nullopt;
        }
        v.seq.resize((size_t) n_seq);
        bool ok = !n_seq || std::fread(v.seq.data(), 4, (size_t) n_seq, f) == (size_t) n_seq;
        v.marks.resize((size_t) n_marks);
        for (auto& m : v.marks) {
            uint64_t pos = 0, bytes = 0;
            ok = ok && get(f, pos) && get(f, bytes);
            if (!ok) break;
            m.pos = (int64_t) pos;
            m.state.resize((size_t) bytes);
            ok = !bytes || std::fread(m.state.data(), 1, (size_t) bytes, f) == (size_t) bytes;
        }
        v.rows.resize((size_t) n_rows);
        for (auto& r : v.rows) {
            uint64_t first = 0, count = 0, keep = 0, bytes = 0;
            ok = ok && get(f, first) && get(f, count) && get(f, keep) && get(f, bytes);
            if (!ok) break;
            r.first = (int64_t) first;
            r.count = (int64_t) count;
            r.keep = (int64_t) keep;
            r.data.resize((size_t) bytes);
            ok = !bytes || std::fread(r.data.data(), 1, (size_t) bytes, f) == (size_t) bytes;
        }
        std::fclose(f);
        return ok ? std::optional<SessionView>(std::move(v)) : std::nullopt;
    }

private:
    template<class T>
    static bool put(FILE* f, T v) {
        return std::fwrite(&v, sizeof v, 1, f) == 1;
    }
    template<class T>
    static bool get(FILE* f, T& v) {
        return std::fread(&v, sizeof v, 1, f) == 1;
    }
};

}  // namespace strata::poly
