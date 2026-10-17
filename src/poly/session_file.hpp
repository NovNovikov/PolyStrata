// src/poly/session_file.hpp - a parked conversation's bytes in a file, not in RAM.
//
// Stage 1 of the disk-session port for the poly engine (the Qwen engine's is the strata-core format).  A
// Conversations<> slot parks by writing one file here and freeing its RAM blobs; a restore reads a file back.
// The layout is engine-agnostic: the payloads are the engine's own save_state()/save_rows() bytes, opaque here.
// A file holds the WHOLE checkpoint chain (the Qwen round's lesson: the deepest checkpoint alone cannot mount a
// prompt that forks before it).
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

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

    struct WStat {
        bool published = false;   // the new file's bytes are the name's now (the rename happened)
        bool dir_flush_unsupported = false;   // POSIX fsync EINVAL: the save succeeded without that flush
        size_t bytes = 0;   // the file's size as the read side will see it
        bool no_room = false;   // the write saw a full disk (ENOSPC / EDQUOT / EFBIG): the server's storage kind
        const char* fail = nullptr;   // who refused, when the write failed
    };

    // One payload read straight from where it lives (a parked blob, or the engine's own rows) into the file's
    // staging buffer, without a full host copy first.  `fill(dst, offset, n)` copies bytes [offset, offset+n)
    // of the part; every span of a part with `align > 1` is a whole number of the alignment (an engine's rows
    // are read whole).  Everything referenced must outlive the write: the engine is idle between requests, the
    // same contract as the Qwen engine's SessionKvSource.  A false fill fails the write (no file left behind).
    struct SessionPart {
        uint64_t bytes = 0;
        uint64_t align = 1;
        std::function<bool(void* dst, size_t offset, size_t n)> fill;
    };

    static SessionPart part_of(const uint8_t* p, size_t bytes) {
        SessionPart part;
        part.bytes = bytes;
        part.fill = [p](void* dst, size_t off, size_t n) {
            std::memcpy(dst, p + off, n);
            return true;
        };
        return part;
    }

    struct StreamRows {
        int64_t first = 0, count = 0, keep = 0;
        SessionPart part;
    };

    // Streams the file with the payloads arriving as parts (no full host copy): `seq` is referenced, not
    // copied (it must outlive the write); each mark's part is one record's state and each rows record's part
    // is its block, in the file's order.  Same refusal contract as write():
    static bool write_stream(const std::filesystem::path& dir, const std::string& name, int64_t prompt,
                             uint64_t sb, uint64_t rb, const std::vector<int32_t>& seq,
                             const std::vector<std::pair<int64_t, SessionPart>>& marks,
                             const std::vector<StreamRows>& rows, WStat* st = nullptr,
                             const std::function<void(uint64_t, uint64_t)>& say = {}) {
        if (st) *st = WStat{};
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
        const auto flush_dir = [&]() -> int {   // 1 (no flush available), 0 (flushed), -1 (flush refused)
#ifdef _WIN32
            return 1;
#else
            const int d = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
            if (d < 0) return errno == EINVAL || errno == ENOTSUP || errno == EACCES || errno == EROFS ? 1 : -1;
            const int rc = ::fsync(d);
            ::close(d);
            return rc == 0 ? 0 : (errno == EINVAL || errno == ENOTSUP || errno == EROFS ? 1 : -1);
#endif
        };
        uint64_t total = (uint64_t) seq.size() * 4;
        uint64_t payload = total;
        size_t align = 1;
        for (const auto& [pos, part] : marks) {
            total += 16 + part.bytes;
            payload += part.bytes;
            if (part.align > 1) align = (size_t) part.align;
        }
        for (const StreamRows& r : rows) {
            total += 32 + r.part.bytes;
            payload += r.part.bytes;
            if (r.part.align > 1) align = (size_t) r.part.align;
        }
        const size_t kChunk = (size_t) 4 << 20;
        std::vector<uint8_t> buf(payload ? (size_t) std::min<uint64_t>(payload, std::max<size_t>(kChunk, align)) : 0);
        uint64_t done = 0, next = 0;
        const auto mark = [&]() {
            if (say && done >= next) {
                say(done, total);
                next = done + ((uint64_t) 1 << 20);   // each MiB, as the server reads between answers
            }
        };
        // a full disk (ENOSPC, or quota/over-limit where the platform counts them) is the server's storage kind
        const auto lacked = [](int e) {
            return false
#ifdef ENOSPC
                || e == ENOSPC
#endif
#ifdef EFBIG
                || e == EFBIG
#endif
#ifdef EDQUOT
                || e == EDQUOT
#endif
                ;
        };
        const char* fail = nullptr;
        bool no_room = false;
        const auto stream = [&](const SessionPart& p) {
            uint64_t off = 0;
            while (off < p.bytes) {
                uint64_t n = p.bytes - off;
                if (n > buf.size()) n = buf.size();
                if (p.align > 1) n -= n % p.align;
                if (n == 0) {   // the part's bytes do not hold on its own alignment
                    fail = "the part's size and alignment disagree";
                    return false;
                }
                if (!p.fill(buf.data(), (size_t) off, (size_t) n)) {
                    fail = "the engine would not hand over its payload";
                    return false;
                }
                const size_t put = std::fwrite(buf.data(), 1, (size_t) n, f);
                done += put;
                mark();
                if (put != n) {
                    fail = "the write fell short";
                    no_room = lacked(errno);
                    return false;
                }
                off += n;
            }
            return true;
        };
        bool ok = put(f, kMagic) && put(f, kVersion) && put(f, (uint64_t) seq.size()) && put(f, (uint64_t) prompt) &&
                  put(f, sb) && put(f, rb) && put(f, (uint64_t) marks.size()) && put(f, (uint64_t) rows.size());
        if (ok && !seq.empty())
            ok = stream(part_of((const uint8_t*) seq.data(), (size_t) seq.size() * 4));
        for (const auto& [pos, part] : marks) {
            if (!ok) break;
            ok = put(f, (uint64_t) pos) && put(f, (uint64_t) part.bytes);
            if (ok) ok = stream(part);
        }
        for (const StreamRows& r : rows) {
            if (!ok) break;
            ok = put(f, (uint64_t) r.first) && put(f, (uint64_t) r.count) && put(f, (uint64_t) r.keep) &&
                 put(f, (uint64_t) r.part.bytes);
            if (ok) ok = stream(r.part);
        }
        if (ok && say && done < total) say(done, total);
        if (ok) {
            ok = std::fflush(f) == 0;
            if (!ok) {
                fail = "the flush refused";
                no_room = lacked(errno);
            }
        }
#ifdef POLY_DEBUG_SESSION
        std::fprintf(stderr, "write_stream: seq=%zu marks=%zu rows=%zu ok=%d\n", seq.size(), marks.size(),
                     rows.size(), (int) ok);
#endif
        std::fclose(f);
        if (!ok) {
            if (st) {
                st->fail = fail;
                st->no_room = no_room;
            }
            std::error_code nix;
            std::filesystem::remove(tmp, nix);
            return false;
        }
        const std::filesystem::path dst = dir / name;
        std::filesystem::remove(dst, ec);   // Windows rename does not replace, so take the old one first
        std::filesystem::rename(tmp, dst, ec);
        if (ec) {
            if (st) {
                st->fail = "the rename refused";
                st->no_room = lacked(ec.value());
            }
            std::error_code nix;
            std::filesystem::remove(tmp, nix);   // a failed rename does not leave the tmp around
            return false;
        }
        // the name holds the new bytes; a folder flush on POSIX is what carries them through a power loss, and
        // its refusal is a failed save of a file the name already published — the server knows `published`
        const int flushed = flush_dir();
        if (st) {
            st->published = true;
            st->dir_flush_unsupported = flushed == 1;
            std::error_code nix;
            const auto sz = std::filesystem::file_size(dst, nix);
            st->bytes = !nix ? (size_t) sz : 0;
        }
        return flushed >= 0;
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

    // The header's counts, once they are held against the file's size: junk counts are refused before any
    // memory is asked for them.
    struct Head0 {
        int64_t prompt = 0;
        uint64_t sb = 0, rb = 0;
        uint64_t n_seq = 0, n_marks = 0, n_rows = 0;
    };

    static bool head_ok(const std::filesystem::path& path, FILE* f, Head0& v, uint64_t& room) {
        uint32_t magic = 0, version = 0;
        if (!get(f, magic) || !get(f, version) || magic != kMagic || version != kVersion ||
            !get(f, v.n_seq) || !get(f, (uint64_t&) v.prompt) || !get(f, v.sb) || !get(f, v.rb) ||
            !get(f, v.n_marks) || !get(f, v.n_rows))
            return false;
        std::error_code ec;
        const uint64_t file_bytes = (uint64_t) std::filesystem::file_size(path, ec);
        if (ec) return false;
        room = file_bytes >= 56 ? file_bytes - 56 : 0;   // the header's own 56 bytes come off first
        if (v.n_seq > room / 4) return false;            // the seq's bytes alone will not fit this file
        room -= v.n_seq * 4;
        // each record carries its own fields (a state 16 bytes, a rows block 32): the counts of records the
        // file cannot hold whole are junk
        if (v.n_marks > room / 16 || v.n_rows > room / 32) return false;
        return true;
    }

    static std::optional<Head> read_head(const std::filesystem::path& path) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return std::nullopt;
        Head0 open;
        uint64_t room = 0;
        if (!head_ok(path, f, open, room)) {
            std::fclose(f);
            return std::nullopt;
        }
        Head h;
        h.seq.resize((size_t) open.n_seq);
        if (open.n_seq && std::fread(h.seq.data(), 4, (size_t) open.n_seq, f) != (size_t) open.n_seq) {
            std::fclose(f);
            return std::nullopt;
        }
        h.prompt = open.prompt;
        h.sb = open.sb;
        h.rb = open.rb;
        size_t payloads = 0;
        bool ok = true;
        for (uint64_t i = 0; ok && i < open.n_marks; ++i) {
            uint64_t pos = 0, bytes = 0;
            ok = get(f, pos) && get(f, bytes) && bytes + 16 <= room;   // the entry's own fields are in the file too
            if (!ok) break;
            room -= bytes + 16;
            payloads += (size_t) bytes;
            h.mark_pos.push_back((int64_t) pos);
            ok = std::fseek(f, (long) bytes, SEEK_CUR) == 0;
        }
        for (uint64_t i = 0; ok && i < open.n_rows; ++i) {
            uint64_t first = 0, count = 0, keep = 0, bytes = 0;
            ok = get(f, first) && get(f, count) && get(f, keep) && get(f, bytes) && bytes + 32 <= room;
            if (!ok) break;
            room -= bytes + 32;
            payloads += (size_t) bytes;
            h.rows_last = (int64_t) (first + keep);
            ok = std::fseek(f, (long) bytes, SEEK_CUR) == 0;
        }
        std::fclose(f);
        if (!ok) return std::nullopt;
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
        Head0 open;
        uint64_t room = 0;
        if (!head_ok(path, f, open, room)) {
            std::fclose(f);
            return std::nullopt;
        }
        SessionView v;
        v.prompt = open.prompt;
        v.sb = open.sb;
        v.rb = open.rb;
        v.seq.resize((size_t) open.n_seq);
        bool ok = !open.n_seq || std::fread(v.seq.data(), 4, (size_t) open.n_seq, f) == (size_t) open.n_seq;
        v.marks.resize((size_t) open.n_marks);
        for (auto& m : v.marks) {
            uint64_t pos = 0, bytes = 0;
            ok = ok && get(f, pos) && get(f, bytes) && bytes + 16 <= room;
            if (!ok) break;
            room -= bytes + 16;
            m.pos = (int64_t) pos;
            m.state.resize((size_t) bytes);
            ok = !bytes || std::fread(m.state.data(), 1, (size_t) bytes, f) == (size_t) bytes;
        }
        v.rows.resize((size_t) open.n_rows);
        for (auto& r : v.rows) {
            uint64_t first = 0, count = 0, keep = 0, bytes = 0;
            ok = ok && get(f, first) && get(f, count) && get(f, keep) && get(f, bytes) && bytes + 32 <= room;
            if (!ok) break;
            room -= bytes + 32;
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
