// src/poly/session_file.hpp - a parked conversation's bytes in a file, not in RAM.
//
// Stage 1 of the disk-session port for the poly engine (the Qwen engine's is the strata-core format).  A
// Conversations<> slot parks by writing one file here and freeing its RAM blobs; a restore reads a file back.
// The layout is engine-agnostic: the payloads are the engine's own save_state()/save_rows() bytes, opaque here.
// A file holds the WHOLE checkpoint chain (the Qwen round's lesson: the deepest checkpoint alone cannot mount a
// prompt that forks before it).
//
// Version 3 packs the payloads: the engine's rows hold a block's compressed row once per block, so most of a
// payload is empty positions (85 % of the bytes of a real DeepSeek4 file).  A run of zeros is written as a
// count, so a version 3 file is about a third of the size.  A version 2 file is still read.
//
// A version 3 record's header is its version 2 header (the same fields, the same sizes) plus the payload's
// packed size, so the reader knows the record's exact span in the file and can refuse a truncated one.
#pragma once

#include <cerrno>
#include <algorithm>
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
    // Version 1 could concatenate independently captured layer-major chunks into a single rows record.
    // Its metadata cannot distinguish those corrupt parks from valid snapshots; reread rather than reuse them.
    static constexpr uint32_t kVersion = 2;           // the payloads as the engine wrote them
    static constexpr uint32_t kVersionPacked = 3;     // the payloads packed: a run of zeros is a count

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

    // The header without the payloads: enough for resume()'s scan (the tokens and where the checkpoints sit).
    struct Head {
        std::vector<int32_t> seq;
        int64_t prompt = 0;
        uint64_t sb = 0, rb = 0;
        std::vector<int64_t> mark_pos;
        int64_t rows_last = 0;   // the last row the conversation keeps, 0 when none
        uint64_t payload = 0;    // the payloads' size unpacked: what a restore holds in memory
        uint64_t marks_bytes = 0;   // the checkpoints' states alone: a restore keeps these
        uint64_t max_record = 0;    // the largest record: a restore that streams needs room for this one
        uint64_t rows_at = 0;       // where the rows' records start in the file
        uint64_t n_rows = 0;        // how many rows' records there are
        uint32_t version = 0;       // the file's version
        size_t bytes = 0;        // the payloads plus the header, as the file is sized
    };

    // The header's counts, once they are held against the file's size: junk counts are refused before any
    // memory is asked for.
    struct Head0 {
        int64_t prompt = 0;
        uint64_t sb = 0, rb = 0;
        uint64_t n_seq = 0, n_marks = 0, n_rows = 0;
        uint32_t version = 0;
    };

    // One record of a file: a checkpoint's state, or a block of rows.
    struct Record {
        enum Kind { kMark, kRows } kind = kMark;
        size_t index = 0;                                // the mark's or rows record's place in the file
        int64_t pos = 0;                                 // a mark's position
        int64_t first = 0, count = 0, keep = 0;          // a rows span's places
        uint64_t raw = 0;                                // the payload's size, unpacked
    };

    // A record's header as it is in the file: what it says, and what it takes of the file.
    struct Raw {
        uint64_t pos = 0, first = 0, count = 0, keep = 0;
        uint64_t bytes = 0;    // the payload's size, unpacked
        uint64_t packed = 0;   // the payload's size in the file
        uint64_t header = 0;   // the header's own size
        uint64_t span = 0;     // the header and the payload together: what the record takes of the file
    };

    // A record's header, read: what the record says and what it takes of the file.  A version 3 record's
    // payload is packed, so it takes less of the file than it says.
    static bool read_raw(FILE* f, uint32_t version, bool rows, Raw& r) {
        if (rows) {
            if (!get(f, r.first) || !get(f, r.count) || !get(f, r.keep) || !get(f, r.bytes)) return false;
            r.header = 32;
        } else {
            if (!get(f, r.pos) || !get(f, r.bytes)) return false;
            r.header = 16;
        }
        if (version == kVersionPacked) {
            if (!get(f, r.packed)) return false;
            r.header += 8;
        } else {
            r.packed = r.bytes;
        }
        r.span = r.header + r.packed;
        return true;
    }

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
            return rc == 0 ? 0 : (errno == EINVAL || errno == ENOTSUP || errno == EACCES || errno == EROFS ? 1 : -1);
#endif
        };
        uint64_t total = (uint64_t) seq.size() * 4;
        uint64_t payload = total;
        size_t align = 1;
        for (const auto& [pos, part] : marks) {
            (void) pos;
            total += 16 + part.bytes;
            payload += part.bytes;
            if (part.align > 1) align = (size_t) part.align;
        }
        for (const auto& r : rows) {
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
        // The payload goes out packed: a run of zeros as a count, the rest as literal bytes.  The packer writes
        // what it has as it goes, so a payload never sits in memory whole.
        Packer packer(f);
        const auto stream = [&](const SessionPart& p) {
            uint64_t off = 0;
            while (off < p.bytes) {
                uint64_t n = p.bytes - off;
                if (n > buf.size()) n = buf.size();
                if (p.align > 1) n -= n % p.align;
                if (n == 0) {   // the part's bytes do not hold on their own alignment
                    fail = "the part's size and alignment disagree";
                    return false;
                }
                if (!p.fill(buf.data(), (size_t) off, (size_t) n)) {
                    fail = "the engine would not hand over its payload";
                    return false;
                }
                packer.put(buf.data(), (size_t) n);
                done += n;
                mark();
                off += n;
            }
            packer.finish();
            if (packer.failed()) {
                fail = "the write fell short";
                no_room = lacked(errno);
                return false;
            }
            return true;
        };
        bool ok = put(f, kMagic) && put(f, kVersionPacked) && put(f, (uint64_t) seq.size()) &&
                  put(f, (uint64_t) prompt) && put(f, sb) && put(f, rb) && put(f, (uint64_t) marks.size()) &&
                  put(f, (uint64_t) rows.size());
        if (ok && !seq.empty())
            ok = std::fwrite(seq.data(), 4, seq.size(), f) == seq.size();
        // A record's header names its payload's packed size, which is only known once the payload is written:
        // the size is left out and filled in after, and the file is a temporary one until it is renamed.
        for (const auto& [pos, part] : marks) {
            if (!ok) break;
            ok = put(f, (uint64_t) pos) && put(f, (uint64_t) part.bytes);
            if (!ok) break;
            const long at = std::ftell(f);
            ok = put(f, (uint64_t) 0);
            if (!ok) break;
            packer.reset();
            if (!stream(part)) { ok = false; break; }
            ok = patch(f, at, packer.written());
        }
        for (const auto& r : rows) {
            if (!ok) break;
            ok = put(f, (uint64_t) r.first) && put(f, (uint64_t) r.count) && put(f, (uint64_t) r.keep) &&
                 put(f, (uint64_t) r.part.bytes);
            if (!ok) break;
            const long at = std::ftell(f);
            ok = put(f, (uint64_t) 0);
            if (!ok) break;
            packer.reset();
            if (!stream(r.part)) { ok = false; break; }
            ok = patch(f, at, packer.written());
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

    static bool head_ok(const std::filesystem::path& path, FILE* f, Head0& v, uint64_t& room) {
        uint32_t magic = 0, version = 0;
        if (!get(f, magic) || !get(f, version) || magic != kMagic ||
            (version != kVersion && version != kVersionPacked) ||
            !get(f, v.n_seq) || !get(f, (uint64_t&) v.prompt) || !get(f, v.sb) || !get(f, v.rb) ||
            !get(f, v.n_marks) || !get(f, v.n_rows))
            return false;
        v.version = version;
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
        h.version = open.version;
        h.n_rows = open.n_rows;
        bool ok = true;
        for (uint64_t i = 0; ok && i < open.n_marks; ++i) {
            Raw r;
            ok = read_raw(f, open.version, false, r) && r.span <= room;
            if (!ok) break;
            room -= r.span;
            h.payload += r.bytes;
            h.marks_bytes += r.bytes;
            h.max_record = std::max(h.max_record, r.bytes);
            h.mark_pos.push_back((int64_t) r.pos);
            ok = skip_payload(f, open.version, r.bytes, r.packed);
        }
        for (uint64_t i = 0; ok && i < open.n_rows; ++i) {
            Raw r;
            ok = read_raw(f, open.version, true, r) && r.span <= room;
            if (!ok) break;
            if (!i) h.rows_at = (uint64_t) std::ftell(f) - r.header;   // where the rows' records start
            room -= r.span;
            h.payload += r.bytes;
            h.max_record = std::max(h.max_record, r.bytes);
            h.rows_last = (int64_t) (r.first + r.keep);
            ok = skip_payload(f, open.version, r.bytes, r.packed);
        }
        std::fclose(f);
        if (!ok) return std::nullopt;
        std::error_code ec;
        h.bytes = (size_t) std::filesystem::file_size(path, ec);
        if (ec) return std::nullopt;
        return h;
    }

    // The whole file, unpacked into memory: what a caller that wants the payloads reads.  A caller that wants
    // them in its own buffers reads them there instead (read_into), so the file's bytes are never held twice.
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
            Raw r;
            ok = ok && read_raw(f, open.version, false, r);
            if (!ok) break;
            room -= r.span;
            m.pos = (int64_t) r.pos;
            m.state.resize((size_t) r.bytes);
            ok = ok && read_payload(f, open.version, m.state.data(), r.bytes, r.packed);
        }
        v.rows.resize((size_t) open.n_rows);
        for (auto& r : v.rows) {
            Raw raw;
            ok = ok && read_raw(f, open.version, true, raw);
            if (!ok) break;
            room -= raw.span;
            r.first = (int64_t) raw.first;
            r.count = (int64_t) raw.count;
            r.keep = (int64_t) raw.keep;
            r.data.resize((size_t) raw.bytes);
            ok = ok && read_payload(f, open.version, r.data.data(), raw.bytes, raw.packed);
        }
        std::fclose(f);
        return ok ? std::optional<SessionView>(std::move(v)) : std::nullopt;
    }

    // The file's records, one at a time: the payload of each is unpacked into the buffer the caller names.
    // `visit` is called with the record and returns where the payload goes; null means the memory for it
    // is not there and the read fails.  The file's token list and sizes are in `head`.
    template<class Visit>
    static bool read_into(const std::filesystem::path& path, Head& head, Visit&& visit) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return false;
        Head0 open;
        uint64_t room = 0;
        if (!head_ok(path, f, open, room)) {
            std::fclose(f);
            return false;
        }
        head.seq.resize((size_t) open.n_seq);
        bool ok = !open.n_seq || std::fread(head.seq.data(), 4, (size_t) open.n_seq, f) == (size_t) open.n_seq;
        head.prompt = open.prompt;
        head.sb = open.sb;
        head.rb = open.rb;
        for (uint64_t i = 0; ok && i < open.n_marks; ++i) {
            Raw raw;
            ok = read_raw(f, open.version, false, raw);
            if (!ok) break;
            room -= raw.span;
            Record r;
            r.kind = Record::kMark;
            r.index = (size_t) i;
            r.pos = (int64_t) raw.pos;
            r.raw = raw.bytes;
            head.mark_pos.push_back(r.pos);
            uint8_t* dst = visit(r);
            if (!dst) { ok = false; break; }
            ok = read_payload(f, open.version, dst, raw.bytes, raw.packed);
        }
        for (uint64_t i = 0; ok && i < open.n_rows; ++i) {
            Raw raw;
            ok = read_raw(f, open.version, true, raw);
            if (!ok) break;
            room -= raw.span;
            Record r;
            r.kind = Record::kRows;
            r.index = (size_t) i;
            r.first = (int64_t) raw.first;
            r.count = (int64_t) raw.count;
            r.keep = (int64_t) raw.keep;
            r.raw = raw.bytes;
            head.rows_last = (int64_t) (raw.first + raw.keep);
            uint8_t* dst = visit(r);
            if (!dst) { ok = false; break; }
            ok = read_payload(f, open.version, dst, raw.bytes, raw.packed);
        }
        std::fclose(f);
        return ok;
    }

    // A payload's bytes, out of the file's packed bytes: a version 2 payload is the bytes as they are, a
    // version 3 payload is packed.  False: the bytes are not the payload they say.
    static bool unpack(uint32_t version, const uint8_t* in, size_t in_bytes, uint8_t* dst, uint64_t bytes) {
        if (version != kVersionPacked) {
            if (in_bytes != bytes) return false;
            if (bytes) std::memcpy(dst, in, (size_t) bytes);
            return true;
        }
        size_t at = 0;
        uint64_t left = bytes;
        while (left) {
            if (at + 8 > in_bytes) return false;
            uint64_t h = 0;
            std::memcpy(&h, in + at, 8);
            at += 8;
            if (h & (1ull << 63)) {   // a run of zeros
                const uint64_t run = h & ~(1ull << 63);
                if (run > left) return false;
                if (dst) std::memset(dst + (bytes - left), 0, (size_t) run);
                left -= run;
                continue;
            }
            if (h > left || at + h > in_bytes) return false;
            if (dst) std::memcpy(dst + (bytes - left), in + at, (size_t) h);
            at += (size_t) h;
            left -= h;
        }
        return true;
    }

    // The file's records, one at a time, with their payloads packed: the caller unpacks a record when it wants
    // it, so the payloads never have to be in the host memory all at once.  `visit` gets the record and the
    // payload's packed bytes in a buffer it takes over; it returns false to stop the read.  A version 2 file's
    // payloads come as they are.
    template<class Visit>
    static bool read_packed(const std::filesystem::path& path, Head& head, Visit&& visit,
                            bool marks_only = false) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return false;
        Head0 open;
        uint64_t room = 0;
        if (!head_ok(path, f, open, room)) {
            std::fclose(f);
            return false;
        }
        head.seq.resize((size_t) open.n_seq);
        bool ok = !open.n_seq || std::fread(head.seq.data(), 4, (size_t) open.n_seq, f) == (size_t) open.n_seq;
        head.prompt = open.prompt;
        head.sb = open.sb;
        head.rb = open.rb;
        head.version = open.version;
        head.n_rows = open.n_rows;
        std::vector<uint8_t> packed;
        for (uint64_t i = 0; ok && i < open.n_marks; ++i) {
            Raw raw;
            ok = read_raw(f, open.version, false, raw);
            if (!ok) break;
            room -= raw.span;
            Record r;
            r.kind = Record::kMark;
            r.index = (size_t) i;
            r.pos = (int64_t) raw.pos;
            r.raw = raw.bytes;
            head.mark_pos.push_back(r.pos);
            head.payload += raw.bytes;
            head.marks_bytes += raw.bytes;
            head.max_record = std::max(head.max_record, raw.bytes);
            packed.assign((size_t) raw.packed, 0);
            if (raw.packed && std::fread(packed.data(), 1, (size_t) raw.packed, f) != (size_t) raw.packed) {
                ok = false;
                break;
            }
            if (!visit(r, std::move(packed), open.version)) {
                ok = false;
                break;
            }
        }
        for (uint64_t i = 0; ok && i < open.n_rows; ++i) {
            Raw raw;
            ok = read_raw(f, open.version, true, raw);
            if (!ok) break;
            room -= raw.span;
            Record r;
            r.kind = Record::kRows;
            r.index = (size_t) i;
            r.first = (int64_t) raw.first;
            r.count = (int64_t) raw.count;
            r.keep = (int64_t) raw.keep;
            r.raw = raw.bytes;
            head.payload += raw.bytes;
            head.max_record = std::max(head.max_record, raw.bytes);
            head.rows_last = (int64_t) (raw.first + raw.keep);
            if (!i) head.rows_at = (uint64_t) std::ftell(f) - raw.header;   // where the rows' records start
            if (marks_only) {   // the rows are not wanted now: the file is stepped over them
                ok = skip_payload(f, open.version, raw.bytes, raw.packed);
                if (!ok) break;
                continue;
            }
            packed.assign((size_t) raw.packed, 0);
            if (raw.packed && std::fread(packed.data(), 1, (size_t) raw.packed, f) != (size_t) raw.packed) {
                ok = false;
                break;
            }
            if (!visit(r, std::move(packed), open.version)) {
                ok = false;
                break;
            }
        }
        std::fclose(f);
        return ok;
    }

    // The rows' records from a file's offset, one at a time, with their payloads packed: a reader that has
    // the checkpoints already does not walk the file again for them.
    template<class Visit>
    static bool read_packed_rows(const std::filesystem::path& path, uint64_t at, uint64_t n_rows, uint32_t version,
                                 Visit&& visit) {
        FILE* f = nullptr;
#if defined(_WIN32)
        _wfopen_s(&f, path.c_str(), L"rb");
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        if (!f) return false;
        if (std::fseek(f, (long) at, SEEK_SET) != 0) {
            std::fclose(f);
            return false;
        }
        bool ok = true;
        std::vector<uint8_t> packed;
        for (uint64_t i = 0; i < n_rows; ++i) {
            Raw raw;
            ok = read_raw(f, version, true, raw);
            if (!ok) break;
            Record r;
            r.kind = Record::kRows;
            r.index = (size_t) i;
            r.first = (int64_t) raw.first;
            r.count = (int64_t) raw.count;
            r.keep = (int64_t) raw.keep;
            r.raw = raw.bytes;
            packed.assign((size_t) raw.packed, 0);
            if (raw.packed && std::fread(packed.data(), 1, (size_t) raw.packed, f) != (size_t) raw.packed) {
                ok = false;
                break;
            }
            if (!visit(r, std::move(packed), version)) {
                ok = false;
                break;
            }
        }
        std::fclose(f);
        return ok;
    }

private:
    // A payload's bytes, packed: a run of zeros is a count, the rest is a literal run.  The engine's rows hold a
    // block's compressed row once per block, so most of a payload is empty positions.
    class Packer {
    public:
        explicit Packer(FILE* f) : f_(f) {}

        void reset() { lit_.clear(); written_ = 0; }
        uint64_t written() const { return written_; }
        bool failed() const { return failed_; }

        void put(const uint8_t* p, size_t n) {
            size_t i = 0;
            while (i < n) {
                const uint8_t* z = (const uint8_t*) std::memchr(p + i, 0, n - i);
                if (!z) { add_literal(p + i, n - i); break; }
                const size_t lit = (size_t) (z - (p + i));
                if (lit) add_literal(p + i, lit);
                size_t run = 1;
                while (z + run < p + n && z[run] == 0) ++run;
                add_zeros(run);
                i += lit + run;
            }
        }

        // The last of a payload's bytes.
        void finish() { flush_lit(); }

private:
        static constexpr size_t kMinRun = 16;   // a shorter run of zeros stays literal: a count costs 8 bytes
        static constexpr size_t kMaxLit = (size_t) 1 << 16;

        void add_literal(const uint8_t* p, size_t n) {
            lit_.insert(lit_.end(), p, p + n);
            while (lit_.size() >= kMaxLit) flush_lit();
        }
        void add_zeros(size_t n) {
            if (n < kMinRun) {   // too short to pay for its count
                lit_.insert(lit_.end(), n, (uint8_t) 0);
                while (lit_.size() >= kMaxLit) flush_lit();
                return;
            }
            flush_lit();
            put_u64((uint64_t) n | (1ull << 63));
        }
        void flush_lit() {
            if (lit_.empty()) return;
            put_u64((uint64_t) lit_.size());
            if (std::fwrite(lit_.data(), 1, lit_.size(), f_) != lit_.size()) failed_ = true;
            written_ += lit_.size();
            lit_.clear();
        }
        void put_u64(uint64_t v) {
            if (std::fwrite(&v, sizeof v, 1, f_) != 1) failed_ = true;
            written_ += sizeof v;
        }

        FILE* f_ = nullptr;
        std::vector<uint8_t> lit_;
        uint64_t written_ = 0;
        bool failed_ = false;
    };

    // The payload of a record, unpacked: a version 2 payload is the bytes as they are, a version 3 payload is
    // packed.  `dst` may be null: the payload is then skipped.  `packed` is the payload's size in the file, which
    // the reader checks against the bytes it reads.
    static bool read_payload(FILE* f, uint32_t version, uint8_t* dst, uint64_t bytes, uint64_t packed) {
        if (version == kVersion) {
            if (!bytes) return true;
            if (!dst) return std::fseek(f, (long) bytes, SEEK_CUR) == 0;
            return std::fread(dst, 1, (size_t) bytes, f) == (size_t) bytes;
        }
        // The packed bytes come in blocks and the decode runs out of the block, not out of the file: a read for
        // every run would be hundreds of thousands of small reads, and a cold file then costs more than the
        // unpacked file did.  `dst` may be null: the payload is then stepped over.
        constexpr size_t kBlock = (size_t) 1 << 20;
        std::vector<uint8_t> buf(kBlock);
        size_t pos = 0, end = 0;
        uint64_t left = bytes, read = 0;
        const auto more = [&](size_t need) -> bool {
            while (end - pos < need) {
                if (pos) {   // what is left of the block to the front
                    std::memmove(buf.data(), buf.data() + pos, end - pos);
                    end -= pos;
                    pos = 0;
                }
                if (need > buf.size()) return false;   // a literal longer than the block
                if ((uint64_t) (need - end) > packed - read) return false;
                const size_t want = (size_t) std::min<uint64_t>(buf.size() - end, packed - read);
                if (!want) return false;
                if (std::fread(buf.data() + end, 1, want, f) != want) return false;
                end += want;
                read += want;
            }
            return true;
        };
        while (left) {
            if (!more(8)) return false;
            uint64_t h = 0;
            std::memcpy(&h, buf.data() + pos, 8);
            pos += 8;
            if (h & (1ull << 63)) {   // a run of zeros
                const uint64_t run = h & ~(1ull << 63);
                if (run > left) return false;
                if (dst) std::memset(dst + (bytes - left), 0, (size_t) run);
                left -= run;
                continue;
            }
            if (h > left) return false;
            if (h <= buf.size() && more((size_t) h)) {
                if (dst) std::memcpy(dst + (bytes - left), buf.data() + pos, (size_t) h);
                pos += (size_t) h;
            } else if (dst) {   // a literal longer than the block: it comes straight from the file
                if (std::fread(dst + (bytes - left), 1, (size_t) h, f) != (size_t) h) return false;
                read += h;
            } else if (std::fseek(f, (long) h, SEEK_CUR) != 0) {
                return false;
            }
            left -= h;
        }
        return read == packed;
    }

    // The payload of a record, passed over: what read_head does to reach the next record.
    static bool skip_payload(FILE* f, uint32_t version, uint64_t bytes, uint64_t packed) {
        return read_payload(f, version, nullptr, bytes, packed);
    }

    // Fills in a record's packed size, which is only known once the payload is written.  The file is a
    // temporary one until it is renamed, so a seek back is safe.
    static bool patch(FILE* f, long at, uint64_t packed) {
        if (std::fseek(f, at, SEEK_SET) != 0) return false;
        if (std::fwrite(&packed, sizeof packed, 1, f) != 1) return false;
        return std::fseek(f, 0, SEEK_END) == 0;
    }

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
