// src/poly/conversations.hpp - the conversations a server has read, kept so that a prompt which starts like one of
// them is not read again from its first token.
//
// The engine holds one sequence at a time.  Coming back to an earlier point of a sequence takes two things: the
// recurrent state at that point, which is a checkpoint, and the rows its tokens left in the attention caches.  The
// rows of the sequence in the engine are there already.  A conversation that makes room for another is parked:
// its rows are copied to host memory, up to its last checkpoint.  What is saved is written once and never changed,
// so conversations that start alike share it, and a parked conversation comes back bit for bit.  Between requests
// it leaves the process for the file cache (release(), spill.hpp).
//
// Nothing here knows the model.  E is the engine:
//     void reset();
//     size_t state_bytes() const;  void save_state(uint8_t*);  void load_state(const uint8_t*, int64_t pos);
//     size_t row_bytes() const;    void save_rows(int64_t first, int64_t n, uint8_t*);
//     void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
#pragma once
#include "transfer_audit.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "session_file.hpp"
#include "spill.hpp"

namespace strata::poly {

struct CacheOptions {
    int checkpoints = 6;       ///< kept per conversation; 0: a prompt is reread unless it goes on from the last token
    // Retained only so older launch configs remain accepted. Checkpoints are scheduled at semantic chat boundaries
    // supplied by the server, plus the prompt end for its logits; a token interval is not a scheduling rule.
    int64_t every = 0;
    int64_t park_mib = 0;      ///< host memory for parked conversations; 0: none is parked unless a session dir is
    int park_slots = 4;        ///< parked conversations at most
    int64_t park_min = 1024;   ///< a conversation with fewer tokens before its last checkpoint is not parked
    bool keep_branches = true; ///< a conversation the client changed keeps its old version as a branch of its
    ///< own; off: the old version is replaced, and the client's own branching does the work
    std::function<bool(size_t)> admit;   ///< whether the host can spare this many bytes; unset: it can
    FILE* log = nullptr;       ///< where parking is reported; the server reads these lines
    std::filesystem::path session_dir;   ///< parked conversations go to files here and free their host memory; the
                                         ///< files of earlier runs come back as parked conversations.  Empty: the
                                         ///< host memory of park_mib, as before
    int64_t session_max_mib = 16384;     ///< the session files take this much disk at most; the least recently used
                                         ///< goes first
    int fail_blob_at = 0;                ///< a test's lever: the N-th blob allocation comes out null; 0: none
};

template<class E>
class Conversations {
public:
    Conversations(E& engine, CacheOptions o)
        : e_(engine), o_(std::move(o)), slots_((size_t) std::max(o_.park_slots, 0) + 1) {
        if (!o_.session_dir.empty()) scan_disk();
    }

    /// The tokens the engine has read.
    const std::vector<int32_t>& seq() const { return slots_[cur_].seq; }

    /// Leaves the engine at the furthest point of `ids` that can be come back to, and returns that position.
    int64_t resume(const std::vector<int32_t>& ids) {
        const int64_t n = (int64_t) ids.size();
        anchors_.clear();
        size_t best = cur_;
        int64_t at = 0, same = 0;
        const auto pick = [&]() {
            best = cur_;
            at = 0, same = 0;
            for (size_t i = 0; i < slots_.size(); ++i) {
                const int64_t c = common(slots_[i].seq, ids), r = reach(i, c, n);
                if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
            }
        };
        pick();
        ++clock_;
        while (at > 0 && !slots_[best].disk.empty() && !materialize(best)) {
            slots_[best] = Slot{};   // the file is gone or not this engine's any more: look at what is left
            pick();
        }
        if (at == 0) {
            // nothing kept starts like this
            if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            const bool disk_in = best != cur_ && !slots_[best].disk.empty();
            if (best != cur_ && !enter(best, at)) {
                // the file vanished or its memory ran out while the engine came onto it: a prompt nothing kept
                // starts like, the leaving one parks if it can, as the at==0 way does
                if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
                slots_[cur_] = Slot{};
                e_.reset();
                at = 0;
            } else {
                const uint8_t* state = state_at(cur(), at);
                std::vector<uint8_t> kept_state;   // a conversation come back from a file: branch() would free the blob
                if ((disk_in || !slots_[cur_].disk.empty()) && state)
                    kept_state.assign(state, state + e_.state_bytes());
                // The conversation is left at `at`.  If the prompt the client sent for it went on from there, the
                // client may send that again, so the conversation is parked and this one goes on in a copy.
                if (!o_.keep_branches || same >= cur().prompt || !branch(at)) cut(cur(), at);
                e_.load_state(kept_state.empty() ? state_at(cur(), at) : kept_state.data(), at);
            }
        }
        cur().stamp = clock_;
        // every way resume() leaves the engine at the conversation's last token (read(), branch()/cut(), adopt());
        // the server's reads then keep the two in step until the next resume()
        cur().held = true;
        return at;
    }

    /// Semantic assistant boundaries in the prompt that resume() took up. Positions are token-prefix lengths and
    /// come from the server's rendered template, never from a guessed control token.
    void checkpoints_at(std::vector<int64_t> at) {
        anchors_ = std::move(at);
        anchors_.erase(std::remove_if(anchors_.begin(), anchors_.end(), [](int64_t p) { return p < 1; }), anchors_.end());
        std::sort(anchors_.begin(), anchors_.end());
        anchors_.erase(std::unique(anchors_.begin(), anchors_.end()), anchors_.end());
    }

    // Compatibility for callers compiled against the old single-boundary API.
    void turn_at(int64_t at) { checkpoints_at(at > 0 ? std::vector<int64_t>{at} : std::vector<int64_t>{}); }

    /// Of `want` tokens from position `at`, how many to read before the next checkpoint is due.
    int64_t until_checkpoint(int64_t at, int64_t want) const {
        if (o_.checkpoints < 1) return want;
        const auto next = std::upper_bound(anchors_.begin(), anchors_.end(), at);
        return next != anchors_.end() ? std::min(want, *next - at) : want;
    }

    /// Before the engine reads the token at `at`; `last` says it is the prompt's last.  Takes a checkpoint if one is
    /// due.  One sits before the last token of every prompt, so coming back to it leaves a token to read and the
    /// logits that come with it.
    void before(int64_t at, bool last) {
        if (o_.checkpoints < 1 || at < 1 || (!last && !std::binary_search(anchors_.begin(), anchors_.end(), at))) return;
        if (const BlobPtr b = mark(at)) e_.save_state(b->data());
    }

    /// For an engine that reads past checkpoints in one go, of n tokens from `at`: the checkpoints due after the
    /// read's first token.  `out` gets, for each, how many of the read's tokens come before it and where its
    /// state_bytes() go; null where no state is wanted, and the engine still breaks its work there as if one
    /// were.  The engine writes a state when it has it.  The checkpoints are the conversation's once read() has
    /// the read's tokens, and not at all when the read is given up (rewind()).
    ///
    /// A state is wanted for the checkpoints the conversation would still have, had each been taken in turn as
    /// before() takes one.  A long prompt has more due than are kept, and a state can be hundreds of MiB.
    void inside(int64_t at, int64_t n, std::vector<std::pair<int64_t, uint8_t*>>& out) {
        out.clear();
        due_.clear();
        after_.clear();
        if (o_.checkpoints < 1) return;
        std::vector<int64_t> due;
        for (const int64_t p : anchors_)
            if (p > at && p < at + n) due.push_back(p);
        if (due.empty()) return;
        for (const Checkpoint& m : cur().marks) after_.push_back(m.pos);
        for (const int64_t p : due) {
            after_.push_back(p);
            while (after_.size() > (size_t) o_.checkpoints) after_.erase(after_.begin() + (std::ptrdiff_t) drop_of(after_));
        }
        for (const int64_t p : due) {
            const bool stays = std::find(after_.begin(), after_.end(), p) != after_.end();
            due_.push_back({p, stays ? blob(e_.state_bytes()) : nullptr});
            out.emplace_back(p - at, due_.back().state ? due_.back().state->data() : nullptr);
        }
    }

    /// The engine has read these tokens.  A read that inside() was asked about is done with them, and its
    /// checkpoints take their places.
    void read(const int32_t* t, int64_t n) {
        Slot& s = cur();
        s.seq.insert(s.seq.end(), t, t + n);
        if (due_.empty()) return;
        const auto stays = [&](const Checkpoint& m) {
            return m.state && std::find(after_.begin(), after_.end(), m.pos) != after_.end();
        };
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (stays(m)) marks.push_back(m);
        for (const Checkpoint& m : due_)
            if (stays(m)) marks.push_back(m);
        s.marks = std::move(marks);
        due_.clear();
    }

    /// A read was given up part way: what it left in the engine is not the conversation's, and the checkpoints
    /// inside it are not taken.  The conversation goes back to its last checkpoint, which the engine is left
    /// at, or is forgotten when it has none.  Returns where the conversation is then.
    int64_t rewind() {
        due_.clear();
        Slot& s = cur();
        if (s.marks.empty()) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        const int64_t at = s.marks.back().pos;
        cut(s, at);
        e_.load_state(s.marks.back().state->data(), at);
        return at;
    }
    /// The first n tokens of the conversation are the prompt the client sent.
    void prompt_was(int64_t n) { cur().prompt = std::min(n, len(cur())); }
    /// After an engine error: the conversation being read is lost.  The caller resets the engine.
    void drop() {
        slots_[cur_] = Slot{};
        due_.clear();
    }

    /// The engine's sequence is about to hold another conversation, read elsewhere (a batch slot's copy): the one
    /// it holds is parked, as resume() parks it, or forgotten.  adopt() then says what it holds.
    void leave() {
        if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
        slots_[cur_] = Slot{};
        due_.clear();
    }
    /// The engine holds `seq` now, with the state at its end: a prompt that goes on from there reads none of it.
            void adopt(std::vector<int32_t> seq) {
        Slot& s = cur();
        s = Slot{};
        s.seq = std::move(seq);
        s.prompt = len(s);
        s.stamp = ++clock_;
        s.held = true;
    }

    /// Whether the current conversation is complete: tokens, and one checkpoint to come back to.  This is what
    /// the line protocol's SAVE needs (the Qwen engine refuses the save without it).
    bool complete() const {
        const Slot& s = slots_[cur_];
        return !s.seq.empty() && !s.marks.empty() && s.marks.back().pos > 0;
    }

    enum class SOut { saved, empty, io, storage, memory };
    enum class SLoad { ok, missing, unreadable, foreign, no_ram };

    /// SAVE <path>: the current conversation whole.  Its rows go to the engine's end and the state there is in
    /// the file, so a return from a restart reads the conversation on from its last token.  The blobs the host
    /// lacks are captured from the engine now, and the conversation keeps them (a later park has the copies in
    /// already).  `emit(done, total)`, when given, is the write's progress in payload bytes.  `empty`: there is
    /// nothing complete to save; `storage`: the disk did not hold the file (the server's 507); `io`: the file
    /// could not be written (the caller reports `file_bytes` and `published` either way, as the server's line
    /// protocol needs, and `flaw`, when given, as the write's own words about it).
    SOut snapshot(const std::filesystem::path& dir, const std::string& name,
                  const std::function<void(uint64_t, uint64_t)>& emit, size_t& file_bytes, bool& published,
                  const char** flaw = nullptr) {
        file_bytes = 0;
        published = false;
        if (flaw) *flaw = nullptr;
        if (!cur().disk.empty() && !materialize(cur_)) {   // the blobs are still in the conversation's file
            slots_[cur_] = Slot{};
            return SOut::empty;
        }
        Slot& s = cur();
        if (!complete()) return SOut::empty;
        const int64_t end = len(s), from = saved(s);
        std::vector<SessionFile::StreamRows> tail_rows;
        if (from < end) {   // the rows the engine still holds alone
            if (o_.session_dir.empty()) {
                // the host is where a conversation's rows live: it takes a copy, and the save and the slot
                // both have it
                const BlobPtr fresh = blob((size_t) (end - from) * e_.row_bytes());
                if (!fresh) return SOut::memory;   // the host could not hold the rows' copy: the server's memory kind
                e_.save_rows(from, end - from, fresh->data());
                s.rows.push_back({fresh, from, end - from, end - from});
            } else {
                // the file is where the rows live: they go to it in chunks, straight from the engine, and the
                // host takes no copy of them
                const size_t rb = e_.row_bytes();
                const int64_t chunk = rb ? (int64_t) std::max<size_t>(1, ((size_t) 4 << 20) / rb) : end - from;
                for (int64_t first = from; first < end; ) {
                    const int64_t count = std::min(chunk, end - first);
                    SessionFile::SessionPart tail;
                    tail.bytes = (uint64_t) count * rb;
                    // A capture is indivisible; splitting it would change the engine's layer strides.
                    tail.align = tail.bytes;
                    tail.fill = [this, first, count, rb](void* dst, size_t off, size_t n) {
                        if (off != 0 || n != (size_t) count * rb) return false;
                        e_.save_rows(first, count, (uint8_t*) dst);
                        return true;
                    };
                    tail_rows.push_back({first, count, count, tail});
                    first += count;
                }
            }
        }
        if (s.marks.back().pos != end) {   // the state of the conversation's end, a checkpoint of the engine's now
            const BlobPtr hold = blob(e_.state_bytes());
            if (!hold) return SOut::memory;
            e_.save_state(hold->data());
            s.marks.push_back({end, hold});
        }
        std::vector<std::pair<int64_t, SessionFile::SessionPart>> parts;
        parts.reserve(s.marks.size());
        for (const Checkpoint& m : s.marks)
            parts.push_back({m.pos, SessionFile::part_of(m.state->data(), m.state->bytes())});
        std::vector<SessionFile::StreamRows> rows;
        rows.reserve(s.rows.size());
        for (const Rows& r : s.rows)
            rows.push_back({r.first, r.count, r.keep, SessionFile::part_of(r.blob->data(), r.blob->bytes())});
        for (const SessionFile::StreamRows& r : tail_rows) rows.push_back(r);
        SessionFile::WStat st;
        const bool ok = SessionFile::write_stream(dir, name, s.prompt, e_.state_bytes(), e_.row_bytes(), s.seq,
                                                  parts, rows, &st, emit);
        file_bytes = st.bytes;
        published = st.published;
        if (o_.log) {
            const Held h = held(cur_);
            std::fprintf(o_.log, "strata-poly: session saved %s: %lld tokens, %zu marks, %zu bytes to %s\n",
                         ok ? "done" : st.dir_flush_unsupported ? "done without the folder's flush" : "refused",
                         (long long) end, s.marks.size(), st.bytes, name.c_str());
            std::fprintf(o_.log,
                         "strata-poly: conversation cache: parked=%zu bytes=%zu evictions=%zu\n",
                         h.count, h.bytes, evictions_);
        }
        if (flaw) *flaw = st.fail;
        return ok ? SOut::saved : st.no_room ? SOut::storage : SOut::io;
    }

    /// RESTORE <path>: the file's conversation takes the engine whole; the outgoing one is given up, as the
    /// Qwen engine's restore does (the parked ones stay).  `floor` is the RAM floor, the conversation cache's
    /// (`conversation-cache-min-free-mib`).  What is asked before the engine is touched: the parse's peak and
    /// then the blobs, that copy the payloads Р Р†Р вЂљРІР‚Сњ short twice (a first short gives the sessions' copies up, as
    /// Qwen's reclaim does), or the file unknown, not this engine's format any more, or no RAM, is a refusal
    /// `err` says.  The engine's writes begin after; a failure there is not this method's to answer Р Р†Р вЂљРІР‚Сњ the
    /// caller stops the engine, for a device state change began.  `touched` is set once the engine's writes
    /// begin: the caller answers a failure after it with the protocol's fatal line, and refuses before it.
    SLoad sideload(const std::filesystem::path& path, int64_t floor_mib, std::string& err, int64_t& tokens,
                   size_t& file_bytes, bool& touched) {
        tokens = 0;
        file_bytes = 0;
        touched = false;
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            err = "not found";
            return SLoad::missing;
        }
        file_bytes = (size_t) size;
        const std::string why = std::to_string(size >> 20) + " MiB plus a floor of " +
                                std::to_string(floor_mib) + " MiB needed";
        std::optional<SessionFile::Head> head = SessionFile::read_head(path);
        if (!head || head->seq.empty()) {
            err = "unreadable: not the format this engine writes";
            return SLoad::unreadable;
        }
        if (head->sb != e_.state_bytes() || head->rb != e_.row_bytes()) {   // not this engine's any more
            err = "the model or its shapes have changed since the file was written";
            return SLoad::unreadable;
        }
        // The host keeps a conversation's rows only when the host is where a parked conversation's rows live.
        // With a session dir the rows are in the file and in the engine, so the file's records are read one at a
        // time and the host lets go of each one as it goes into the engine: a restore then needs room for one
        // record, not for the whole conversation.
        const bool keep_rows = o_.session_dir.empty();
        const size_t ask = keep_rows ? (size_t) head->payload
                                     : (size_t) (head->marks_bytes + head->max_record);
        if (o_.admit && !o_.admit(ask)) {
            size_t was = 0;   // the sessions' copies go up; the file replaces the sessions wholesale, as Qwen does
            for (const Slot& z : slots_) was += droppings(z);
            for (Slot& z : slots_) z = Slot{};
            err = "not enough RAM to read it (" + why + ", " + std::to_string(was >> 20) +
                  " MiB of the sessions' copies was given up for it)";
            if (o_.admit && !o_.admit(ask)) return SLoad::no_ram;
        }
        return mount(path, *head, err, tokens, (size_t) size, touched);
    }

    /// The bytes the engine holds that the SAVE of the whole session takes: the seq, the rows it did not save
    /// yet, and one state more than the checkpoints' count (a save takes the state of the conversation's end).
    size_t save_bytes() const {
        const Slot& s = slots_[cur_];
        return s.seq.size() * 4 + (size_t) (len(s) - saved(s)) * e_.row_bytes() +
               (s.marks.size() + 1) * e_.state_bytes() + 128;
    }

    /// Between requests: what the parked conversations hold leaves the process.  On the laptop four parked
    /// conversations of Qwen3.6 held 1.1 to 1.7 GB, one of 29K tokens 1.05 GB.  The one in the engine keeps its
    /// states in memory: every request saves one (63 MB of Qwen3.6's), and the next request most likely goes on
    /// from it, so written to the disk it would mostly be read back at once.
    void release() const {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == cur_) continue;
            for (const Checkpoint& m : slots_[i].marks) m.state->release();
            for (const Rows& r : slots_[i].rows) r.blob->release();
        }
    }

    /// The parked conversations, the bytes they hold, and how many were pushed out so far.
    size_t parked() const { return held(slots_.size()).count; }
    size_t parked_bytes() const { return held(slots_.size()).bytes; }
    size_t evictions() const { return evictions_; }

private:
    // sideload's placement: the conversation's blobs from the file (all the RAM questions answered before),
    // the engine's writes last.  `tokens` is what the engine holds, the file's conversation when its rows reach
    // its end and the deepest checkpoint row-covered when they do not (its tail is the next prompts' to redo).
    SLoad mount(const std::filesystem::path& path, SessionFile::Head& head, std::string& err, int64_t& tokens,
                size_t file_bytes, bool& touched) {
        // The host keeps a conversation's rows only when the host is where a parked conversation's rows live.
        // With a session dir the rows are in the file and in the engine, so the file's records are read one at a
        // time and the host lets go of each one as it goes into the engine: a restore then needs room for one
        // record, not for the whole conversation.  Without a session dir the host is where the rows live, and
        // they are read as they were.
        if (o_.session_dir.empty()) return mount_blobs(path, err, tokens, file_bytes, touched);
        const int64_t rows_last = head.rows_last;
        // the file's rows reach to the deepest checkpoint that the rows cover: the marks past them are dropped
        int64_t end = 0;
        for (const int64_t p : head.mark_pos)
            if (p <= rows_last) end = std::max(end, p);
        if (!end) {
            err = "unreadable: the file holds checkpoints no row covers";
            return SLoad::unreadable;
        }
        std::vector<Checkpoint> marks;
        std::vector<Rows> rows;
        bool no_ram = false;
        bool engine_touched = false;
        const bool ok = SessionFile::read_packed(
            path, head, [&](const SessionFile::Record& r, std::vector<uint8_t>&& packed, uint32_t version) {
                if (r.kind == SessionFile::Record::kMark) {
                    const BlobPtr b = blob(r.raw);
                    if (!b) {
                        no_ram = true;
                        return false;
                    }
                    if (!SessionFile::unpack(version, packed.data(), packed.size(), b->data(), r.raw)) return false;
                    marks.push_back({r.pos, b});
                    return true;
                }
                const int64_t a = r.first, b = std::min(r.first + r.keep, end);
                if (a >= b) return true;   // the rows past the deepest checkpoint are not the conversation's
                if (!engine_touched) {
                    // the engine is written to from here on: a failure after this is not this method's to answer
                    engine_touched = true;
                    touched = true;
                    e_.reset();
                    slots_[cur_] = Slot{};
                }
                // the rows go into the engine now, and the host lets go of them: the file is where they live
                std::vector<uint8_t> tmp((size_t) r.raw);
                if (!SessionFile::unpack(version, packed.data(), packed.size(), tmp.data(), r.raw)) return false;
                e_.load_rows(tmp.data(), r.first, r.count, a, b - a);
                return true;
            });
        if (!ok) {
            err = no_ram ? "not enough RAM to read it: the file's payloads would not stay in RAM"
                          : "unreadable: not the format this engine writes";
            return no_ram ? SLoad::no_ram : SLoad::unreadable;
        }
        if (!engine_touched) {
            err = "unreadable: the file holds no rows";
            return SLoad::unreadable;
        }
        Slot s;
        s.seq = std::move(head.seq);
        s.prompt = head.prompt;
        s.stamp = ++clock_;
        s.marks = std::move(marks);
        s.rows = std::move(rows);
        cut(s, rows_last);              // the file's tail past its rows is the next prompts' to read again
        if (s.marks.empty()) {          // the chain did not reach a row-covered place
            err = "unreadable: the file holds checkpoints no row covers";
            return SLoad::unreadable;
        }
        const int64_t at = s.marks.back().pos;   // the state at the deepest row-covered checkpoint
        e_.load_state(s.marks.back().state->data(), at);
        const bool at_end = at == rows_last && at == len(s);   // the rows and the state both reach the last token
        s.held = at_end;
        slots_[cur_] = std::move(s);
        tokens = len(slots_[cur_]);
        if (o_.log)
            std::fprintf(o_.log,
                         "strata-poly: session restored %lld tokens of %zu bytes, %zu marks, the engine at "
                         "%lld%s\n",
                         (long long) tokens, file_bytes, slots_[cur_].marks.size(), (long long) at,
                         at_end ? "" : " (the file's tail is cut where its rows end)");
        return SLoad::ok;
    }

    // A restore without a session dir: the host is where a conversation's rows live, so they are all read first
    // and the engine is written to afterwards, so that a refusal touches no engine state.
    SLoad mount_blobs(const std::filesystem::path& path, std::string& err, int64_t& tokens, size_t file_bytes,
                      bool& touched) {
        std::vector<int32_t> seq;
        int64_t prompt = 0;
        uint64_t sb = 0, rb = 0;
        std::vector<Checkpoint> marks;
        std::vector<Rows> rows;
        bool no_ram = false;
        if (!load_blobs(path, seq, prompt, sb, rb, marks, rows, no_ram)) {
            if (no_ram) {
                err = "not enough RAM to read it: the file's payloads would not stay in RAM";
                return SLoad::no_ram;
            }
            err = "unreadable: not the format this engine writes";
            return SLoad::unreadable;
        }
        int64_t rows_last = 0;
        for (const Rows& r : rows) rows_last = std::max(rows_last, r.first + r.keep);
        Slot s;
        s.seq = std::move(seq);
        s.prompt = prompt;
        s.stamp = ++clock_;
        s.marks = std::move(marks);
        s.rows = std::move(rows);
        cut(s, rows_last);              // the file's tail past its rows is the next prompts' to read again
        if (s.marks.empty()) {          // the chain did not reach a row-covered place
            err = "unreadable: the file holds checkpoints no row covers";
            return SLoad::unreadable;
        }
        const int64_t end = s.marks.back().pos;   // the state at the deepest row-covered checkpoint
        // from here on the engine is written to: a failure ends it, the caller stops the server off this
        touched = true;
        e_.reset();
        slots_[cur_] = Slot{};
        for (const Rows& r : s.rows) {
            const int64_t a = r.first, b = std::min(r.first + r.keep, end);
            if (a < b) e_.load_rows(r.blob->data(), r.first, r.count, a, b - a);
        }
        e_.load_state(s.marks.back().state->data(), end);
        const bool at_end = end == rows_last && end == len(s);   // the rows and the state both reach the last token
        s.held = at_end;
        slots_[cur_] = std::move(s);
        tokens = len(slots_[cur_]);
        if (o_.log)
            std::fprintf(o_.log,
                         "strata-poly: session restored %lld tokens of %zu bytes, %zu marks, the engine at "
                         "%lld%s\n",
                         (long long) tokens, file_bytes, slots_[cur_].marks.size(), (long long) end,
                         at_end ? "" : " (the file's tail is cut where its rows end)");
        return SLoad::ok;
    }


private:
    using Clock = std::chrono::steady_clock;

    using Blob = Spill;
    using BlobPtr = std::shared_ptr<const Blob>;
    struct Checkpoint {
        int64_t pos;
        BlobPtr state;
    };
    // The blob holds `count` rows from row `first`; the first `keep` of them are this conversation's.
    struct Rows {
        BlobPtr blob;
        int64_t first, count, keep;
    };

    // The file's payloads, read straight into blobs: the file's bytes are never in RAM twice.  False: the
    // file is not readable (the caller says so) or there is no memory for a payload (`no_ram` says which).
    bool load_blobs(const std::filesystem::path& path, std::vector<int32_t>& seq, int64_t& prompt,
                    uint64_t& sb, uint64_t& rb, std::vector<Checkpoint>& marks, std::vector<Rows>& rows,
                    bool& no_ram) {
        no_ram = false;
        SessionFile::Head head;
        const auto take = [&](const SessionFile::Record& r) -> uint8_t* {
            const BlobPtr b = blob((size_t) r.raw);
            if (!b) {
                no_ram = true;
                return nullptr;
            }
            if (r.kind == SessionFile::Record::kMark) marks.push_back({r.pos, b});
            else rows.push_back({b, r.first, r.count, r.keep});
            return b->data();
        };
        if (!SessionFile::read_into(path, head, take)) return false;
        seq = std::move(head.seq);
        prompt = head.prompt;
        sb = head.sb;
        rb = head.rb;
        return true;
    }
    struct Slot {
        std::vector<int32_t> seq;
        int64_t prompt = 0;             // seq's first tokens that the client sent
        std::vector<Checkpoint> marks;  // by position
        std::vector<Rows> rows;         // back to back from row 0; a parked conversation has all of its rows
        uint64_t stamp = 0;
        bool held = false;              ///< the engine sits at this conversation's last token
        std::string disk;               // the session file that holds the blobs; empty: they are here
        std::vector<int64_t> disk_marks;  // where that file's checkpoints are, while its blobs are in it
        bool materialized = false;      ///< the file's checkpoints are in the host, its rows are still in the file
        // One file of the chain a conversation's rows come from: a file may hold only what came after its
        // parent's rows, and the chain of them is the conversation.
        struct File {
            std::string name;      // the file's name in the session dir
            uint64_t rows_at = 0;  // where its rows' records start
            uint64_t n_rows = 0;   // how many rows' records it holds
            uint32_t version = 0;  // the file's version
            uint64_t rows_from = 0;  // where this file's rows begin: the parent's part ends there
        };
        std::vector<File> chain;        // the files the rows come from, the root first
        uint64_t disk_rows_end = 0;     // where the rows in the files reach
        size_t disk_bytes = 0;
    };
    struct Held {
        size_t count = 0, bytes = 0;
    };

    // The bytes the host holds for a slot (its session-file blobs are on the disk and count nothing).  A blob
    // parked conversations share between them counts once.
    static size_t droppings(const Slot& s) {
        std::vector<const Blob*> seen;
        for (const Checkpoint& m : s.marks) seen.push_back(m.state.get());
        for (const Rows& r : s.rows) seen.push_back(r.blob.get());
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        size_t b = 0;
        for (const Blob* p : seen) b += p->bytes();
        return b;
    }

    Slot& cur() { return slots_[cur_]; }
    const Slot& cur() const { return slots_[cur_]; }
    static int64_t len(const Slot& s) { return (int64_t) s.seq.size(); }
    static int64_t saved(const Slot& s) { return s.rows.empty() ? 0 : s.rows.back().first + s.rows.back().keep; }

    static int64_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    }

    // How far into a prompt of n tokens slot i gets it, given that they start with `same` tokens alike.  A slot
    // whose blobs are in its session file reaches the positions of the file's checkpoints.
    int64_t reach(size_t i, int64_t same, int64_t n) const {
        const Slot& s = slots_[i];
        if (i == cur_ && s.held && same > 0 && same == len(s) && n > same) return same;   // the engine is right there
        int64_t at = 0;
        if (!s.disk.empty()) {
            for (const int64_t p : s.disk_marks)
                if (p <= std::min(same, n - 1)) at = p;
            return at;
        }
        for (const Checkpoint& m : s.marks)
            if (m.pos <= std::min(same, n - 1)) at = m.pos;
        return at;
    }

    static const uint8_t* state_at(const Slot& s, int64_t at) {
        for (const Checkpoint& m : s.marks)
            if (m.pos == at) return m.state->data();
        return nullptr;   // resume() only goes to positions reach() found
    }

    // A blob of `bytes`, or null: the host did not have them (or a test's lever asked this one to fail).
    BlobPtr blob(size_t bytes) {
        if (o_.fail_blob_at > 0 && ++blobs_ == o_.fail_blob_at) return nullptr;
        auto b = std::make_shared<Blob>(bytes);
        return b->data() ? b : nullptr;
    }

    // A checkpoint of the current conversation at `at`, its state still to be written.  Null: there is one there,
    // or no memory for it.
    BlobPtr mark(int64_t at) {
        Slot& s = cur();
        if (!s.marks.empty() && s.marks.back().pos == at) return nullptr;
        const BlobPtr b = blob(e_.state_bytes());
        if (!b) return nullptr;
        s.marks.push_back({at, b});
        while (s.marks.size() > (size_t) o_.checkpoints) {
            std::vector<int64_t> pos;
            for (const Checkpoint& m : s.marks) pos.push_back(m.pos);
            s.marks.erase(s.marks.begin() + (std::ptrdiff_t) drop_of(pos));
        }
        return b;
    }

    // Of more checkpoints than are kept, the one that goes: the one whose neighbours are closest together.  The
    // last two stay, the end of this prompt and of the one before being where a client most often comes back to.
    static size_t drop_of(const std::vector<int64_t>& pos) {
        size_t drop = 0;
        int64_t gap = INT64_MAX;
        for (size_t k = 0; k + 2 < pos.size(); ++k) {
            const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
            if (g < gap) gap = g, drop = k;
        }
        return drop;
    }

    static void cut(Slot& s, int64_t at) {
        if (at >= len(s)) return;
        s.seq.resize((size_t) at);
        s.prompt = std::min(s.prompt, at);
        while (!s.marks.empty() && s.marks.back().pos > at) s.marks.pop_back();
        while (!s.rows.empty() && s.rows.back().first >= at) s.rows.pop_back();
        if (!s.rows.empty()) s.rows.back().keep = std::min(s.rows.back().keep, at - s.rows.back().first);
        // The conversation stops here: the rows past it are not its own, and the file it adds to is used up
        // to here as well.
        if (s.disk_rows_end > (uint64_t) at) s.disk_rows_end = (uint64_t) at;
        while (!s.disk_marks.empty() && s.disk_marks.back() > at) s.disk_marks.pop_back();
    }

    // The parked conversations and what they hold; a blob several of them share counts once.  The current one
    // counts when `with` is its slot.
    Held held(size_t with) const {
        Held h;
        std::vector<const Blob*> seen;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot& s = slots_[i];
            if (s.seq.empty() || (i == cur_ && i != with)) continue;
            ++h.count;
            for (const Checkpoint& m : s.marks) seen.push_back(m.state.get());
            for (const Rows& r : s.rows) seen.push_back(r.blob.get());
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const Blob* b : seen) h.bytes += b->bytes();
        return h;
    }

    // Makes the parked conversations, the current one among them and `extra` bytes more, fit the budget and the
    // slots.  The least recently used goes first, whatever its length: a long conversation can lose its place to
    // several short ones.  Slot `keep` stays; it is the one about to take the engine, if any, so it frees a slot.
    // False: it does not fit, or the host cannot spare the bytes.
    bool fit(size_t extra, size_t keep) {
        const size_t budget = (size_t) o_.park_mib << 20;
        for (;;) {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            if (h.bytes + extra <= budget && h.count - leaving <= (size_t) o_.park_slots) break;
            size_t out = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty() &&
                    (out == slots_.size() || slots_[i].stamp < slots_[out].stamp))
                    out = i;
            if (out == slots_.size()) return false;
            unfile(slots_[out]);
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Copies the rows the current conversation still has only in the engine, up to its last checkpoint; what it
    // read after that cannot be come back to and is dropped.  With a session dir the whole conversation goes to a
    // file there instead of the host memory.  False: it cannot be kept.
    bool park(size_t keep) {
        audit::Scope audit_phase("park");
        audit::HostSpan audit_span(__FILE__, __LINE__, "session-park-total");
        Slot& s = cur();
        const bool disk = !o_.session_dir.empty();
        if ((o_.park_mib < 1 && !disk) || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min)
            return false;
        const Clock::time_point t0 = Clock::now();
        const int64_t end = s.marks.back().pos, from = saved(s);
        cut(s, end);
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = !disk && from < end && fit(bytes, keep) ? blob(bytes) : nullptr;
        if (!disk && from < end && !b) {
            say("skipped", end, t0, slots_.size());
            return false;
        }
        if (b) {
            e_.save_rows(from, end - from, b->data());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (disk) {
            // The file takes the whole chain and all the rows, streamed from where they live: the blobs exist
            // and the engine reads its own tail straight to the file, so nothing is copied whole into RAM.
            // When the slot has a file already, the new file holds only what came after that one's rows: it is
            // a delta, and it names the file it adds to.
            const uint64_t held = s.disk_rows_end;   // where the rows in the slot's files reach
            const std::string name = SessionFile::name_for(s.seq, ++park_serial_);
            // A file that would be written over its own parent is written whole: the name is the conversation's,
            // and a delta's parent has to be another file.
            const bool whole = !held || name == s.disk;
            const int64_t first_new = whole ? from : std::max<int64_t>(from, (int64_t) held);
            const std::string parent = whole ? std::string() : s.disk;
            std::vector<std::pair<int64_t, SessionFile::SessionPart>> parts;
            parts.reserve(s.marks.size());
            for (const Checkpoint& m : s.marks) {
                // the checkpoints the slot's file already has are in that file
                if (!whole && std::find(s.disk_marks.begin(), s.disk_marks.end(), m.pos) != s.disk_marks.end())
                    continue;
                parts.push_back({m.pos, SessionFile::part_of(m.state->data(), m.state->bytes())});
            }
            std::vector<SessionFile::StreamRows> rows;
            rows.reserve(s.rows.size() + 1);
            if (whole) {   // the rows the host holds are the file's
                for (const Rows& r : s.rows) {
                    rows.push_back({r.first, r.count, r.keep, SessionFile::part_of(r.blob->data(), r.blob->bytes())});
                }
            }
            // save_rows() is opaque and normally layer-major: two independently saved spans cannot be
            // concatenated into one rows record. Give each bounded capture its own count and record, so
            // load_rows() uses exactly the layout save_rows() wrote, also for a partial restore of that record.
            const size_t rb = e_.row_bytes();
            const int64_t chunk = rb ? (int64_t) std::max<size_t>(1, ((size_t) 4 << 20) / rb) : end - from;
            for (int64_t first = first_new; first < end; ) {
                const int64_t count = std::min(chunk, end - first);
                SessionFile::SessionPart tail;
                tail.bytes = (uint64_t) count * rb;
                // A capture is indivisible; splitting it would change the engine's layer strides.
                tail.align = tail.bytes;
                tail.fill = [this, first, count, rb](void* dst, size_t off, size_t n) {
                    if (off != 0 || n != (size_t) count * rb) return false;
                    e_.save_rows(first, count, (uint8_t*) dst);
                    return true;
                };
                rows.push_back({first, count, count, tail});
                first += count;
            }
            SessionFile::WStat st;
            if (!whole && rows.empty() && parts.empty()) {
                // nothing new: the file stays as it is, and the conversation keeps it
                s.marks.clear();
                s.rows.clear();
                s.held = false;
                say("parked", end, t0, cur_);
                return true;
            }
            if (SessionFile::write_stream(o_.session_dir, name, s.prompt, e_.state_bytes(), e_.row_bytes(), s.seq,
                                          parts, rows, &st, {}, parent, (uint64_t) first_new)) {
                s.disk = name;
                for (const Checkpoint& m : s.marks) {
                    if (std::find(s.disk_marks.begin(), s.disk_marks.end(), m.pos) == s.disk_marks.end())
                        s.disk_marks.push_back(m.pos);
                }
                s.disk_bytes = whole ? st.bytes : s.disk_bytes + st.bytes;   // the chain's size
                s.disk_rows_end = (uint64_t) end;
                s.chain.clear();
                s.marks.clear();
                s.rows.clear();
                disk_fit(keep);
                if (o_.log)
                    std::fprintf(o_.log, "strata-poly: session file %s: %zu bytes\n", name.c_str(), st.bytes);
                say("parked", end, t0, cur_);
                s.held = false;
                return true;
            }
            if (o_.log) std::fprintf(o_.log, "strata-poly: session file %s could not be written\n", name.c_str());
            return false;   // as when the host memory does not fit: the caller forgets the conversation
        }
        if (!fit(0, keep)) return false;
        say("parked", end, t0, cur_);
        s.held = false;
        return true;
    }

    // An empty slot other than `keep`.
    size_t take(size_t keep) {
        size_t out = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == keep) continue;
            if (slots_[i].seq.empty()) return i;
            if (out == slots_.size() || slots_[i].stamp < slots_[out].stamp) out = i;
        }
        unfile(slots_[out]);
        slots_[out] = Slot{};   // park() left one empty, so this is not reached
        ++evictions_;
        return out;
    }
    // Slot i takes the engine, with its rows before `need` back in the caches.  The conversation that leaves is
    // parked if it can be.  Rows that both have from the same blob are in the caches already.  False: the
    // incoming conversation's file would not come in; the engine and the leaving conversation stand as they were.
    bool enter(size_t i, int64_t need) {
        const Clock::time_point t0 = Clock::now();
        if (!slots_[i].materialized && !slots_[i].disk.empty() && !materialize(i)) {
            slots_[i] = Slot{};
            return false;
        }
        e_.reset();   // the leaving conversation's state, ahead positions and the row-pool lower bound go first
        int64_t have = 0;
        for (size_t k = 0; k < slots_[i].rows.size() && k < cur().rows.size() &&
                           slots_[i].rows[k].blob == cur().rows[k].blob; ++k) {
            have = slots_[i].rows[k].first + std::min(slots_[i].rows[k].keep, cur().rows[k].keep);
            if (slots_[i].rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || !park(i)) slots_[cur_] = Slot{};
        if (!slots_[i].disk.empty()) {
            // the rows come from the file, one record at a time: the leaving conversation was parked before
            // this, so the engine's rows are still the leaving one's where the park read them
            if (!load_rows_from_file(i, need)) {
                slots_[i] = Slot{};
                return false;
            }
            // the file is still the conversation's home: the rows are in it, and the next park adds to it
            slots_[i].materialized = false;
        } else {
            for (const Rows& r : slots_[i].rows) {
                const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
                if (a < b) e_.load_rows(r.blob->data(), r.first, r.count, a, b - a);
            }
        }
        cur_ = i;
        slots_[i].held = true;
        say("restored", need, t0, slots_.size());
        return true;
    }

    // Parks the current conversation whole and goes on at `at` in a copy of its start, which shares what is
    // saved.  False: it cannot be parked.
    bool branch(int64_t at) {
        const size_t from = cur_;
        // the copy keeps its start's checkpoints and rows: taken before park() clears them, which park()
        // does when the session dir takes them into a file, else the copy would be a sequence nothing
        // comes back to
        std::vector<Checkpoint> marks;
        std::vector<Rows> rows;
        {
            const Slot& a = slots_[from];
            for (const Checkpoint& m : a.marks)
                if (m.pos <= at) marks.push_back(m);
            for (const Rows& r : a.rows)
                if (r.first < at) rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        }
        if (!park(slots_.size())) return false;
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        b.marks = std::move(marks);
        b.rows = std::move(rows);
        // The copy continues the same conversation up to the point of the divergence: its rows are the
        // original's up to there, and the file it adds to holds them.  The rows past the divergence are the
        // original's own, and the copy does not have them.
        b.disk = a.disk;
        b.chain = a.chain;
        b.disk_rows_end = (uint64_t) at;
        for (const int64_t p : a.disk_marks)
            if (p <= at) b.disk_marks.push_back(p);
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

    // The names the files of the session dir name as their parents: those files are a part of a chain, and
    // they are not conversations of their own.
    std::vector<std::string> parent_names() const {
        std::vector<std::string> names;
        std::error_code ec;
        std::filesystem::directory_iterator it(o_.session_dir, ec), end;
        for (; it != end && !ec; it.increment(ec)) {
            if (!it->is_regular_file(ec) || it->path().extension() != ".bin") continue;
            const std::optional<SessionFile::Head> h = SessionFile::read_head(it->path());
            if (h && !h->parent.empty()) names.push_back(h->parent);
        }
        return names;
    }

    // The session files earlier runs left in the session dir become parked conversations: their headers are read
    // (the tokens and where the checkpoints sit), the payloads when one of them takes the engine.  A file that
    // another file adds to is a part of that one's chain, not a conversation of its own.  More files than
    // there are empty slots stay unlooked at; they still come back after a restart.
    void scan_disk() {
        std::error_code ec;
        std::filesystem::directory_iterator it(o_.session_dir, ec), end;
        if (ec) return;
        std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
        for (; it != end && !ec; it.increment(ec))
            if (it->is_regular_file(ec) && it->path().extension() == ".bin") files.emplace_back(it->last_write_time(ec), it->path());
        if (ec) return;
        // The headers: what each file holds and what it adds to.  A file that another file continues strictly
        // (the child's rows begin where the parent's end) is a part of that one's chain; a file another file
        // only branches from (the child's rows begin earlier) is a conversation of its own as well.
        struct Info {
            std::filesystem::path path;
            std::filesystem::file_time_type when;
            uint64_t rows_last = 0;
            uint64_t branch_at = 0;
            uint32_t version = 0;
            std::string parent;
        };
        std::vector<Info> infos;
        for (const auto& [when, path] : files) {
            const std::optional<SessionFile::Head> h = SessionFile::read_head(path);
            if (!h || h->seq.empty()) continue;
            infos.push_back({path, when, (uint64_t) h->rows_last, h->branch_at, h->version, h->parent});
        }
        std::vector<std::string> continued;   // the files that are a part of another's chain
        for (const Info& child : infos) {
            if (child.parent.empty()) continue;
            for (const Info& parent : infos) {
                if (parent.path.filename().string() != child.parent) continue;
                const bool strict = child.version == SessionFile::kVersionBranch
                                        ? (child.branch_at && child.branch_at == parent.rows_last)
                                        : child.version == SessionFile::kVersionDelta;
                if (strict) continued.push_back(child.parent);
            }
        }
        std::sort(files.begin(), files.end(), std::greater<>());   // the newest first
        for (const auto& [when, path] : files) {
            size_t at = slots_.size();
            for (size_t i = 1; i < slots_.size(); ++i)   // the current slot stays empty: the engine holds nothing
                if (slots_[i].seq.empty()) {
                    at = i;
                    break;
                }
            if (at == slots_.size()) break;
            const std::string name = path.filename().string();
            if (std::find(continued.begin(), continued.end(), name) != continued.end()) continue;   // a chain's part
            const std::optional<SessionFile::Head> h = SessionFile::read_head(path);
            if (!h || h->seq.empty()) continue;
            if (h->sb != e_.state_bytes() || h->rb != e_.row_bytes()) {   // not this engine's any more
                std::error_code back;
                std::filesystem::remove(path, back);
                continue;
            }
            Slot& s = slots_[at];
            s.seq = h->seq;
            s.prompt = h->prompt;
            s.disk = name;
            s.disk_marks = h->mark_pos;
            s.disk_bytes = h->bytes;
            s.stamp = (uint64_t) when.time_since_epoch().count();
        }
    }

    // A conversation whose rows are in its session file reads them back.  The checkpoints' states come into the
    // host now (the host needs them to know where the conversation can be come back to); the rows stay in the
    // files until the conversation takes the engine, which reads them one record at a time: the host never holds
    // the whole conversation.  A file may hold only what came after its parent's rows, so the chain of them is
    // the conversation.  False: a file of the chain is gone or not this engine's any more (the model or its row
    // size changed; the files go, and the caller forgets the conversation), or the host could not take the states.
    bool materialize(size_t i) {
        audit::Scope audit_phase("restore");
        audit::HostSpan audit_span(__FILE__, __LINE__, "session-restore-total");
        Slot& s = slots_[i];
        std::vector<std::filesystem::path> chain;
        if (!SessionFile::chain_of(o_.session_dir / s.disk, chain)) {
            std::error_code ec;
            std::filesystem::remove(o_.session_dir / s.disk, ec);
            return false;
        }
        std::reverse(chain.begin(), chain.end());   // the root first: the chain is read in that order
        // Each file's header says where its rows begin: the parent's part ends there.  A file that continues
        // its parent (the ordinary delta) begins where the parent's rows ended; a file that diverges from it
        // (a branch) begins at the point of the divergence, and the parent's rows past it are not its own.
        std::vector<typename Slot::File> files;
        uint64_t prev_end = 0;
        bool prev_end_known = false;
        for (size_t k = 0; k < chain.size(); ++k) {
            SessionFile::Head0 h;
            FILE* f = nullptr;
#if defined(_WIN32)
            _wfopen_s(&f, chain[k].c_str(), L"rb");
#else
            f = std::fopen(chain[k].c_str(), "rb");
#endif
            if (!f) return false;
            uint64_t room = 0;
            const bool ok = SessionFile::head_ok(chain[k], f, h, room);
            std::fclose(f);
            if (!ok) return false;
            typename Slot::File entry;
            entry.name = chain[k].filename().string();
            entry.version = h.version;
            if (k == 0) {
                entry.rows_from = 0;
            } else if (h.version == SessionFile::kVersionBranch) {
                entry.rows_from = h.branch_at;   // this file says where its rows begin
            } else {   // a version 4 file: it continues where the previous one's rows ended
                if (!prev_end_known) {
                    const std::optional<SessionFile::Head> full = SessionFile::read_head(chain[k - 1]);
                    prev_end = full ? (uint64_t) full->rows_last : 0;
                    prev_end_known = true;
                }
                entry.rows_from = prev_end;
            }
            files.push_back(entry);
        }
        std::vector<Checkpoint> marks;
        bool no_ram = false;
        bool ok = true;
        s.chain.clear();
        SessionFile::Head head;
        for (size_t k = 0; k < files.size() && ok; ++k) {
            const uint64_t bound = k + 1 < files.size() ? files[k + 1].rows_from : (uint64_t) -1;
            ok = SessionFile::read_packed(
                o_.session_dir / files[k].name, head,
                [&](const SessionFile::Record& r, std::vector<uint8_t>&& packed, uint32_t version) {
                    if (r.kind != SessionFile::Record::kMark) return true;   // the rows come with the engine
                    if (r.pos > bound) return true;   // a checkpoint past the divergence is the parent's own
                    const BlobPtr b = blob(r.raw);
                    if (!b) {
                        no_ram = true;
                        return false;
                    }
                    if (!SessionFile::unpack(version, packed.data(), packed.size(), b->data(), r.raw))
                        return false;
                    marks.push_back({r.pos, b});
                    return true;
                },
                true);
            if (!ok || no_ram) break;
            files[k].rows_at = head.rows_at;
            files[k].n_rows = head.n_rows;
        }
        if (!ok || no_ram || head.seq.empty() || head.sb != e_.state_bytes() || head.rb != e_.row_bytes()) {
            std::error_code ec;
            std::filesystem::remove(o_.session_dir / s.disk, ec);
            return false;
        }
        s.chain = std::move(files);
        s.marks = std::move(marks);
        s.materialized = true;   // the rows are still in the files: enter() reads them
        // where the rows in the files reach: the deepest checkpoint of the chain, which is where the file
        // that was written last ended.  A park after this adds to the chain.
        s.disk_rows_end = s.marks.empty() ? 0 : (uint64_t) s.marks.back().pos;
        return true;
    }

    // The rows of a parked conversation, from its files, one record at a time: the host holds one record, not
    // the whole conversation.  The files are not walked again for the checkpoints: where their rows begin is known.
    bool load_rows_from_file(size_t i, int64_t need) {
        audit::Scope audit_phase("restore");
        audit::HostSpan audit_span(__FILE__, __LINE__, "session-restore-total");
        Slot& s = slots_[i];
        for (size_t k = 0; k < s.chain.size(); ++k) {
            const typename Slot::File& f = s.chain[k];
            // the parent's part ends where this file's rows begin: the rows past that are the parent's own,
            // and a branch does not have them
            const uint64_t bound = k + 1 < s.chain.size() ? s.chain[k + 1].rows_from : (uint64_t) -1;
            const bool ok = SessionFile::read_packed_rows(
                o_.session_dir / f.name, f.rows_at, f.n_rows, f.version,
                [&](const SessionFile::Record& r, std::vector<uint8_t>&& packed, uint32_t version) {
                    const int64_t a = std::max<int64_t>(r.first, (int64_t) f.rows_from);
                    const int64_t b = std::min<int64_t>(r.first + r.keep, (int64_t) std::min<uint64_t>(bound, (uint64_t) need));
                    if (a >= b) return true;   // the rows past the place the engine comes back to are not wanted
                    std::vector<uint8_t> tmp((size_t) r.raw);
                    if (!SessionFile::unpack(version, packed.data(), packed.size(), tmp.data(), r.raw)) return false;
                    e_.load_rows(tmp.data(), r.first, r.count, a, b - a);
                    return true;
                });
            if (!ok) return false;
        }
        return true;
    }

    // The session files take session_max_mib at most; the least recently used goes, the current one and `keep`
    // (the slot about to take the engine) stay.
    void disk_fit(size_t keep) {
        if (o_.session_max_mib < 1) return;
        for (;;) {
            size_t out = slots_.size(), total = 0;
            for (size_t i = 0; i < slots_.size(); ++i) {
                if (slots_[i].disk.empty()) continue;
                total += slots_[i].disk_bytes;
                if (i != cur_ && i != keep && (out == slots_.size() || slots_[i].stamp < slots_[out].stamp)) out = i;
            }
            if (out == slots_.size() || total <= ((size_t) o_.session_max_mib << 20)) break;
            unfile(slots_[out]);
            slots_[out] = Slot{};
        }
    }

    // Forgets a slot's session files: the file and the ones it adds to, unless another file names one of them
    // as its parent.
    void unfile(Slot& s) {
        if (s.disk.empty()) return;
        std::error_code ec;
        std::vector<std::filesystem::path> chain;
        if (SessionFile::chain_of(o_.session_dir / s.disk, chain)) {
            const std::vector<std::string> parents = parent_names();
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                const std::string name = it->filename().string();
                if (std::find(parents.begin(), parents.end(), name) != parents.end()) continue;   // another's parent
                std::filesystem::remove(*it, ec);
            }
        } else {
            std::filesystem::remove(o_.session_dir / s.disk, ec);
        }
        s.disk.clear();
        s.disk_marks.clear();
        s.disk_bytes = 0;
        s.disk_rows_end = 0;
        s.chain.clear();
    }

    // `with` is the current slot when its conversation is the one just parked.
    void say(const char* what, int64_t tokens, Clock::time_point t0, size_t with) const {
        if (!o_.log) return;
        const Held h = held(with);
        std::fprintf(o_.log,
                     "strata-poly: conversation cache: %s %lld tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu\n",
                     what, (long long) tokens, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     h.count, h.bytes, evictions_);
    }

    E& e_;
    CacheOptions o_;
    std::vector<Slot> slots_;   // the current conversation and the parked ones
    size_t cur_ = 0;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    uint64_t park_serial_ = 0;   // the files' names carry it, so a delta's parent keeps its own name
    std::vector<Checkpoint> due_;   // inside()'s checkpoints, until read() makes them the conversation's
    std::vector<int64_t> after_;    // where the conversation's checkpoints are once that read is done
    std::vector<int64_t> anchors_;  // semantic assistant boundaries in the prompt being read
    int blobs_ = 0;                 // the blobs ever asked for, for a test's fail_blob_at lever
};

}  // namespace strata::poly
