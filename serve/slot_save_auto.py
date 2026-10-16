"""serve/slot_save_auto.py - automatic disk sessions (stage 1 of the disk-checkpoint port).

Manual POST /slots/0?action=save|restore (Service.slot_action) covers named files.  This
adds the automatic half on top of the same SAVE/RESTORE engine lines, mirroring
NovNovikov/llama.cpp's --slot-save-auto design:

- after a clean solo request completes, SAVE the whole session under auto/<sha>-<n>.bin;
- before the next solo request's GEN, RESTORE the longest stored prefix of its ids;
- bounded store: max count / max bytes, LRU by mtime, auto files only (a manual save is
  never touched);
- floor: prefixes below the minimum are not worth a multi-GB write (about 1.2 GB per
  63K-token conversation);
- opt-in and off by default: with it off nothing here runs and the default path is
  byte-identical to the release.

Matching is by token prefix (sha256 of the ids) plus the request's cache keys (model,
sampling, turn).  A mismatch re-reads the prompt: always correct, at most slower.

Stage 1 saves whole sessions only (the tail).  Message-boundary slices (first answer,
last answer end, fifth-from-end) are stage 2 and need no engine changes either; only
incremental deltas would (engine C++).

Stage 2: the stored file still holds the whole session, but it is indexed under every
turn-start prefix inside it (first answer start, fifth-from-end start, ... - the same
turn markers turn_key() and keep_start() use, so this is template-agnostic).  A restore
loads the whole file and the engine reuses the matching prefix from it (its own
checkpoints cover the rest; a short tail past the last boundary is read again).  The
last-answer-end boundary is its next turn's start, off by that tail.  Hashes for all
boundaries come from one pass (hash copies), so indexing costs O(tokens), not
O(tokens x boundaries).  Metas without "prefixes" (stage 1) read as one pair.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

AUTO_SUBDIR = "auto"
SESSION_SUFFIX = ".bin"
META_SUFFIX = ".json"


def auto_dir(base: str | os.PathLike) -> Path:
    """The automatic store inside the slot save path (private to this user, like it)."""
    path = Path(base) / AUTO_SUBDIR
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def prefix_sha(ids, n: int) -> str:
    """sha256 of the first n prompt ids (deterministic across starts)."""
    h = hashlib.sha256()
    for t in ids[:n]:
        h.update(str(int(t)).encode("ascii"))
        h.update(b",")
    return h.hexdigest()


def turn_starts(ids, markers) -> list[int]:
    """Every turn-start position in the token stream (the turn_ids markers, as turn_key()
    and keep_start() read them; a marker echoed inside text reads as a start too)."""
    mark = set(markers or ())
    if not mark:
        return []
    return [i for i, t in enumerate(ids) if int(t) in mark and i > 0]


def prefix_hashes(ids, positions) -> dict[int, str]:
    """sha256 of every requested prefix length in one pass (identical to prefix_sha)."""
    want = sorted({int(p) for p in positions if 0 < int(p) <= len(ids)})
    out: dict[int, str] = {}
    h = hashlib.sha256()
    ids = list(ids)
    for i, t in enumerate(ids, 1):
        h.update(str(int(t)).encode("ascii"))
        h.update(b",")
        while want and want[0] == i:
            out[i] = h.copy().hexdigest()
            want.pop(0)
        if not want:
            break
    return out


def boundary_pairs(ids, markers, floor: int) -> list[tuple[int, str]]:
    """(length, sha) of the whole sequence plus every turn-start prefix (longest first)."""
    n = len(ids)
    positions = turn_starts(ids, markers) + [n]
    hashes = prefix_hashes(ids, positions)
    return sorted(((length, hashes[length]) for length in positions
                   if length >= floor and length in hashes), reverse=True)


def entry_name(sha: str, n: int) -> str:
    return f"{sha[:16]}-{int(n)}{SESSION_SUFFIX}"


def meta_name(session: str) -> str:
    return session + META_SUFFIX


def strict_auto_name(name: str) -> bool:
    """Only our own <16 hex>-<digits>.bin names (never a manual save, never a link target)."""
    stem, dot, ext = name.partition(".")
    if dot != "." or ext != SESSION_SUFFIX[1:]:
        return False
    sha, dash, num = stem.partition("-")
    return (dash == "-" and len(sha) == 16 and all(c in "0123456789abcdef" for c in sha)
            and num.isdigit() and int(num) > 0)


def write_meta(dir: Path, session: str, meta: dict) -> None:
    """The sidecar, written AFTER a successful SAVE (meta-last: a torn write never indexes)."""
    meta = {**meta, "v": 1, "at": int(time.time())}
    tmp = dir / (meta_name(session) + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, dir / meta_name(session))


def read_meta(dir: Path, session: str) -> dict | None:
    try:
        return json.loads((dir / meta_name(session)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _meta_length(meta: dict) -> int:
    """The full length for ordering (a corrupt value sorts first, never crashes)."""
    try:
        return int(meta.get("ntokens") or 0)
    except (TypeError, ValueError):
        return 0


def list_entries(dir: Path) -> list[tuple[str, dict]]:
    """(session, meta) of every complete auto pair, longest first."""
    out = []
    try:
        names = os.listdir(dir)
    except OSError:
        return out
    for name in names:
        if not strict_auto_name(name):
            continue
        meta = read_meta(dir, name)
        if not isinstance(meta, dict) or meta.get("v") != 1:
            continue
        try:
            if not (dir / name).is_file() or os.lstat(dir / name).st_size <= 0:
                continue
        except OSError:
            continue
        out.append((name, meta))
    out.sort(key=lambda e: (-_meta_length(e[1]), e[0]))
    return out


def entry_pairs(meta: dict) -> list[tuple[int, str]]:
    """Indexed (length, sha) pairs, longest first (stage-1 metas read as one pair)."""
    pairs = meta.get("prefixes")
    if isinstance(pairs, list) and pairs:
        out = []
        for item in pairs:
            try:
                length, sha = int(item[0]), str(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if length > 0 and len(sha) == 64:
                out.append((length, sha))
        return sorted(set(out), reverse=True)
    try:
        return [(int(meta.get("ntokens") or 0), str(meta.get("sha256") or ""))]
    except (TypeError, ValueError):
        return []


def find_best(dir: Path, ids, keys: str, model: str, floor: int,
              entries=None) -> tuple[str, dict] | None:
    """The longest stored prefix this prompt starts with (keys equal, prefix hash equal)."""
    n_all = len(ids)
    all_entries = entries if entries is not None else list_entries(dir)
    entries = [(session, meta) for session, meta in all_entries
               if meta.get("model") == model and meta.get("keys") == keys]
    if not entries:
        return None
    want: set[int] = set()
    for _, meta in entries:
        for length, _ in entry_pairs(meta):
            if floor <= length <= n_all:
                want.add(length)
    if not want:
        return None
    hashes = prefix_hashes(ids, sorted(want))
    best: tuple[str, dict] | None = None
    best_len = 0
    for session, meta in entries:
        for length, sha in entry_pairs(meta):
            if length > best_len and hashes.get(length) == sha:
                best, best_len = (session, meta), length
    return best


def covered(dir: Path, sha_full: str, entries=None) -> bool:
    """Whether this exact full prefix is already stored (dedup: skip the rewrite)."""
    for _, meta in entries if entries is not None else list_entries(dir):
        if meta.get("sha256") == sha_full:
            return True
    return False


def evict(dir: Path, max_count: int, max_bytes: int, entries=None) -> None:
    """LRU by mtime over auto pairs only (manual saves are never touched).

    Always sweeps orphan .bin files with no valid meta first: a SAVE whose meta never
    landed (crash, refused write, full disk) is unusable and would otherwise leak.  This
    is safe because our own SAVE-then-meta flow is synchronous under the request FIFO:
    any meta-less file seen here belongs to a finished flow, never to one in flight."""
    try:
        names = os.listdir(dir)
    except OSError:
        return
    for name in names:
        if not strict_auto_name(name):
            continue
        meta = read_meta(dir, name)
        if isinstance(meta, dict) and meta.get("v") == 1:
            continue
        try:
            os.remove(dir / name)
        except OSError:
            pass
    if max_count <= 0 and max_bytes <= 0:
        return
    pairs = entries if entries is not None else list_entries(dir)
    items = []
    for session, meta in pairs:
        try:
            st = os.lstat(dir / session)
            mst = os.lstat(dir / meta_name(session))
        except OSError:
            continue
        items.append((min(st.st_mtime, mst.st_mtime), st.st_size + mst.st_size, session))
    items.sort()
    total = sum(size for _, size, _ in items)
    while items and ((max_count > 0 and len(items) > max_count)
                     or (max_bytes > 0 and total > max_bytes)):
        _, size, session = items.pop(0)
        total -= size
        for name in (session, meta_name(session)):
            try:
                os.remove(dir / name)
            except OSError:
                pass


def remove_pair(dir: Path, session: str) -> None:
    """Drop a rejected candidate (bad restore): both files, never anything else."""
    if not strict_auto_name(session):
        return
    for name in (session, meta_name(session)):
        try:
            os.remove(dir / name)
        except OSError:
            pass


class StoreIndex:
    """In-memory index of the auto store: one readdir per change, not per request.

    A single server owns its store (like the fork's server-owned auto-cache dir), so
    the (name, size, mtime) signature of the auto files detects every change.
    Callers pass entries() wherever list_entries() went; a signature mismatch (an
    outside writer, a crash mid-flow) falls back to a fresh read, never to stale data."""

    def __init__(self) -> None:
        self.sig = None
        self.cached: list[tuple[str, dict]] = []

    def signature(self, dir: Path):
        try:
            names = os.listdir(dir)
        except OSError:
            return None
        sig = []
        for name in names:
            if strict_auto_name(name) or (name.endswith(META_SUFFIX)
                                          and strict_auto_name(name[:-len(META_SUFFIX)])):
                try:
                    st = os.lstat(dir / name)
                except OSError:
                    continue
                sig.append((name, st.st_size, st.st_mtime_ns))
        return tuple(sorted(sig))

    def entries(self, dir: Path) -> list[tuple[str, dict]]:
        sig = self.signature(dir)
        if sig is None:
            self.sig, self.cached = None, []
            return []
        if sig != self.sig:
            self.sig, self.cached = sig, list_entries(dir)
        return self.cached

    def invalidate(self) -> None:
        self.sig, self.cached = None, []
