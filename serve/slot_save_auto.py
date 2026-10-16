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
    out.sort(key=lambda e: (-int(e[1].get("ntokens") or 0), e[0]))
    return out


def find_best(dir: Path, ids, keys: str, model: str, floor: int) -> tuple[str, dict] | None:
    """The longest stored entry this prompt starts with (keys equal, prefix hash equal)."""
    n_all = len(ids)
    for session, meta in list_entries(dir):
        if meta.get("model") != model or meta.get("keys") != keys:
            continue
        try:
            n = int(meta.get("ntokens") or 0)
        except (TypeError, ValueError):
            continue
        if n < floor or n > n_all:
            continue
        if prefix_sha(ids, n) == meta.get("sha256"):
            return session, meta
    return None


def covered(dir: Path, sha_full: str) -> bool:
    """Whether this exact full prefix is already stored (dedup: skip the rewrite)."""
    for session, meta in list_entries(dir):
        if meta.get("sha256") == sha_full:
            return True
    return False


def evict(dir: Path, max_count: int, max_bytes: int) -> None:
    """LRU by mtime over auto pairs only (manual saves are never touched)."""
    if max_count <= 0 and max_bytes <= 0:
        return
    entries = []
    for session, meta in list_entries(dir):
        try:
            st = os.lstat(dir / session)
            mst = os.lstat(dir / meta_name(session))
        except OSError:
            continue
        entries.append((min(st.st_mtime, mst.st_mtime), st.st_size + mst.st_size, session))
    entries.sort()
    total = sum(size for _, size, _ in entries)
    while entries and ((max_count > 0 and len(entries) > max_count)
                       or (max_bytes > 0 and total > max_bytes)):
        _, size, session = entries.pop(0)
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
