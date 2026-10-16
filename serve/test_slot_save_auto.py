"""serve/test_slot_save_auto.py - automatic disk sessions (no GPU, no model).

    python -m unittest serve.test_slot_save_auto -v
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine  # noqa: E402
from serve import slot_save_auto as auto  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = "model\n temperature=0.7"
MODEL = "qwen3.8-flash-next"


def make_ids(n, start=1):
    return list(range(start, start + n))


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = auto.auto_dir(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def put(self, ids, keys=KEYS, model=MODEL):
        n = len(ids)
        session = auto.entry_name(auto.prefix_sha(ids, n), n)
        (self.store / session).write_bytes(b"x" * 100)
        auto.write_meta(self.store, session, {"model": model, "keys": keys, "ntokens": n,
                                              "sha256": auto.prefix_sha(ids, n), "bytes": 100})
        return session

    def test_strict_names_keep_manual_saves_away(self):
        self.assertTrue(auto.strict_auto_name("abcdef0123456789-123.bin"))
        for bad in ("a.bin", "../x.bin", "abcdef0123456789-123.json", "ABCDEF0123456789-1.bin",
                    "abcdef0123456789-0.bin", "abcdef0123456789-12.bin.foo", ".bin"):
            self.assertFalse(auto.strict_auto_name(bad), bad)

    def test_meta_roundtrip_and_torn_meta_ignored(self):
        session = self.put(make_ids(10))
        self.assertEqual(auto.read_meta(self.store, session)["ntokens"], 10)
        (self.store / auto.meta_name(session)).write_text("{broken", encoding="utf-8")
        self.assertEqual(auto.list_entries(self.store), [])

    def test_find_best_takes_longest_matching_prefix(self):
        short, long = make_ids(1500), make_ids(2500)
        self.put(short)
        self.put(long)
        ids = make_ids(3000)
        found = auto.find_best(self.store, ids, KEYS, MODEL, 1024)
        self.assertIsNotNone(found)
        self.assertEqual(found[1]["ntokens"], 2500)

    def test_find_best_rejects_keys_model_floor_and_divergence(self):
        ids = make_ids(2000)
        self.put(ids)
        other = make_ids(2000, start=5000)
        self.assertIsNone(auto.find_best(self.store, other, KEYS, MODEL, 1024))
        self.assertIsNone(auto.find_best(self.store, ids, "other keys", MODEL, 1024))
        self.assertIsNone(auto.find_best(self.store, ids, KEYS, "other-model", 1024))
        self.assertIsNone(auto.find_best(self.store, ids, KEYS, MODEL, 5000))

    def test_covered_dedups_exact_prefix(self):
        ids = make_ids(1500)
        self.assertFalse(auto.covered(self.store, auto.prefix_sha(ids, 1500)))
        self.put(ids)
        self.assertTrue(auto.covered(self.store, auto.prefix_sha(ids, 1500)))

    def test_evict_caps_count_and_bytes_but_never_manual(self):
        manual = Path(self.dir.name) / "manual.bin"
        manual.write_bytes(b"m" * 10000)
        for start in (1, 100000, 200000):
            self.put(make_ids(1500, start))
            os.utime(self.store / auto.entry_name(auto.prefix_sha(make_ids(1500, start), 1500), 1500),
                     (start, start))
        auto.evict(self.store, 2, 0)
        self.assertEqual(len(auto.list_entries(self.store)), 2)
        self.assertTrue(manual.exists())
        auto.evict(self.store, 0, 700)
        self.assertEqual(len(auto.list_entries(self.store)), 2)
        self.assertTrue(manual.exists())

    def test_remove_pair_refuses_foreign_names(self):
        evil = Path(self.dir.name) / "manual.bin"
        evil.write_bytes(b"m")
        auto.remove_pair(self.store, "../manual.bin")
        auto.remove_pair(self.store, "manual.bin")
        self.assertTrue(evil.exists())

    def test_turn_starts_and_rolling_hashes(self):
        ids = [5, 7, 9, 7, 11, 7, 13]
        self.assertEqual(auto.turn_starts(ids, {7}), [1, 3, 5])
        self.assertEqual(auto.turn_starts(ids, set()), [])
        hashes = auto.prefix_hashes(ids, [1, 3, 5, 7])
        for pos in (1, 3, 5, 7):
            self.assertEqual(hashes[pos], auto.prefix_sha(ids, pos))
        pairs = auto.boundary_pairs(ids, {7}, 1)
        self.assertEqual([length for length, _ in pairs], [7, 5, 3, 1])
        self.assertEqual(auto.boundary_pairs(ids, {7}, 6), [(7, auto.prefix_sha(ids, 7))])

    def test_find_best_matches_a_turn_boundary_then_diverges(self):
        base = [5, 7, 9, 7, 11]
        full = base + [13, 15]
        session = auto.entry_name(auto.prefix_sha(full, len(full)), len(full))
        pairs = auto.boundary_pairs(full, {7}, 1)
        (self.store / session).write_bytes(b"x" * 100)
        auto.write_meta(self.store, session, {"model": MODEL, "keys": KEYS, "ntokens": len(full),
                                              "sha256": pairs[0][1],
                                              "prefixes": [[length, sha] for length, sha in pairs],
                                              "bytes": 100})
        # same turn boundary (position 3), different tail: the boundary matches
        again = base[:4] + [99, 100, 101]
        found = auto.find_best(self.store, again, KEYS, MODEL, 1)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], session)

    def test_stage1_meta_without_prefixes_reads_as_one_pair(self):
        # put() writes no "prefixes" (the stage-1 shape): it must match as one pair
        ids = make_ids(1500)
        session = self.put(ids)
        self.assertNotIn("prefixes", auto.read_meta(self.store, session))
        found = auto.find_best(self.store, make_ids(1600), KEYS, MODEL, 1024)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], session)


class FakeProc:
    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []

    def write(self, s):
        self.sent.append(s)
        cmd, _, path = s.strip().partition(" ")
        if cmd == "SAVE":
            if self.engine.fail:
                self.engine.lines.put("SERR invalid 0 no complete session to save\n")
            else:
                Path(path).write_bytes(b"x" * 1234)
                self.engine.lines.put("SAVED 62993 1234 401.5\n")
        elif cmd == "RESTORE":
            if not Path(path).exists() or self.engine.fail:
                self.engine.lines.put("SERR invalid 0 session file: saved with another model\n")
            else:
                self.engine.lines.put("RESTORED 62993 1234 560.2\n")

    def flush(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class FakeEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.fail = False
        self.silence_s = 300.0
        self.proc = FakeProc(self)

    def turn_key(self, ids):
        return ""


class Hooks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.ids = make_ids(2000)

    def tearDown(self):
        self.dir.cleanup()

    def sampling(self):
        return {"temperature": 0.7}

    def test_off_by_default_sends_no_save_lines(self):
        self.assertFalse(self.svc.slot_save_auto)
        self.svc.auto_save_session(self.ids, [1, 2], self.sampling())
        self.assertEqual(self.engine.proc.sent, [])

    def test_save_then_restore_roundtrip(self):
        self.svc.slot_save_auto = True
        gen = make_ids(50, 9000)
        self.svc.auto_save_session(self.ids, gen, self.sampling())
        self.assertTrue(any(s.startswith("SAVE ") for s in self.engine.proc.sent))
        again = self.ids + gen + make_ids(30, 9500)
        n = self.svc.auto_restore_session(again, self.sampling())
        self.assertEqual(n, 2050)
        self.assertTrue(any(s.startswith("RESTORE ") for s in self.engine.proc.sent))

    def test_save_below_floor_skipped(self):
        self.svc.slot_save_auto = True
        self.svc.auto_save_session(make_ids(100), [1], self.sampling())
        self.assertEqual(self.engine.proc.sent, [])

    def test_engine_refusal_never_fails_and_drops_pair(self):
        self.svc.slot_save_auto = True
        self.svc.auto_save_session(self.ids, [1, 2], self.sampling())
        self.engine.fail = True
        n = self.svc.auto_restore_session(self.ids + [1, 2, 3], self.sampling())
        self.assertEqual(n, 0)
        self.assertEqual(auto.list_entries(auto.auto_dir(self.dir.name)), [])

    def test_batch_engine_disables_hooks(self):
        self.svc.slot_save_auto = True
        self.engine.batch = 2
        self.svc.auto_save_session(self.ids, [1, 2], self.sampling())
        self.assertEqual(self.engine.proc.sent, [])


if __name__ == "__main__":
    unittest.main()
