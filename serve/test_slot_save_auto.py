"""serve/test_slot_save_auto.py - automatic disk sessions (no GPU, no model).

    python -m unittest serve.test_slot_save_auto -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402
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
        self.assertFalse(auto.covered(self.store, MODEL, KEYS, auto.prefix_sha(ids, 1500)))
        self.put(ids)
        self.assertTrue(auto.covered(self.store, MODEL, KEYS, auto.prefix_sha(ids, 1500)))

    def test_covered_is_model_and_keys_scoped(self):
        ids = make_ids(1500)
        self.put(ids)
        sha = auto.prefix_sha(ids, 1500)
        self.assertFalse(auto.covered(self.store, "other-model", KEYS, sha))
        self.assertFalse(auto.covered(self.store, MODEL, "other keys", sha))

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
        self.assertEqual(auto.prefix_hashes(ids, [0, -4, 99]), {})
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

    def test_corrupt_meta_never_breaks_lookup(self):
        import json as _json

        session = self.put(make_ids(1500))
        (self.store / auto.meta_name(session)).write_text(
            _json.dumps({"v": 1, "ntokens": "abc", "model": MODEL, "keys": KEYS}),
            encoding="utf-8")
        self.assertEqual(auto.list_entries(self.store), [(session, auto.read_meta(self.store, session))])
        self.assertIsNone(auto.find_best(self.store, make_ids(1600), KEYS, MODEL, 1024))

    def test_evict_sweeps_orphans_and_wrong_versions(self):
        orphan = self.store / "abcdef0123456789-10.bin"
        orphan.write_bytes(b"o" * 10)
        old = self.store / "abcdef0123456788-10.bin"
        old.write_bytes(b"o" * 10)
        (self.store / auto.meta_name(old.name)).write_text('{"v": 0}', encoding="utf-8")
        auto.evict(self.store, 0, 0)
        self.assertFalse(orphan.exists())
        self.assertFalse(old.exists())


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
            if self.engine.fail_dead:
                self.engine.lines.put(None)
            elif not Path(path).exists() or self.engine.fail:
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
        self.fail_dead = False
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

    def test_server_side_evicts_to_max_count(self):
        self.svc.slot_save_max_count = 1
        self.svc.slot_save_auto = True
        self.svc.auto_save_session(make_ids(2000), [1], self.sampling())
        self.svc.auto_save_session(make_ids(2000, 50000), [2], self.sampling())
        self.assertEqual(len(auto.list_entries(auto.auto_dir(self.dir.name))), 1)

    def test_dead_engine_keeps_the_pair(self):
        self.svc.slot_save_auto = True
        self.svc.auto_save_session(self.ids, [1, 2], self.sampling())
        (session, _), = auto.list_entries(auto.auto_dir(self.dir.name))
        self.engine.fail_dead = True
        n = self.svc.auto_restore_session(self.ids + [1, 2, 3], self.sampling())
        self.assertEqual(n, 0)
        self.assertEqual([s for s, _ in auto.list_entries(auto.auto_dir(self.dir.name))], [session])


class StoreIndexTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = auto.auto_dir(self.dir.name)
        self.index = auto.StoreIndex()

    def tearDown(self):
        self.dir.cleanup()

    def put(self, ids):
        n = len(ids)
        session = auto.entry_name(auto.prefix_sha(ids, n), n)
        (self.store / session).write_bytes(b"x" * 10)
        auto.write_meta(self.store, session, {"model": MODEL, "keys": KEYS, "ntokens": n,
                                              "sha256": auto.prefix_sha(ids, n), "bytes": 10})
        return session

    def test_cached_identity_until_change(self):
        self.put(make_ids(1500))
        first = self.index.entries(self.store)
        again = self.index.entries(self.store)
        self.assertEqual(first, again)
        self.assertIsNot(first, again)
        self.put(make_ids(1500, 9000))
        second = self.index.entries(self.store)
        self.assertEqual(len(second), 2)

    def test_external_removal_refreshes(self):
        session = self.put(make_ids(1500))
        self.assertEqual(len(self.index.entries(self.store)), 1)
        (self.store / session).unlink()
        (self.store / auto.meta_name(session)).unlink()
        self.assertEqual(self.index.entries(self.store), [])


class IdleShutdown(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.svc.slot_save_auto = True
        self.svc.slot_save_min_tokens = 5
        self.ids = make_ids(100)

    def tearDown(self):
        self.dir.cleanup()

    def retain(self):
        self.svc._auto_last = (list(self.ids), [1, 2], self.sampling())

    def sampling(self):
        return {"temperature": 0.7}

    def auto_files(self):
        return [p for p in auto.auto_dir(self.dir.name).iterdir() if p.suffix == ".bin"]

    def test_idle_flush_after_timeout(self):
        self.retain()
        self.svc.last_request_at = time.time() - 3600
        self.svc.slot_save_idle_seconds = 60
        self.svc.auto_idle_save()
        self.assertEqual(len(self.auto_files()), 1)

    def test_idle_skips_recent_activity_and_repeats(self):
        self.retain()
        self.svc.last_request_at = time.time()
        self.svc.slot_save_idle_seconds = 60
        self.svc.auto_idle_save()
        self.assertEqual(self.auto_files(), [])
        self.svc.last_request_at = time.time() - 3600
        self.svc.auto_idle_save()
        self.assertEqual(len(self.auto_files()), 1)
        self.svc.auto_idle_save()
        self.assertEqual(len(self.auto_files()), 1)

    def test_idle_disabled(self):
        self.retain()
        self.svc.last_request_at = time.time() - 3600
        self.svc.slot_save_idle_seconds = -1
        self.svc.auto_idle_save()
        self.assertEqual(self.auto_files(), [])

    def test_idle_busy_keeps_armed(self):
        self.retain()
        self.svc.last_request_at = time.time() - 3600
        self.svc.slot_save_idle_seconds = 60
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        try:
            self.svc.auto_idle_save()
        finally:
            self.svc.fifo.release()
        self.assertEqual(self.auto_files(), [])
        self.assertIsNotNone(self.svc._auto_last)
        self.svc.auto_idle_save()
        self.assertEqual(len(self.auto_files()), 1)
        self.assertIsNone(self.svc._auto_last)

    def test_shutdown_saves_once(self):
        self.retain()
        self.svc.auto_save_at_shutdown()
        self.assertEqual(len(self.auto_files()), 1)
        self.svc.auto_save_at_shutdown()
        self.assertEqual(len(self.auto_files()), 1)
        self.assertIsNone(self.svc._auto_last)

    def test_start_auto_idle_only_when_wanted(self):
        started = []
        real = threading.Thread

        class Probe(real):
            def start(self):
                started.append(self.name)

        threading.Thread = Probe
        try:
            self.svc.start_auto_idle()
            self.svc.slot_save_auto = False
            self.svc.start_auto_idle()
        finally:
            threading.Thread = real
        self.assertEqual(started, ["auto-slot-idle"])


class SessionMockEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved: list[str] = []
        self.restored: list[str] = []

    def turn_key(self, ids):
        return ""

    def session_file(self, action, path):
        if action == "save":
            Path(path).write_bytes(b"x" * 64)
            self.saved.append(Path(path).name)
            return {"tokens": 100, "bytes": 64, "ms": 1.0}
        Path(path).read_bytes()
        self.restored.append(Path(path).name)
        return {"tokens": 100, "bytes": 64, "ms": 1.0}


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = SessionMockEngine(tok, "Hello!", max_context=32768)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.svc.slot_save_auto = True
        cls.svc.slot_save_min_tokens = 5
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def chat(self, messages):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps({"model": "x", "max_tokens": 16, "messages": messages}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())

    def test_repeat_reuses_and_followup_extends(self):
        first = [{"role": "user", "content": "hello world, this is a test"}]
        s, b = self.chat(first)
        self.assertEqual(s, 200, b)
        self.assertEqual(len(self.engine.saved), 1)
        s, b = self.chat(first)
        self.assertEqual(s, 200, b)
        self.assertEqual(len(self.engine.restored), 1)
        self.assertEqual(len(self.engine.saved), 1)
        follow = first + [{"role": "assistant", "content": b["choices"][0]["message"]["content"]},
                          {"role": "user", "content": "and again, please"}]
        s, b = self.chat(follow)
        self.assertEqual(s, 200, b)
        self.assertEqual(len(self.engine.saved), 2)
        self.assertEqual(len(self.engine.restored), 2)


if __name__ == "__main__":
    unittest.main()
