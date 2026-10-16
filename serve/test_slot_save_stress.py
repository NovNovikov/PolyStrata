"""serve/test_slot_save_stress.py - auto sessions under concurrent traffic (no GPU, no model).

What single-threaded tests cannot reach: the FIFO discipline of the hooks, the idle
flush racing requests, eviction churn and a garbage store under load.  The mock engine
answers instantly, so this stays fast.

    python -m unittest serve.test_slot_save_stress -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve import slot_save_auto as auto  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SessionMockEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved: list[str] = []
        self.restored: list[str] = []
        self.lock = threading.Lock()

    def turn_key(self, ids):
        return ""

    def session_file(self, action, path):
        if action == "save":
            Path(path).write_bytes(b"x" * 64)
            with self.lock:
                self.saved.append(Path(path).name)
            return {"tokens": 100, "bytes": 64, "ms": 1.0}
        Path(path).read_bytes()
        with self.lock:
            self.restored.append(Path(path).name)
        return {"tokens": 100, "bytes": 64, "ms": 1.0}


class Stress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = SessionMockEngine(tok, "Hello!", max_context=32768)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.svc.slot_save_auto = True
        cls.svc.slot_save_min_tokens = 5
        cls.svc.slot_save_max_count = 4
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def chat(self, text):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps({"model": "x", "max_tokens": 16,
                             "messages": [{"role": "user", "content": text}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())

    def wave(self, texts):
        barrier = threading.Barrier(len(texts) + 1)
        out = [None] * len(texts)

        def one(i, text):
            barrier.wait(timeout=30)
            try:
                out[i] = self.chat(text)
            except Exception as e:  # noqa: BLE001 - a failed request is the finding
                out[i] = e

        threads = [threading.Thread(target=one, args=(i, text)) for i, text in enumerate(texts)]
        for t in threads:
            t.start()
        barrier.wait(timeout=30)
        for t in threads:
            t.join(timeout=180)
        return out

    @staticmethod
    def text_of(body):
        msg = body["choices"][0]["message"]
        return str(msg.get("content")) + str(msg.get("reasoning_content"))

    def setUp(self):
        self.engine.saved.clear()
        self.engine.restored.clear()
        store = auto.auto_dir(self.dir.name)
        for p in store.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        self.svc._auto_index.invalidate()

    def test_concurrent_shared_prefix_all_200_and_saved_once(self):
        res = self.wave(["shared prompt, same for everyone"] * 8)
        for r in res:
            self.assertEqual(r[0], 200, r[1] if not isinstance(r, tuple) else r)
            self.assertIn("Hello!", self.text_of(r[1]))
        self.assertEqual(len(self.engine.saved), 1)
        self.assertEqual(len(self.engine.restored), 7)
        res = self.wave(["shared prompt, same for everyone"] * 8)
        for r in res:
            self.assertEqual(r[0], 200)
        self.assertEqual(len(self.engine.restored), 15)
        self.assertEqual(len(self.engine.saved), 1)

    def test_concurrent_distinct_respects_caps_and_stays_valid(self):
        res = self.wave([f"distinct prompt number {i} with enough words" for i in range(12)])
        for r in res:
            self.assertEqual(r[0], 200)
            self.assertIn("Hello!", self.text_of(r[1]))
        entries = auto.list_entries(auto.auto_dir(self.dir.name))
        self.assertLessEqual(len(entries), 4)
        for _, meta in entries:
            self.assertEqual(meta.get("v"), 1)
            self.assertTrue(meta.get("prefixes"))

    def test_garbage_store_under_load(self):
        store = auto.auto_dir(self.dir.name)
        (store / "abcdef0123456789-10.bin").write_bytes(b"orphan")
        (store / "manual.bin").write_bytes(b"hands off")
        (store / "abcdef0123456788-10.bin").write_bytes(b"o")
        (store / "abcdef0123456788-10.bin.json").write_text('{"v": 0}', encoding="utf-8")
        s, _ = self.chat("a fresh prompt after garbage, quite long indeed")
        self.assertEqual(s, 200)
        names = {p.name for p in store.iterdir()}
        self.assertIn("manual.bin", names)
        self.assertNotIn("abcdef0123456789-10.bin", names)
        self.assertNotIn("abcdef0123456788-10.bin", names)


if __name__ == "__main__":
    unittest.main()
