"""Manual GPU test: a long conversation parked to the disk and brought back, at the scale of a real chat.

The engine is started with the launch config (the session store and the expert profile go to fresh test
paths).  One long synthetic prompt makes a conversation of `--tokens` tokens; a different prompt then
branches off it, which parks the conversation to a file in the session store, and the first prompt again
brings it back from that file.  The engine's own log lines carry the times: `parked N tokens in X ms`
and `restored N tokens in X ms`.

Only synthetic prompts and fresh synthetic sessions are used; no user session is read or written.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer


class Engine:
    def __init__(self, config, args, home, exe=None):
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.lines = []
        env = os.environ.copy()
        env.pop('STRATA_FREEZE_EXPERTS', None)
        env['PATH'] = os.pathsep.join(config.get('lib_dirs', []) + [env.get('PATH', '')])
        self.err = (home / 'engine.stderr').open('w', encoding='utf8')
        self.command = [exe or config['exe'], '--serve'] + args
        self.p = subprocess.Popen(self.command, cwd=config['cwd'], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.q = queue.Queue()

        def read():
            for line in self.p.stdout:
                self.q.put(line.rstrip())
            self.q.put(None)

        threading.Thread(target=read, daemon=True).start()

    def line(self):
        line = self.q.get(timeout=3600)
        if line is None:
            raise RuntimeError(f'engine exited: {self.p.wait()}; see {self.home}')
        self.lines.append(line)
        if line.startswith(('ERR ', 'FATAL ', 'SERR ')):
            raise RuntimeError(line)
        return line

    def ready(self):
        start = time.monotonic()
        while not self.line().startswith('READY '):
            pass
        self.load_s = time.monotonic() - start
        print('READY', round(self.load_s, 2), flush=True)

    def send(self, command):
        self.p.stdin.write(command + '\n')
        self.p.stdin.flush()

    def generate(self, label, ids, count):
        self.send(f'GEN {count} temperature=0 seed=42 ' + ','.join(map(str, ids)))
        tokens = []
        last = time.monotonic()
        while True:
            line = self.line()
            if line.startswith('T '):
                tokens.append(int(line.split()[1]))
            if time.monotonic() - last > 15:
                print(label, f'generated {len(tokens)}', flush=True)
                last = time.monotonic()
            if line.startswith('DONE '):
                d = line.split()
                break
        assert len(tokens) == count, (label, len(tokens))
        print(label, 'done', 'prompt_ms', d[3], 'decode_ms', d[4], 'resume', d[8], flush=True)
        return dict(label=label, prompt_tokens=len(ids), generated=count, prompt_ms=float(d[3]),
                    decode_ms=float(d[4]), resume=int(d[8]))

    def close(self):
        try:
            if self.p.poll() is None:
                self.send('QUIT')
                self.p.wait(timeout=120)
        finally:
            if self.p.poll() is None:
                self.p.kill()
                self.p.wait()
            self.err.close()
        (self.home / 'protocol.txt').write_text('\n'.join(self.lines) + '\n', encoding='utf8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True, help='public synthetic-test profile, not user history')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--exe', type=Path, default=None, help='the engine to run; the config\'s own by default')
    ap.add_argument('--tokens', type=int, default=90000)
    a = ap.parse_args()
    a.work = a.work.resolve()
    a.output = a.output.resolve()
    a.work.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    sessions = a.work / 'sessions'
    sessions.mkdir(exist_ok=True)

    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    args = list(config['args'])
    args[args.index('--session-dir') + 1] = str(sessions)
    profile = a.work / 'expert-profile.txt'
    if not profile.exists():
        shutil.copyfile(a.profile, profile)
    args[args.index('--expert-profile') + 1] = str(profile)

    tok = Tokenizer.from_dir(config['tokenizer'])
    text = 'Synthetic long session scale test.\n' + ''.join(
        f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(40000))
    ids = tok.encode(text)[:a.tokens]
    other = list(ids)
    other[0] = next(t for t in ids if t != ids[0])
    del tok, text
    gc.collect()
    assert len(ids) == a.tokens and len(other) == a.tokens
    print('prompt tokens', len(ids), flush=True)

    e = Engine(config, args, a.output / 'engine', a.exe)
    summary = dict(exe=str(a.exe or config['exe']), tokens=a.tokens, work=str(a.work))
    try:
        e.ready()
        first = e.generate('A-long', ids, 64)
        second = e.generate('B-branch', other, 64)
        files = sorted(sessions.iterdir())
        assert files, 'the conversation was not parked to a file'
        summary['parked_files'] = [dict(name=p.name, bytes=p.stat().st_size) for p in files]
        back = e.generate('A-again', ids, 64)
        summary['requests'] = [first, second, back]
    finally:
        e.close()
    text_err = (a.output / 'engine' / 'engine.stderr').read_text(encoding='utf8', errors='replace')
    (a.output / 'engine.stderr').write_text(text_err, encoding='utf8')
    summary['log'] = [l for l in text_err.splitlines() if 'parked' in l or 'restored' in l or 'session file' in l]
    for line in summary['log']:
        print(line, flush=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('SESSION_SCALE_DONE', flush=True)


if __name__ == '__main__':
    main()
