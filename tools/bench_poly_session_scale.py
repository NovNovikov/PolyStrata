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
        self.ram = []      # the system's free RAM and the engine's resident size, every half second
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

        def sample_ram():
            import psutil
            while self.p.poll() is None:
                try:
                    avail = psutil.virtual_memory().available
                    rss = psutil.Process(self.p.pid).memory_info().rss
                    self.ram.append((avail, rss))
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=sample_ram, daemon=True).start()

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


def chain_of(path):
    """The file's header: the version, the parent's name and the rows' end."""
    with open(path, 'rb') as f:
        head = f.read(120)
    if len(head) < 56:
        return None
    magic, version = int.from_bytes(head[0:4], 'little'), int.from_bytes(head[4:8], 'little')
    if magic != 0x53505353:
        return None
    out = dict(version=version, n_marks=int.from_bytes(head[40:48], 'little'),
               n_rows=int.from_bytes(head[48:56], 'little'), parent='')
    if version == 4 and len(head) >= 120:
        out['parent'] = head[56:120].split(b'\x00')[0].decode('utf8', 'replace')
    return out


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
    more = tok.encode('Grown synthetic tail. Cedar copper dune fern gravel.\n' * 400)[:1000]
    other = list(ids)
    other[0] = next(t for t in ids if t != ids[0])
    del tok, text
    gc.collect()
    assert len(ids) == a.tokens and len(other) == a.tokens and len(more) > 0
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
        # The conversation grows and is parked again: the new file is a delta that names the first.
        grown = list(ids) + more
        e.generate('A-grown', grown, 64)
        e.generate('B-branch-again', other, 64)
        summary['requests'] = [first, second, back]
        summary['chain'] = [dict(name=p.name, bytes=p.stat().st_size, **(chain_of(p) or {}))
                            for p in sorted(sessions.iterdir())]
    finally:
        e.close()
    text_err = (a.output / 'engine' / 'engine.stderr').read_text(encoding='utf8', errors='replace')
    (a.output / 'engine.stderr').write_text(text_err, encoding='utf8')
    summary['log'] = [l for l in text_err.splitlines() if 'parked' in l or 'restored' in l or 'session file' in l]
    if e.ram:
        avail = [r[0] for r in e.ram]
        rss = [r[1] for r in e.ram]
        summary['ram'] = dict(samples=len(e.ram), min_available_mib=min(avail) / 1048576,
                              max_rss_mib=max(rss) / 1048576)
        print('RAM: the system had %.0f MiB free at its tightest, the engine RSS peaked at %.0f MiB'
              % (summary['ram']['min_available_mib'], summary['ram']['max_rss_mib']), flush=True)
    for line in summary['log']:
        print(line, flush=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('SESSION_SCALE_DONE', flush=True)


if __name__ == '__main__':
    main()
