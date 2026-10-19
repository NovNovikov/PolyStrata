"""Manual GPU integration test: synthetic checkpoints, disk sessions and the full adaptive expert cache.

Uses a launch config, with private logs and session stores replaced by fresh test paths. Never reads a
user session. Large session payloads stay under --work; only text measurements go under --output.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

from strata_tokenizer import Tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.telemetry import gpu_process_memory


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


class Engine:
    def __init__(self, config, args, home):
        self.home = home
        home.mkdir()
        self.lines = []
        self.results = []
        self.gpu_memory = []
        env = os.environ.copy()
        env.pop('STRATA_FREEZE_EXPERTS', None)
        env['STRATA_VERIFY_STAGE_CACHE'] = '1'
        env['PATH'] = os.pathsep.join(config.get('lib_dirs', []) + [env.get('PATH', '')])
        self.err = (home / 'engine.stderr').open('w', encoding='utf8')
        self.command = [config['exe'], '--serve'] + args
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
        line = self.q.get(timeout=900)
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
        print(self.home.name, 'READY', round(self.load_s, 2), flush=True)
        self.sample_memory('READY')

    def sample_memory(self, phase):
        sample = dict(phase=phase, **gpu_process_memory(self.p.pid))
        self.gpu_memory.append(sample)
        print('GPU_PROCESS_MEMORY', json.dumps(sample), flush=True)

    def send(self, command):
        self.p.stdin.write(command + '\n')
        self.p.stdin.flush()

    def generate(self, label, ids, anchors, count):
        self.send(f'GEN {count} temperature=0 seed=42 checkpoints=' + ','.join(map(str, anchors)) +
                  ' ' + ','.join(map(str, ids)))
        tokens, stamp, resume = [], [], None
        last = time.monotonic()
        while True:
            line = self.line()
            if line.startswith('RESUME '):
                resume = int(line.split()[1])
            if line.startswith('T '):
                tokens.append(int(line.split()[1]))
                stamp.append(time.monotonic())
            if time.monotonic() - last > 10:
                print(label, line if line.startswith('PP ') else f'generated {len(tokens)}', flush=True)
                last = time.monotonic()
            if line.startswith('DONE '):
                d = line.split()
                break
        assert len(tokens) == count and int(d[1]) == count, d
        assert int(d[8]) == resume, d
        result = dict(label=label, prompt_tokens=len(ids), generated_ids=tokens, resume=resume,
                      prompt_ms=float(d[3]), decode_ms=float(d[4]), guessed_right=int(d[6]),
                      guessed=int(d[7]), hits=int(d[9]), lookups=int(d[10]), read=int(d[14]),
                      decode_tps=(count - 1) * 1000 / float(d[4]),
                      token_times=[round(t - stamp[0], 6) for t in stamp])
        result['hit_rate'] = result['hits'] / max(1, result['lookups'])
        self.results.append(result)
        self.sample_memory(label)
        self.record()
        print(label, json.dumps({k:v for k,v in result.items() if k not in ('generated_ids','token_times')}), flush=True)
        return result

    def session(self, command, path, prefix):
        self.send(command + ' ' + str(path))
        while True:
            line = self.line()
            if line.startswith(prefix + ' '):
                print(command, line, flush=True)
                return line

    def record(self):
        (self.home / 'protocol.txt').write_text('\n'.join(self.lines) + '\n', encoding='utf8')
        (self.home / 'results.json').write_text(json.dumps(dict(command=self.command,
            load_s=getattr(self, 'load_s', None), requests=self.results,
            gpu_process_memory=self.gpu_memory), indent=2), encoding='utf8')

    def close(self):
        try:
            if self.p.poll() is None:
                self.send('QUIT')
                self.p.wait(timeout=60)
        finally:
            if self.p.poll() is None:
                self.p.kill()
                self.p.wait()
            self.err.close()
            self.record()
        assert self.p.returncode == 0, self.p.returncode


def finish(a, config, args, summary, ids, first, tail):
    e = Engine(config, args, a.output / 'restart')
    summary['stages'].append('restart')
    try:
        e.ready()
        resumed = e.generate('A-after-process-restart', ids, summary['anchors'], 64)
        assert resumed['resume'] >= 8192, resumed
        e.session('RESTORE', a.work / 'explicit.session', 'RESTORED')
        # The old continuation is also parked. A different suffix makes the explicitly restored state
        # the deepest matching checkpoint, so this tests device restore rather than a better parked match.
        suffix = list(tail[:64])
        suffix[0] = next(t for t in ids if t != suffix[0])
        probe = e.generate('A-isolated-explicit-restore', ids + first['generated_ids'] + suffix,
                           summary['anchors'] + [10511], 64)
        assert probe['resume'] == 10511, probe
    finally:
        e.close()
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    cache = {}
    for label in summary['stages']:
        text = (a.output / label / 'engine.stderr').read_text(encoding='utf8', errors='replace')
        slots = re.findall(r'staging_cache=(\d+)/(\d+) slots \(([\d.]+) MiB', text)
        verified = re.findall(r'staging cache verified: (\d+) GPU payloads match original RAM; zero tails OK', text)
        assert verified and slots, f'no cache proof in {label}'
        cache[label] = dict(max_used_slots=max(int(s[0]) for s in slots),
                            capacity=max(int(s[1]) for s in slots),
                            max_used_mib=max(float(s[2]) for s in slots), verified_counts=list(map(int,verified)))
    assert cache['initial']['capacity'] == 255 and cache['initial']['max_used_slots'] == 255
    assert cache['initial']['max_used_mib'] >= 2800
    assert max(cache['initial']['verified_counts']) == 255
    summary['cache'] = cache
    summary['automatic_sessions'] = [{'name':p.name,'bytes':p.stat().st_size}
                                     for p in (a.work/'sessions').iterdir()]
    summary['success'] = True
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('CHECKPOINT_CACHE_INTEGRATION_PASSED', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True, help='public synthetic-test profile, not user history')
    ap.add_argument('--work', type=Path, required=True, help='fresh directory for large synthetic session files')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', action='store_true', help='finish the restart phase of this synthetic test only')
    a = ap.parse_args()
    a.work = a.work.resolve()
    a.output = a.output.resolve()
    if a.resume:
        summary = json.loads((a.output/'summary.json').read_text())
        initial = json.loads((a.output/'initial/results.json').read_text())
        config = json.loads(a.config.read_text(encoding='utf-8-sig'))
        assert str(a.work) == summary['work'] and sha(a.config) == summary['config_sha256']
        assert sha(Path(config['exe'])) == summary['exe_sha256']
        assert summary['session_sha256'] == sha(a.work/'explicit.session') == summary['roundtrip_sha256']
        assert len(initial['requests']) == 5 and initial['requests'][3]['resume'] == 4096
        assert initial['requests'][4]['resume'] >= 10511
        assert initial['requests'][1]['generated_ids'] == initial['requests'][4]['generated_ids']
        summary['continued_from_initial_run'] = True
        tok = Tokenizer.from_dir(config['tokenizer'])
        text = 'Synthetic checkpoint and adaptive cache integration test.\n' + ''.join(
            f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(15000))
        ids = tok.encode(text)[:10000]
        tail = tok.encode('Changed synthetic suffix. Cedar copper dune fern.\n' * 2000)
        del tok, text
        gc.collect()
        finish(a, config, initial['command'][2:], summary, ids, initial['requests'][0], tail)
        return
    a.work.mkdir(parents=True, exist_ok=False)
    a.output.mkdir(parents=True, exist_ok=False)
    sessions = a.work / 'sessions'
    sessions.mkdir()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    args = list(config['args'])
    assert args[args.index('--prompt-cache')+1] == '6'
    assert args[args.index('--stage-mib')+1] == '2848'
    assert args[args.index('--expert-swaps')+1] == '64'
    profile = a.work / 'expert-profile.txt'
    shutil.copyfile(a.profile, profile)
    args[args.index('--session-dir')+1] = str(sessions)
    args[args.index('--expert-profile')+1] = str(profile)
    tok = Tokenizer.from_dir(config['tokenizer'])
    text = 'Synthetic checkpoint and adaptive cache integration test.\n' + ''.join(
        f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(15000))
    ids = tok.encode(text)[:10000]
    tail = tok.encode('Changed synthetic suffix. Cedar copper dune fern.\n' * 2000)
    assert len(ids) == 10000
    del tok, text
    gc.collect()
    anchors = [2048, 4096, 6144, 8192]
    summary = dict(exe_sha256=sha(Path(config['exe'])), config_sha256=sha(a.config),
                   work=str(a.work), anchors=anchors, stages=[])
    def engine(label):
        e = Engine(config, args, a.output / label)
        summary['stages'].append(label)
        return e
    e = engine('initial')
    try:
        e.ready()
        first = e.generate('A-full-cache', ids, anchors, 512)
        assert first['resume'] == 0
        saved = a.work / 'explicit.session'
        e.session('SAVE', saved, 'SAVED')
        continuation = ids + first['generated_ids'] + tail[:64]
        cont = e.generate('A-continuation', continuation, anchors + [10511], 64)
        assert cont['resume'] == 10511, cont
        other = list(ids)
        other[0] = next(t for t in ids if t != ids[0])
        second = e.generate('B-switch', other, anchors, 64)
        assert second['resume'] == 0
        assert list(sessions.iterdir()), 'automatic disk parking did not create a file'
        edited = ids[:5000] + tail[:2048]
        branch = e.generate('A-interior-checkpoint', edited, [2048,4096], 512)
        assert branch['resume'] == 4096, branch
        e.session('RESTORE', saved, 'RESTORED')
        roundtrip = a.work / 'roundtrip.session'
        e.session('SAVE', roundtrip, 'SAVED')
        summary['session_sha256'] = sha(saved)
        summary['roundtrip_sha256'] = sha(roundtrip)
        summary['session_bytes'] = saved.stat().st_size
        assert summary['session_sha256'] == summary['roundtrip_sha256'], 'SAVE/RESTORE changed session bytes'
        again = e.generate('A-after-explicit-restore', continuation, anchors + [10511], 64)
        assert again['resume'] >= 10511, again
        assert again['generated_ids'] == cont['generated_ids'], 'restored continuation token IDs changed'
    finally:
        e.close()
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    finish(a, config, args, summary, ids, first, tail)


if __name__ == '__main__':
    main()
