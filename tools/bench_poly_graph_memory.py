"""Manual synthetic GPU test: graph creation costs and repeated warm checkpoint restores.

No user sessions/profiles/logs are read. --trace fences each graph creation and changes timings.
"""
import argparse
import errno
import gc
import json
import os
import re
from pathlib import Path
import shutil
import time
import psutil

from bench_poly_checkpoints import Engine, sha
from strata_tokenizer import Tokenizer


class MemoryEngine(Engine):
    def sample_memory(self, phase):
        super().sample_memory(phase)
        memory = psutil.Process(self.p.pid).memory_info()
        self.gpu_memory[-1].update(ram_resident=memory.rss, ram_private=memory.private)

    def record(self):
        # A transient Windows open failure must not discard a loaded model and its measurements.
        # Retry only artifact I/O, outside the engine's timed generation; propagate persistent errors.
        for attempt in range(5):
            try:
                return super().record()
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.EACCES) or attempt == 4:
                    raise
                self.artifact_write_retries = getattr(self,'artifact_write_retries',0) + 1
                print(f'ARTIFACT_WRITE_RETRY {self.home.name} {attempt+1}: {error}',flush=True)
                time.sleep(0.05 * (attempt+1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--engine-arg', action='append', default=[])
    ap.add_argument('--cycles', type=int, default=32)
    ap.add_argument('--sessions', action='store_true', help='also test disk parking, SAVE/RESTORE and process restart')
    ap.add_argument('--no-restart', action='store_true', help='with --sessions, stop after the first process (control run)')
    a = ap.parse_args()
    a.work = a.work.resolve()
    a.output = a.output.resolve()
    a.work.mkdir(parents=True, exist_ok=False)
    a.output.mkdir(parents=True, exist_ok=False)
    sessions = a.work / 'sessions'
    sessions.mkdir()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    args = list(config['args'])
    profile = a.work / 'expert-profile.txt'
    shutil.copyfile(a.profile, profile)
    args[args.index('--session-dir')+1] = str(sessions)
    args[args.index('--expert-profile')+1] = str(profile)
    args += ['--dump-buffers'] + a.engine_arg
    if a.trace:
        args.append('--trace-graphs')
    tok = Tokenizer.from_dir(config['tokenizer'])
    ids = tok.encode('Synthetic checkpoint and adaptive cache integration test.\n' + ''.join(
        f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(15000)))[:10000]
    suffix = tok.encode('Changed synthetic suffix. Cedar copper dune fern.\n' * 2000)[:2048]
    del tok
    gc.collect()
    summary = dict(binary_sha256=sha(Path(config['exe'])), config_sha256=sha(a.config),
                   source_profile_sha256=sha(a.profile), trace=a.trace, cycles=a.cycles,
                   sessions=a.sessions, restart=a.sessions and not a.no_restart, success=False)
    summary['draft_policy_environment'] = {name: os.environ.get(name) for name in (
        'STRATA_TRACE_DRAFT_POLICY', 'STRATA_REFRESH_DRAFT_COSTS')}
    root = Path(__file__).resolve().parents[1]
    summary['source_sha256'] = {name:sha(root/name) for name in (
        'src/poly/engine.cpp','src/poly/engine.hpp','src/poly/deepseek4.hpp','src/poly/main.cpp',
        'include/strata/spec/draft_policy.hpp','src/spec/draft_policy.cpp',
        'tools/bench_poly_graph_memory.py')}
    def record():
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    e = MemoryEngine(config, args, a.output / 'engine')
    try:
        e.ready()
        first = e.generate('long-10k', ids, [2048,4096,6144,8192], 512)
        assert first['resume'] == 0
        if a.sessions:
            e.session('SAVE', a.work/'explicit.session', 'SAVED')
        edited = ids[:5000] + suffix
        branch = e.generate('interior-4k', edited, [2048,4096], 512)
        assert branch['resume'] == 4096, branch
        for cycle in range(a.cycles):
            result = e.generate(f'warm-repeat-{cycle+1}', edited, [2048,4096], 64)
            assert result['resume'] == len(edited)-1, result
        if a.sessions:
            other = list(ids)
            other[0] = next(t for t in ids if t != ids[0])
            switched = e.generate('B-switch', other, [2048,4096,6144,8192], 64)
            assert switched['resume'] == 0, switched
            assert list(sessions.iterdir()), 'automatic parking did not create a session'
            e.session('RESTORE', a.work/'explicit.session', 'RESTORED')
            e.session('SAVE', a.work/'roundtrip.session', 'SAVED')
            summary['session_sha256'] = sha(a.work/'explicit.session')
            assert summary['session_sha256'] == sha(a.work/'roundtrip.session'), 'SAVE/RESTORE bytes changed'
            continued = ids + first['generated_ids'] + suffix[:64]
            resumed = e.generate('A-explicit-restore', continued, [2048,4096,6144,8192,10511], 64)
            assert resumed['resume'] == 10511, resumed
    finally:
        try:
            e.close()
        finally:
            record()
    if a.sessions and not a.no_restart:
        e = MemoryEngine(config, args, a.output/'restart')
        try:
            e.ready()
            resumed = e.generate('A-automatic-after-restart', ids, [2048,4096,6144,8192], 64)
            assert resumed['resume'] >= 8192, resumed
            e.session('RESTORE', a.work/'explicit.session', 'RESTORED')
            changed = list(suffix[:64])
            changed[0] = next(t for t in ids if t != changed[0])
            restored = e.generate('A-isolated-after-restart', ids + first['generated_ids'] + changed,
                                  [2048,4096,6144,8192,10511], 64)
            assert restored['resume'] == 10511, restored
        finally:
            try:
                e.close()
            finally:
                record()
    minimum = int(args[args.index('--vram-reserve-mib')+1])
    summary['checks'] = {}
    for label in ['engine'] + (['restart'] if a.sessions and not a.no_restart else []):
        log = (a.output/label/'engine.stderr').read_text(encoding='utf8',errors='replace')
        free = [int(n)/2**20 for n in re.findall(r'device-memory\[.*?global_free=(\d+)',log)]
        slots = re.findall(r'staging_cache=(\d+)/(\d+) slots',log)
        verified = [int(n) for n in re.findall(r'staging cache verified: (\d+) GPU payloads',log)]
        assert slots and verified, f'no adaptive cache proof in {label}'
        check = dict(minimum_cuda_free_mib=min(free), reserve_mib=minimum,
                     reserve_satisfied=min(free)>=minimum,
                     cache_capacity=max(int(n[1]) for n in slots),
                     cache_used=max(int(n[0]) for n in slots), verified_payloads=max(verified))
        assert check['cache_capacity'] == 255 and check['cache_used'] > 0 and check['verified_payloads'] > 0, check
        if label == 'engine':
            assert check['cache_used'] == check['verified_payloads'] == 255, check
            data = json.loads((a.output/label/'results.json').read_text())
            counts = [int(n) for n in re.findall(r'graphs\[.*?execs=(\d+)',log)]
            warm = [(m,c) for m,c in zip(data['gpu_process_memory'],counts) if m['phase'].startswith('warm-repeat-')][-10:]
            assert len(warm) == 10, 'at least ten warm repeats required'
            check['last_ten_graph_counts'] = [c for m,c in warm]
            check['last_ten_local_spread_mib'] = (max(m['used'] for m,c in warm)-min(m['used'] for m,c in warm))/2**20
            check['plateau'] = len({c for m,c in warm}) == 1 and check['last_ten_local_spread_mib'] <= 4
        summary['checks'][label] = check
    summary['functional_success'] = True
    summary['success'] = all(c['reserve_satisfied'] and c.get('plateau',True) for c in summary['checks'].values())
    record()
    assert summary['success'], summary['checks']
    print('GRAPH_MEMORY_SCENARIO_PASSED', flush=True)


if __name__ == '__main__':
    main()
