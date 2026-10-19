"""Opt-in GPU benchmark: synthetic text, fresh process per run, no user prompts or logs.

Run with the server's Python environment. --baseline-exe enables a byte comparison of generated token IDs
between the previous binary and the current default path. --no-disk-runs 2 tests independent resident loads.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer
import psutil


def run(a, label, exe, ids, resident=False):
    home = a.output / label
    home.mkdir(parents=True, exist_ok=False)
    profile = home / 'expert-profile.txt'
    if a.profile:
        shutil.copyfile(a.profile, profile)
    cmd = [str(exe), '--serve', '--model', str(a.model), '--max-context', str(a.context),
           '--prompt-batch', str(a.batch), '--stage-mib', str(a.stage_mib), '--threads', str(a.threads),
           '--vram-reserve-mib', str(a.reserve_mib), '--prompt-cache', '0', '--conversation-cache-mib', '0',
           '--no-draft']
    if a.read_chunk:
        cmd += ['--read-chunk', str(a.read_chunk)]
    if a.profile:
        cmd += ['--expert-profile', str(profile)]
    if a.teacher_forced:
        cmd += ['--follow', ','.join(map(str, ids[-a.generate:]))]
    if resident:
        cmd += ['--no-disk', '--no-disk-vram-margin-mib', str(a.load_margin_mib)]
        if a.rows_vram:
            cmd += ['--rows-vram']
        if a.ram_limit_mib:
            cmd += ['--ram-limit-mib', str(a.ram_limit_mib)]
    env = os.environ.copy()
    if a.adaptive_experts:
        env.pop('STRATA_FREEZE_EXPERTS', None)
    else:
        env['STRATA_FREEZE_EXPERTS'] = '1'
    if a.verify_stage_cache:
        env['STRATA_VERIFY_STAGE_CACHE'] = '1'
    else:
        env.pop('STRATA_VERIFY_STAGE_CACHE', None)
    cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin'
    if os.name == 'nt':
        env['PATH'] = cuda + os.pathsep + env.get('PATH', '')
    print(label, 'loading', flush=True)
    started = time.monotonic()
    transcript = []
    with (home / 'engine.stderr').open('w', encoding='utf-8') as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                             text=True, bufsize=1, env=env,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        messages = queue.Queue()
        def reader():
            for line in p.stdout:
                messages.put(line.rstrip())
            messages.put(None)
        threading.Thread(target=reader, daemon=True).start()
        def line(timeout=600):
            text = messages.get(timeout=timeout)
            if text is None:
                raise RuntimeError(f'{label}: engine exited ({p.wait(timeout=10)}); see {home / "engine.stderr"}')
            transcript.append(text)
            return text
        try:
            while not line().startswith('READY '):
                pass
            load_s = time.monotonic() - started
            print(label, f'ready in {load_s:.2f}s', flush=True)
            engine = psutil.Process(p.pid)
            model_open = any(Path(f.path) == a.model for f in engine.open_files())
            if resident: assert not model_open, 'the model file is still open after READY'
            rss_start = engine.memory_info().rss
            rss = [rss_start]
            stop = threading.Event()
            def sample():
                while not stop.wait(.25):
                    try: rss.append(engine.memory_info().rss)
                    except psutil.Error: return
            threading.Thread(target=sample, daemon=True).start()
            io0 = psutil.disk_io_counters()
            swap0 = psutil.swap_memory().used
            process_io0 = engine.io_counters()
            t0 = time.monotonic()
            requests = []
            stdin_bytes = 0
            for request in range(a.requests):
                request_ids = list(ids)
                if request % 2:
                    request_ids[0] = next(t for t in ids if t != ids[0])  # force a new prefill, not prefix reuse
                command = f'GEN {a.generate} temperature=0 seed=42 ' + ','.join(map(str, request_ids)) + '\n'
                stdin_bytes += len(command.encode('ascii')) + (1 if os.name == 'nt' else 0)
                p.stdin.write(command)
                p.stdin.flush()
                tokens, times = [], []
                last = time.monotonic()
                while True:
                    text = line()
                    if text.startswith('T '):
                        tokens.append(int(text.split()[1]))
                        times.append(time.monotonic())
                    if text.startswith(('ERR ', 'FATAL ')): raise RuntimeError(text)
                    if time.monotonic() - last >= 10:
                        print(label, f'request {request+1}', text if text.startswith('PP ') else f'generated {len(tokens)}', flush=True)
                        last = time.monotonic()
                    if text.startswith('DONE '):
                        done = text.split()
                        break
                assert int(done[8]) == 0, done
                if a.teacher_forced:
                    assert tokens == ids[-a.generate:], 'forced token sequence ended early'
                lookups = int(done[10])
                requests.append(dict(generated=len(tokens), prompt_ms=float(done[3]), decode_ms=float(done[4]),
                                     decode_tps=(len(tokens)-1)*1000/float(done[4]) if float(done[4]) else 0,
                                     hits=int(done[9]), lookups=lookups, hit_rate=int(done[9])/lookups if lookups else 0,
                                     finish=done[5], token_times=[round(t-times[0],6) for t in times]))
                print(label, f'request {request+1}', json.dumps({k:v for k,v in requests[-1].items() if k!='token_times'}), flush=True)
            io1 = psutil.disk_io_counters()
            process_io1 = engine.io_counters()
            stop.set()
            result = dict(label=label, resident=resident, rows_vram=resident and a.rows_vram,
                          command=cmd, context=a.context, batch=a.batch, read_chunk=a.read_chunk,
                          reserve_mib=a.reserve_mib, load_margin_mib=a.load_margin_mib if resident else None,
                          adaptive_experts=a.adaptive_experts, teacher_forced=a.teacher_forced, requests=requests,
                          prompt_tokens=len(ids), generated_ids=tokens, load_s=round(load_s, 3),
                          elapsed_s=round(time.monotonic()-t0, 3), prompt_ms=float(done[3]),
                          prompt_tps=round(len(ids)*1000/float(done[3]), 3), finish=done[5],
                          cached_tokens=int(done[8]), rss_start=rss_start, rss_min=min(rss), rss_max=max(rss),
                          swap_growth=psutil.swap_memory().used-swap0,
                          model_file_open_after_load=model_open,
                          process_read_bytes=process_io1.read_bytes-process_io0.read_bytes,
                          process_read_calls=process_io1.read_count-process_io0.read_count,
                          stdin_bytes=stdin_bytes,
                          process_read_bytes_besides_stdin=process_io1.read_bytes-process_io0.read_bytes-stdin_bytes,
                          system_disk_read_bytes=io1.read_bytes-io0.read_bytes if io0 and io1 else None,
                          system_disk_write_bytes=io1.write_bytes-io0.write_bytes if io0 and io1 else None)
            assert result['cached_tokens'] == 0, result
            if resident:
                p.stdin.write('VRAM 0\n'); p.stdin.flush()
                refusal = line()
                assert refusal.startswith('ERR ') and 'no-disk' in refusal, refusal
                result['resize_refused'] = True
            print(json.dumps(result), flush=True)
            return result
        finally:
            try:
                if p.poll() is None:
                    p.stdin.write('QUIT\n'); p.stdin.flush(); p.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                p.kill(); p.wait(timeout=30)
            try: p.stdin.close()
            except OSError: pass
            (home / 'protocol.txt').write_text('\n'.join(transcript), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', type=Path, required=True)
    ap.add_argument('--baseline-exe', type=Path)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--profile', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--tokens', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--read-chunk', type=int, default=0)
    ap.add_argument('--stage-mib', type=int, default=2048)
    ap.add_argument('--reserve-mib', type=int, default=2048)
    ap.add_argument('--load-margin-mib', type=int, default=512)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--generate', type=int, default=8)
    ap.add_argument('--requests', type=int, default=1)
    ap.add_argument('--adaptive-experts', action='store_true')
    ap.add_argument('--verify-stage-cache', action='store_true', help='compare published GPU cache payloads with RAM after each request, outside decode timing')
    ap.add_argument('--teacher-forced', action='store_true', help='replay a fixed synthetic token sequence for cache/performance comparisons')
    ap.add_argument('--ram-limit-mib', type=int, default=0)
    ap.add_argument('--rows-vram', action='store_true')
    ap.add_argument('--no-disk-runs', type=int, default=2)
    ap.add_argument('--skip-default', action='store_true')
    a = ap.parse_args()
    if not 16 <= a.tokens < a.context: ap.error('tokens must be at least 16 and below context')
    if a.read_chunk < 0 or a.load_margin_mib < 0: ap.error('read chunk and load margin cannot be negative')
    if a.generate < 1 or a.requests < 1: ap.error('generate and requests must be positive')
    if a.teacher_forced and a.generate > a.tokens: ap.error('forced output cannot exceed synthetic source length')
    a.output.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_dir(a.tokenizer)
    text = 'Reference data for a synthetic performance test.\n' + ''.join(
        f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(a.tokens))
    ids = tok.encode(text)[:a.tokens]
    assert len(ids) == a.tokens
    del tok, text  # the runner must not retain a tokenizer/model-sized vocabulary beside a tight RAM load
    import gc
    gc.collect()
    results = []
    jobs = []
    if a.baseline_exe: jobs.append(('baseline-default', a.baseline_exe, False))
    if not a.skip_default: jobs.append(('current-default', a.exe, False))
    jobs += [(f'no-disk-{n+1}', a.exe, True) for n in range(a.no_disk_runs)]
    for label, exe, resident in jobs:
        results.append(run(a, label, exe, ids, resident))
        (a.output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if a.baseline_exe and not a.skip_default:
        assert results[0]['generated_ids'] == results[1]['generated_ids'], 'default output token IDs differ'
        print('DEFAULT_TOKEN_BYTES_IDENTICAL', flush=True)

if __name__ == '__main__': main()
