"""Allocation-only batch sweep, no inference or private input.

Uses mmap, full-context KV, zero permanent experts and QUIT immediately after READY.
This measures compute allocations without pinning 110+ GiB or conflating a reduced
workspace with a larger automatically planned permanent expert arena.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-exe', type=Path, required=True)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if os.name == 'nt':
        env['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin' + os.pathsep + env.get('PATH', '')
    rows = []
    for batch in (1, 1024, 4096):
        for compact, exe in ((False, a.baseline_exe), (True, a.exe)):
            cmd = [str(exe), '--serve', '--model', str(a.model), '--max-context', '131072',
                   '--prompt-batch', str(batch), '--read-chunk', '32768', '--stage-mib', '2848',
                   '--vram-reserve-mib', '32768', '--rows-vram', '--kv-whole', '--dump-buffers',
                   '--no-draft', '--threads', '12', '--prompt-cache', '0', '--conversation-cache-mib', '0']
            if compact: cmd += ['--compact-scratch']
            label = f'{"compact" if compact else "baseline"}-b{batch}'
            print(label, 'loading allocation inventory', flush=True)
            run = subprocess.run(cmd, input='QUIT\n', capture_output=True, text=True, env=env, timeout=240)
            (a.output / f'{label}.stderr').write_text(run.stderr, encoding='utf-8')
            (a.output / f'{label}.stdout').write_text(run.stdout, encoding='utf-8')
            assert run.returncode == 0 and 'READY ' in run.stdout, (label, run.returncode)
            mem = re.search(r'VRAM allocations total=([\d.]+) MiB: dense=([\d.]+), experts=([\d.]+), '
                            r'KV=([\d.]+), staging=([\d.]+), rows=([\d.]+), other=([\d.]+)', run.stderr)
            assert mem and float(mem[3]) == 0, label
            spans = [(int(off), int(size)) for off, size in
                     re.findall(r'buffers\[load\] scratch .*? offset=\s*(\d+) bytes=\s*(\d+)', run.stderr)]
            assert spans, label
            covered = end = 0
            for off, size in sorted(spans):
                covered += max(0, off + size - max(end, off))
                end = max(end, off + size)
            digest = hashlib.sha256()
            with exe.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    digest.update(block)
            sha = digest.hexdigest()
            row = dict(label=label, command=cmd, exe_sha256=sha, batch=batch,
                       total_mib=float(mem[1]), dense_mib=float(mem[2]), experts_mib=float(mem[3]),
                       kv_mib=float(mem[4]), staging_mib=float(mem[5]), rows_mib=float(mem[6]),
                       other_mib=float(mem[7]), scratch_logical_bytes=sum(size for _, size in spans),
                       scratch_covered_bytes=covered, scratch_extent_bytes=end)
            assert covered == end, row
            rows.append(row)
            (a.output / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in row.items() if k not in ('command', 'exe_sha256')}), flush=True)


if __name__ == '__main__':
    main()
