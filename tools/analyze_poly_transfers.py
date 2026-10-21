"""Read numeric STRATA_TRACE_TRANSFERS output; never open prompts, chats or session payloads.

Outputs are exclusive-create. Host API times, GPU intervals and parent spans are deliberately not added.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def analyze(home):
    records = []
    for path in sorted(home.rglob('engine.stderr')):
        for line in path.read_text(encoding='utf8').splitlines():
            if line.startswith('poly-audit '):
                row = json.loads(line[len('poly-audit '):])
                row['run'] = str(path.parent.relative_to(home))
                records.append(row)
    if not records:
        raise RuntimeError('No transfer audit records; no claims can be made from this run')
    grouped = defaultdict(lambda: defaultdict(float))
    sites = defaultdict(lambda: defaultdict(float))
    for row in records:
        if 'phase' not in row:
            continue
        key = (row['run'], row['request'], row['phase'], row['work'], row['op'])
        site = key + (row['file'], row['line'])
        for dest in [grouped[key], sites[site]]:
            for k in ['calls', 'bytes', 'failed', 'host_ms', 'gpu_ms', 'gpu_samples',
                      'busy_before', 'busy_after', 'busy_cleared', 'host_busy_before_ms']:
                dest[k] += row.get(k, 0)
            dest['max_ms'] = max(dest['max_ms'], row.get('max_ms', 0))
    def rows(source, detail=False):
        names = ['run', 'request', 'phase', 'work', 'op'] + (['file', 'line'] if detail else [])
        return [dict(zip(names, key), **value) for key, value in sorted(source.items())]
    groups = rows(grouped)
    dma = []
    for g in groups:
        if g['op'] not in ['host-register', 'host-unregister']:
            continue
        calls, busy = g['calls'], g['busy_before']
        dma.append(dict(g,
            mean_busy_entry_ms=g['host_busy_before_ms']/busy if busy else None,
            mean_idle_entry_ms=(g['host_ms']-g['host_busy_before_ms'])/(calls-busy) if calls>busy else None,
            bytes_per_call=g['bytes']/calls if calls else None))
    failures = [r for r in records if r.get('failed', 0) or any(x in r['op'] for x in
        ['unaccounted', 'failed', 'capacity-exceeded'])]
    scalar_probes = [r for r in records if r.get('op') == 'CPU-memcpy'
        and r.get('file') == 'session_file.hpp' and r.get('calls', 0) > 1000
        and r.get('bytes') == 8*r['calls']]
    return dict(records=records, groups=groups, sites=rows(sites, True), dma=dma,
        observer_warnings=[dict(run=r['run'], request=r['request'], phase=r['phase'],
            site=f"{r['file']}:{r['line']}", calls=r['calls'],
            reason='Scalar word loads were individually probed; parent phase timings are perturbed, not baseline costs')
            for r in scalar_probes],
        diagnostic_failures=failures,
        interpretation=[
            'capture-template entries are construction, excluded from executed transfer totals',
            'graph-replay entries count pointer memcpy nodes at every launch',
            'GPU intervals may include idle/dependencies and overlap; do not sum into wall-time',
            'busy_cleared means watched streams finished during host API; it does not prove internal causation',
            'host-register/unregister bytes are mappings, never additional H2D bytes',
            'file bytes are buffered API payload, not physical disk traffic',
            'cache eviction without use concerns completed/admitted cache lifetimes; active survivors are not waste',
            'speculative cache hits are classified at keep(); draft/verify work remains separate'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('home', type=Path)
    args = p.parse_args()
    result = analyze(args.home)
    with (args.home/'transfer-analysis.json').open('x', encoding='utf8') as out:
        json.dump(result, out, indent=2)
    for r in result['dma']:
        print(json.dumps(r))
    print('audit records:', len(result['records']), 'coverage failures:', len(result['diagnostic_failures']))


if __name__ == '__main__':
    main()
