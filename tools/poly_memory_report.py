"""Summarize process consumption from an explicitly selected synthetic GPU run.

Whole-device free memory is deliberately excluded from consumption acceptance.
Samples are request boundaries, not continuous peaks or proof against every leak.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def _series(values):
    assert values and all(math.isfinite(v) and v >= 0 for v in values), 'invalid consumption samples'
    half = len(values) // 2
    return dict(first=values[0], last=values[-1], minimum=min(values), maximum=max(values),
                median=statistics.median(values), spread=max(values)-min(values),
                latter_half_median_change=(statistics.median(values[half:])-
                                          statistics.median(values[:half])) if half else 0)


def summarize_consumption(data, log, require_warm=True):
    samples = data['gpu_process_memory']
    counts = [int(n) for n in re.findall(r'graphs\[.*?execs=(\d+)', log)]
    allocations = [float(n) for n in re.findall(r'VRAM allocations total=([\d.]+) MiB', log)]
    assert len(samples) == len(counts) == len(allocations) and samples, 'memory/graph/allocation samples do not align'
    counters = {'local_mib': 'used', 'nonlocal_mib': 'spilled', 'committed_mib': 'asked'}
    for sample in samples:
        assert all(isinstance(sample.get(k), (int, float)) for k in counters.values()), 'missing process GPU counters'
    rows = [dict(phase=m['phase'], explicit_device_mib=a, graph_execs=c,
                 **{name: m[key]/2**20 for name, key in counters.items()})
            for m, a, c in zip(samples, allocations, counts)]
    metrics = ['explicit_device_mib', *counters]
    out = dict(units='MiB', samples=rows,
               sampled_consumption={key: _series([r[key] for r in rows]) for key in metrics})
    warm = [r for r in rows if r['phase'].startswith('warm-repeat-')][-10:]
    if require_warm:
        assert len(warm) == 10, 'at least ten warm samples required'
    if len(warm) == 10:
        stats = {key: _series([r[key] for r in warm]) for key in metrics}
        # Keep the existing 4 MiB Local Usage tolerance. Report Non Local and committed
        # separately: registered host pages also contribute, so neither is called VRAM spill.
        # Median halves screen for sustained growth without treating a temporary DMA-window
        # fluctuation as an accumulating allocation. This is a bounded-run heuristic.
        checks = dict(graph_counts_stable=len({r['graph_execs'] for r in warm}) == 1,
                      explicit_allocations_stable=stats['explicit_device_mib']['spread'] <= 0.05,
                      local_usage_stable=stats['local_mib']['spread'] <= 4,
                      nonlocal_no_sustained_growth=stats['nonlocal_mib']['latter_half_median_change'] <= 4,
                      committed_no_sustained_growth=stats['committed_mib']['latter_half_median_change'] <= 4)
        out['warm_last_ten'] = dict(consumption=stats, graph_counts=[r['graph_execs'] for r in warm],
                                    tolerance_mib=4, checks=checks, plateau=all(checks.values()))
    else:
        # A restart with two requests provides consumption figures, not a warm plateau.
        out['warm_last_ten'] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='synthetic engine directory containing results.json and engine.stderr')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads((args.run/'results.json').read_text(encoding='utf8'))
    log = (args.run/'engine.stderr').read_text(encoding='utf8')
    report = summarize_consumption(data, log)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['warm_last_ten'], indent=2))


if __name__ == '__main__':
    main()
