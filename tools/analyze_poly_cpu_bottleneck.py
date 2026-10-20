"""Summarize the public CPU/phase probes without modifying their raw results."""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    summary = {}
    for label in ['stock', 'vnni']:
        lines = (a.directory / 'isa-pair' / (label + '.csv')).read_text().splitlines()
        rows = list(csv.DictReader(x for x in lines if not x.startswith('#')))
        assert all(r['bitwise_reference'] == '1' for r in rows)
        summary[label] = {}
        for threads in ['1', '3', '6', '8', '12']:
            selected = [r for r in rows if r['threads'] == threads and r['mode'] == 'stream_layers']
            summary[label][threads] = {
                k: statistics.median(float(r[k]) for r in selected)
                for k in ['ms_per_sweep', 'prepare_ms', 'gate_up_ms', 'quantize_ms', 'down_ms']}
            summary[label][threads]['samples_ms'] = [float(r['ms_per_sweep']) for r in selected]
    log = (a.directory / 'phase-probe/engine/engine.stderr').read_text(encoding='utf8')
    phases = [{k: float(v) for k, v in (x.split('=') for x in line.split()[1:])}
              for line in log.splitlines() if line.startswith('GPU_PHASE ')]
    phases = [x for x in phases if x['pos'] >= 22445]
    assert phases and all(x['layers'] == 43 for x in phases)
    summary['gpu_single_position_samples'] = len(phases)
    keys = [k for k in phases[0] if k.endswith('_ms')]
    summary['gpu_head_phase_medians_ms'] = {k: statistics.median(x[k] for x in phases) for k in keys}
    summary['gpu_head_total_median_ms'] = statistics.median(sum(x[k] for k in keys) for x in phases)
    costs = dict(x.split('=') for x in re.search(r'decode-costs (.*)', log).group(1).split())
    summary['host_costs'] = {k: float(v) for k, v in costs.items()}
    outputs = int(costs['generated']) - 1
    summary['host_ms_per_output_interval'] = {k: float(costs[k]) / outputs
                                              for k in ['wall_ms', 'cpu_expert_ms', 'route_wait_ms']}
    answer = dict(x.split('=') for x in re.search(r'expert-cache answer: ([^;]+)', log).group(1).split())
    positions = (int(answer['GPU']) + int(answer['CPU'])) / (43 * 6)
    summary['evaluated_position_equivalents_from_expert_counts'] = positions
    summary['extra_verification_positions'] = positions - outputs
    summary['cpu_experts_per_evaluated_position_per_layer'] = int(answer['CPU']) / positions / 43
    geometry = json.loads((a.directory / 'attention-geometry.json').read_text())
    counts = {int(k): v for k, v in geometry['actual_ratio_counts'].items()}
    summary['isolated_attention_pipeline_ms'] = {}
    rows = list(csv.DictReader((a.directory / 'gpu-kernels.csv').open()))
    for context in sorted({r['context'] for r in rows}, key=int):
        total = 0
        for row in rows:
            if row['context'] != context:
                continue
            count = counts[int(row['kernel'][12:])] if row['kernel'].startswith('attend_ratio') else counts[4]
            total += float(row['us_per_call']) * count / 1000
        summary['isolated_attention_pipeline_ms'][context] = total
    with (a.directory / 'analysis.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
