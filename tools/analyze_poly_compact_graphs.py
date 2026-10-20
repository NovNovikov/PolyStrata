"""Offline review of explicitly named, preserved synthetic compact-graph experiments."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from poly_memory_report import summarize_consumption


COMPLETE_RUNS = ['control', 'verified', 'sixlane-control', 'sixlane-compact',
                 'rowlayout-control', 'rowlayout-control3', 'rowlayout-control4',
                 'rowlayout-compact', 'rowlayout-compact2']
PAIRS = [('control', 'verified'), ('sixlane-control', 'sixlane-compact'),
         ('rowlayout-control', 'rowlayout-compact'), ('rowlayout-control4', 'rowlayout-compact2'),
         ('rowlayout-control3', 'rowlayout-compact')]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def analyze(source):
    out = dict(source=str(source), method='Offline analysis of preserved synthetic request-boundary data',
               runs={}, pairs=[], parity={})
    raw = {}
    for name in COMPLETE_RUNS:
        p = source/name
        data = json.loads((p/'engine/results.json').read_text(encoding='utf8'))
        summary = json.loads((p/'summary.json').read_text(encoding='utf8'))
        log = (p/'engine/engine.stderr').read_text(encoding='utf8')
        memory = summarize_consumption(data, log)
        warm = [r for r in data['requests'] if r['label'].startswith('warm-repeat-')]
        assert len(warm) == 32 and len(data['requests']) == 36, name
        plan = re.search(r'(\d+) permanent in VRAM', log)
        expert_bytes = re.search(r'VRAM allocations total=[\d.]+ MiB: dense=[\d.]+, experts=([\d.]+)', log)
        slots = re.findall(r'staging_cache=(\d+)/(\d+) slots', log)
        verified = list(map(int, re.findall(r'staging cache verified: (\d+) GPU payloads', log)))
        assert plan and expert_bytes and slots and verified, name
        record = dict(binary_sha256=summary['binary_sha256'], config_sha256=summary['config_sha256'],
                      source_profile_sha256=summary['source_profile_sha256'],
                      source_hashes_at_run_time=summary.get('source_sha256'),
                      compact='--compact-graphs' in data['command'], historical_success=summary['success'],
                      functional_success=summary.get('functional_success'),
                      permanent_gpu_experts=int(plan[1]), permanent_gpu_experts_mib=float(expert_bytes[1]),
                      cache_capacity=max(int(s[1]) for s in slots), verified_cache_payloads=max(verified),
                      warm_median_tps=statistics.median(r['decode_tps'] for r in warm),
                      warm_median_hit_percent=100*statistics.median(r['hit_rate'] for r in warm),
                      warm_guessed_right_histogram=dict(Counter(r['guessed_right'] for r in warm)),
                      long512_tps=data['requests'][0]['decode_tps'], edited512_tps=data['requests'][1]['decode_tps'],
                      maximum_sampled_graph_execs=max(r['graph_execs'] for r in memory['samples']),
                      consumption=memory,
                      evidence_sha256={f:sha(p/f) for f in ['summary.json','engine/results.json','engine/engine.stderr']})
        out['runs'][name] = record
        raw[name] = data
    for control, candidate in PAIRS:
        a,b = (out['runs'][n] for n in (control,candidate))
        x,y = (raw[n]['requests'] for n in (control,candidate))
        identical = len(x) == len(y) and all(q['label']==r['label'] and q['generated_ids']==r['generated_ids'] for q,r in zip(x,y))
        assert identical and a['binary_sha256']==b['binary_sha256'], (control,candidate)
        local = lambda r:r['consumption']['sampled_consumption']['local_mib']['maximum']
        out['pairs'].append(dict(control=control,candidate=candidate,identical_generated_responses=len(x),
                                 same_binary=True,same_config=a['config_sha256']==b['config_sha256'],
                                 same_expert_seed=a['source_profile_sha256']==b['source_profile_sha256'],
                                 sampled_local_peak_reduction_mib=local(a)-local(b),
                                 observed_warm_median_tps_change_percent=100*(b['warm_median_tps']/a['warm_median_tps']-1)))
    for name in ['parity-sixlane.txt','parity-row-layout-final.txt']:
        path = source/name
        log = path.read_text(encoding='utf8')
        passed = re.findall(r'^PASS: batch=.*$',log,re.M)
        assert len(passed)==1 and 'logits=23141120 bit_differences=0 max_abs=0 mean_abs=0' in passed[0],name
        counts = re.findall(r'buffers\[parity\] graphs\[.*?execs=(\d+)',log)
        assert len(counts)==2, (name,'expected reference and candidate graph dumps')
        out['parity'][name] = dict(sha256=sha(path),result=passed[0],reference_execs=int(counts[0]),candidate_execs=int(counts[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=analyze(args.source)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2)
    for name,r in result['runs'].items():
        warm=r['consumption']['warm_last_ten']
        print(name,'Local MiB',round(warm['consumption']['local_mib']['median'],2),
              'graphs',warm['graph_counts'][-1], 'sampled peak',round(r['consumption']['sampled_consumption']['local_mib']['maximum'],2))
    print('OFFLINE_COMPACT_GRAPH_REVIEW_COMPLETE')


if __name__ == '__main__':
    main()
