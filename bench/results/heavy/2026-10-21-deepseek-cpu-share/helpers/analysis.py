"""Summarize this run's public samples; leave every raw result untouched."""
import hashlib
import json
from pathlib import Path
import re
import statistics

HOME = Path(__file__).resolve().parent


def stats(values):
    v = sorted(x for x in values if isinstance(x, (int, float)))
    if not v:
        return None
    return dict(min=v[0], median=statistics.median(v), max=v[-1], mean=statistics.mean(v))


def summarize(rows):
    if not rows:
        return {}
    out = dict(samples=len(rows), sampled_s=rows[-1]['t']-rows[0]['t'])
    out['cpu_percent'] = stats([r['cpu_percent'] for r in rows])
    for key in ['rss', 'private', 'num_page_faults']:
        out[key] = stats([r['memory'].get(key) for r in rows])
    out['available_ram'] = stats([r['system_ram']['available'] for r in rows])
    for key in ['used', 'spilled', 'shared', 'asked']:
        out['process_gpu_' + key] = stats([r['process_gpu'].get(key) for r in rows])
    for key in ['util', 'sm_mhz', 'memory_mhz', 'power', 'temp', 'pcie_rx_mb', 'pcie_tx_mb', 'mem_used']:
        out['gpu_' + key] = stats([r.get('gpu', {}).get(key) for r in rows])
    out['gpu_throttle_reason_values'] = sorted({r['gpu']['throttle_reasons'] for r in rows
                                               if r.get('gpu', {}).get('throttle_reasons') is not None})
    for scope, key in [('system_disk', 'read_bytes'), ('system_disk', 'write_bytes'),
                       ('io', 'read_bytes'), ('io', 'write_bytes')]:
        out[scope + '_' + key + '_delta'] = rows[-1][scope][key]-rows[0][scope][key]
    out['page_fault_delta'] = rows[-1]['memory']['num_page_faults']-rows[0]['memory']['num_page_faults']
    return out


def prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    summary = {}
    full = {}
    for label in ['mmap-control', 'bat-no-disk']:
        home = HOME / label
        if not (home / 'results.json').exists():
            continue
        data = json.loads((home / 'results.json').read_text())
        full[label] = data
        rows = [json.loads(line) for line in (home / 'samples.jsonl').read_text().splitlines()]
        log = (home / 'engine.stderr').read_text()
        allocation = []
        for phase, payload in re.findall(r'memory\[([^]]+)\] card=0 VRAM allocations total=(.*)', log):
            fields = {k:float(v) for k, v in re.findall(r'(total|dense|experts|KV|staging|rows|other)=([0-9.]+)',
                                                       'total=' + payload)}
            allocation.append(dict(phase=phase, **fields))
        summary[label] = dict(load_s=data['load_s'], exit=data['exit'], command=data['command'],
                              load=summarize([r for r in rows if r['phase']=='load']),
                              all_samples=summarize(rows), allocations_mib=allocation, requests=[])
        for req in data['requests']:
            number = req['number']
            r = {k:v for k,v in req.items() if k not in ['ids', 'token_times', 'cpu_before','cpu_after','io_before','io_after']}
            costs = req.get('decode_costs', {})
            if req['generated'] > 1:
                for key in ['cpu_expert_ms', 'route_wait_ms']:
                    if key in costs: r[key+'_per_output_interval'] = costs[key]/(req['generated']-1)
            r['cpu_process_s'] = sum(req['cpu_after'][k]-req['cpu_before'][k] for k in ['user','system'])
            r['process_io_read_bytes'] = req['io_after']['read_bytes']-req['io_before']['read_bytes']
            times = req['token_times']
            if len(times) >= 128: r['first128_arrival_tps'] = 127/(times[127]-times[0])
            if len(times) >= 256: r['last256_arrival_tps'] = 255/(times[-1]-times[-256])
            r['token_intervals_s'] = stats([b-a for a,b in zip(times,times[1:])])
            r['output_sha256'] = hashlib.sha256(json.dumps(req['ids']).encode()).hexdigest()
            for phase in ['prefill','decode']:
                r[phase+'_samples'] = summarize([x for x in rows if x['phase']==f'request-{number}/{phase}'])
            summary[label]['requests'].append(r)
        summary[label]['warm_decode_tps_median'] = statistics.median(r['decode_tps'] for r in data['requests'][1:])
        summary[label]['warm_hit_rate_median'] = statistics.median(r['decode_hit_rate'] for r in data['requests'][1:])
        summary[label]['cold_start_to_first_token_s'] = data['load_s']+data['requests'][0]['ttft_s']
    if len(full) == 2:
        summary['comparison'] = dict(cross_mode_output_common_prefix=[prefix(a['ids'],b['ids']) for a,b in
            zip(full['mmap-control']['requests'],full['bat-no-disk']['requests'])])
        a,b = summary['mmap-control'],summary['bat-no-disk']
        summary['comparison']['no_disk_warm_speed_delta_percent'] = (b['warm_decode_tps_median']/a['warm_decode_tps_median']-1)*100
    with (HOME / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k:{j:v for j,v in x.items() if j in ['load_s','warm_decode_tps_median','warm_hit_rate_median','cold_start_to_first_token_s']}
                      if k!='comparison' else x for k,x in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
