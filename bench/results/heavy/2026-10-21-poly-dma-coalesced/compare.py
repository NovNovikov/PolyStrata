"""Offline numeric comparison against the existing control; never launches an engine."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / '2026-10-21-poly-transfer-audit'
FIELDS = ('calls', 'bytes', 'host_ms', 'gpu_ms', 'gpu_samples', 'busy_before', 'busy_after', 'busy_cleared')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def summed(rows):
    rows = tuple(rows)
    return {field: sum(row.get(field, 0) for row in rows) for field in FIELDS}

def stage_lines(source):
    lines = source.splitlines()
    first = next(i+1 for i,s in enumerate(lines) if s.startswith('void Engine::stage('))
    last = next(i+1 for i,s in enumerate(lines) if s.startswith('void Engine::moe_rows('))
    return first, last

baseline_source = subprocess.check_output(['git','show','3fa1efa6:src/poly/engine.cpp'],cwd=ROOT).decode('utf-8')
changed_source = (ROOT/'src/poly/engine.cpp').read_text(encoding='utf-8')
out = {'baseline': str(BASE), 'changed': str(HERE), 'baseline_rerun': False, 'variants': {}}
for label, home, source in [('baseline',BASE,baseline_source),('changed',HERE,changed_source)]:
    analysis = read(home/'transfer-analysis.json')
    records = [r for r in analysis['records'] if 'phase' in r]
    results = read(home/'combined/results.json')
    first,last = stage_lines(source)
    variant = {'manifest': read(home/'manifest.json'), 'model_exit': results['exit'],
               'coverage_failures': analysis['diagnostic_failures'], 'requests': [], 'memory': {}}
    for req in results['requests']:
        n = req['number']
        selected = [r for r in records if r['request']==n and r['phase']=='prefill']
        v = {k:req[k] for k in ['number','prompt_tokens','generated','resume','read','prompt_ms','decode_ms',
                              'decode_tps','prompt_tps','drafts_accepted','drafts_offered']}
        v['first_token'] = req['ids'][0] if req['ids'] else None
        v['stage_h2d'] = summed(r for r in selected if r['op']=='H2D' and r['file']=='engine.cpp' and first<=r['line']<last)
        v['main_prefill_dma'] = {op: summed(r for r in selected if r['op']==op and r['layer']>=0)
                                 for op in ['host-register','host-unregister']}
        v['before_head'] = summed(r for r in selected if r['op']=='dma-before-head')
        v['covered_skips'] = summed(r for r in selected if r['op']=='dma-stage-covered-skip')
        v['over_budget_layers'] = summed(r for r in selected if r['op']=='dma-layer-span-over-budget')
        v['cache'] = {op:summed(r for r in records if r['request']==n and r['phase']=='decode' and r['op']==op)
                      for op in ['cache-first-upload','cache-reupload','cache-use','cache-evicted-unused']}
        v['park_spans'] = [g for g in analysis['groups'] if g['request']==n and g['phase']=='park'
                           and ('park' in g['op'] or 'pack' in g['op'])]
        variant['requests'].append(v)
    samples = [json.loads(line) for line in (home/'combined/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    for phase in sorted({s['phase'] for s in samples}):
        group = [s for s in samples if s['phase']==phase]
        def peak(keys):
            values=[]
            for s in group:
                for k in keys:
                    s=s.get(k,{}) if isinstance(s,dict) else {}
                if isinstance(s,(int,float)): values.append(s)
            return max(values) if values else None
        variant['memory'][phase] = {'process_gpu_peak':peak(['process_gpu','used']),
                                    'process_gpu_asked_peak':peak(['process_gpu','asked']),
                                    'process_gpu_spilled_peak':peak(['process_gpu','spilled']),
                                    'process_gpu_shared_peak':peak(['process_gpu','shared']),
                                    'working_set_peak':peak(['memory','rss']),
                                    'private_peak':peak(['memory','private'])}
    variant['ready'] = read(home/'combined/ready.json')
    out['variants'][label] = variant

bm=out['variants']['baseline']['manifest']; cm=out['variants']['changed']['manifest']
out['same_inputs'] = bm['input_sha256']==cm['input_sha256']
out['same_initial_profile'] = bm['profile_seed_sha256']==cm['profile_seed_sha256']
out['short_input_equal'] = read(BASE/'public-short-input.json')['ids']==read(HERE/'public-short-input.json')['ids']
out['verification'] = {'CUDA_build':True,'no_disk_checks':2485,'dma_probe_two_streams_payload_rollback_cleanup':True,
                       'transfer_audit_self_test':True,'HIP_tested':False,'SYCL_tested':False,
                       'full_model_logits_parity_checked':False,'scalar_pack_probe_removed_before_changed_run':True}
with (HERE/'comparison-v2.json').open('x',encoding='utf-8') as f:
    json.dump(out,f,indent=2)
for label,v in out['variants'].items():
    print(label,'exit',v['model_exit'],'coverage failures',len(v['coverage_failures']))
    for r in v['requests']:
        print(json.dumps(r))
