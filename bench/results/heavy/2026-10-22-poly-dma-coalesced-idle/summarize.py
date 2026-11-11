"""Offline comparison of saved synthetic results. Does not launch an engine."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent/'2026-10-21-poly-dma-coalesced'
CONTROL = HERE.parent/'2026-10-21-poly-transfer-audit'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

summary = {'model_loads_this_series':1,'automatic_retries':0,'baseline_rerun':False,'variants':{}}
for label,home in [('control',CONTROL),('previous_changed',PREVIOUS),('idle_changed',HERE)]:
    result=read(home/'combined/results.json')
    analysis=read(home/'transfer-analysis.json')
    manifest=read(home/'manifest.json')
    rows=[]
    for r in result['requests']:
        n=r['number']
        park=sum(g['host_ms'] for g in analysis['groups'] if g['request']==n and g['phase']=='park' and g['op']=='session-park-total')
        row={k:r[k] for k in ['number','prompt_tokens','read','resume','generated','prompt_ms','decode_ms','decode_tps',
                              'drafts_accepted','drafts_offered','decode_costs']}
        row['park_ms']=park
        row['prefill_excluding_park_ms']=r['prompt_ms']-park
        row['prefill_tps_excluding_park']=r['read']*1000/(r['prompt_ms']-park) if r['read']>1 else None
        row['same_generated_ids_as_previous_changed']=r['ids']==read(PREVIOUS/f'combined/request-{n}.json')['ids']
        rows.append(row)
    summary['variants'][label]={'manifest':manifest,'exit':result['exit'],'coverage_failures':len(analysis['diagnostic_failures']),
                               'load_s':result['load_s'],'requests':rows}

old=summary['variants']['previous_changed']['manifest']; new=summary['variants']['idle_changed']['manifest']
for k in ['executable_sha256','input_sha256','profile_seed_sha256','source_commit']:
    assert old[k]==new[k],k
assert read(PREVIOUS/'public-short-input.json')['ids']==read(HERE/'public-short-input.json')['ids']
exe=Path(new['command'][0])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==new['executable_sha256']
summary['same_exe_inputs_seed_short_input']=True
summary['user_reported_no_interactive_computer_use']=True
summary['background_load_not_experimentally_controlled_in_earlier_series']=True
summary['isolation_limit']='Same changed binary repeat, not a quiet baseline-vs-changed comparison. Adaptive routing and speculative work may differ.'

samples=[json.loads(s) for s in (HERE/'combined/samples.jsonl').read_text(encoding='utf-8').splitlines()]
summary['decode_clock_samples']={str(n):[dict(elapsed=s['elapsed'],sm_mhz=s.get('gpu',{}).get('sm_mhz'),
                                                   memory_mhz=s.get('gpu',{}).get('memory_mhz'))
                                          for s in samples if s['phase']==f'request-{n}/decode'] for n in (1,2,3,4)}
with (HERE/'summary.json').open('x',encoding='utf-8') as f:
    json.dump(summary,f,indent=2)
for label,v in summary['variants'].items():
    print(label,'exit',v['exit'],'failures',v['coverage_failures'])
    for r in v['requests']:
        print(r['number'],r['prefill_tps_excluding_park'],r['decode_tps'],'same ids',r['same_generated_ids_as_previous_changed'])
