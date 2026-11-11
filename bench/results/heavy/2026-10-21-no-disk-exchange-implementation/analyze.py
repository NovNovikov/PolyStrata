import importlib.util,json,re,statistics,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('public_analysis',p.parent/'2026-10-20-glm-mmap-vs-no-disk/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
summary={}
for home in sorted(p.iterdir()):
    if not home.is_dir() or not (home/'results.json').exists():continue
    d=json.loads((home/'results.json').read_text())
    rows=[json.loads(x) for x in (home/'samples.jsonl').read_text().splitlines()]
    log=(home/'engine.stderr').read_text()
    cmd=json.loads((home/'command.json').read_text())
    requests=[]
    for r in d['requests']:
        q={k:r.get(k) for k in ['number','prompt_tokens','read','resume','generated','finish','prompt_ms','decode_ms','decode_tps','decode_hit_rate','drafts_accepted','drafts_offered']}
        q['output_sha256']=hashlib.sha256(json.dumps(r['ids']).encode()).hexdigest()
        q['decode_costs']=r.get('decode_costs')
        q['prefill_samples']=a.summarize([x for x in rows if x['phase']==f"request-{r['number']}/prefill"])
        q['decode_samples']=a.summarize([x for x in rows if x['phase']==f"request-{r['number']}/decode"])
        requests.append(q)
    allocation=[]
    for phase,payload in re.findall(r'memory\[([^]]+)\] card=0 VRAM allocations total=(.*)',log):
        allocation.append(dict(phase=phase,**{k:float(v) for k,v in re.findall(r'(total|dense|experts|KV|staging|rows|other)=([0-9.]+)','total='+payload)}))
    warm=[x for x in rows if x['phase'] in ['request-2/decode','request-3/decode']]
    summary[home.name]=dict(exit=d['exit'],binary_sha256=cmd['exe_sha256'],command=d['command'],load_s=d['load_s'],requests=requests,
        allocations_mib=allocation,warm_hardware=a.summarize(warm),
        warm_tps_median=statistics.median([r['decode_tps'] for r in d['requests'][1:3] if r.get('decode_tps') is not None]) if any(r.get('decode_tps') is not None for r in d['requests'][1:3]) else None,
        exchange_lines=[x for x in log.splitlines() if 'strata-poly: exchange:' in x],
        graph_lines=[x for x in log.splitlines() if 'graphs[pos=' in x],
        placement_lines=[x for x in log.splitlines() if 'expert placement:' in x])
import sys
with (p/(sys.argv[1] if len(sys.argv)>1 else 'summary.json')).open('x') as f:json.dump(summary,f,indent=2)
for name,d in summary.items():print(name,d['exit'],[round(r['decode_tps'],2) if r['decode_tps'] is not None else None for r in d['requests']],round(d['warm_tps_median'],2) if d['warm_tps_median'] is not None else None)
