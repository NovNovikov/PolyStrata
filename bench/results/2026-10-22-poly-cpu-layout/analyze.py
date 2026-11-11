"""Offline analysis of the one-process interleaved CPU experiment; never starts tests."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from collections import defaultdict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
rows=list(csv.DictReader((HERE/'data/timings.csv').open()))
assert len(rows)==90,('incomplete series',len(rows))
assert (HERE/'data/validation.txt').is_file()
groups=defaultdict(list)
for row in rows:
    row['round']=int(row['round']);row['rows']=int(row['rows'])
    for k in ['ms','prepare_ms','gate_up_ms','quantize_ms','down_ms']:row[k]=float(row[k])
    groups[(row['rows'],row['variant'])].append(row)
summary={'rounds':5,'variants':8,'sweeps_per_sample':8,'threads':12,'new_full_model_runs':0,
    'validation':(HERE/'data/validation.txt').read_text().strip(),'variants':{},'controls':{},
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['CMakeLists.txt','src/poly/cpu_experts.cpp','src/poly/cpu_experts.hpp','src/poly/cpu_layout_probe.cpp']}}
for (n,name),data in sorted(groups.items()):
    med=statistics.median(r['ms'] for r in data)
    paired=[]
    for r in data:
        baseline=[t['ms'] for t in groups[(n,'stock')] if t['round']==r['round']]
        paired.append(100*(1-r['ms']/statistics.mean(baseline)))
    summary['variants'][f'{n}/{name}']=dict(median_ms=med,min_ms=min(r['ms'] for r in data),max_ms=max(r['ms'] for r in data),
        median_gate_ms=statistics.median(r['gate_up_ms'] for r in data),
        median_down_ms=statistics.median(r['down_ms'] for r in data),
        paired_time_reduction_percent=paired,paired_median_percent=statistics.median(paired),
        wins=sum(v>0 for v in paired),samples=len(paired))
    print(n,name,'ms',round(med,3),'range',round(min(r['ms'] for r in data),3),round(max(r['ms'] for r in data),3),
          'paired %',round(statistics.median(paired),2),'wins',sum(v>0 for v in paired),'/',len(paired))
for n in (1,4):
    control=[]
    for round in range(5):
        a=[r['ms'] for r in groups[(n,'stock')] if r['round']==round]
        control.append(dict(round=round,ms=a,ratio=max(a)/min(a)))
    summary['controls'][n]=control
with (HERE/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
