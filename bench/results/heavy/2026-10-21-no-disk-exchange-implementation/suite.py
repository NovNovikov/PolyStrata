import json, subprocess, sys, time
from pathlib import Path
p=Path(__file__).resolve().parent
labels=['glm-control','glm-exchange-max2','glm-exchange-max4','glm-exchange-max8','glm-smoke-no-graphs','deepseek-smoke']
results={}
for label in labels:
    print('START',label,flush=True)
    with (p/(label+'-console.txt')).open('x') as f:
        r=subprocess.run([sys.executable,str(p/'runner.py'),label],stdout=f,stderr=subprocess.STDOUT)
    results[label]={'exit':r.returncode}
    path=p/(label+'-outcome.json')
    if path.exists():results[label].update(json.loads(path.read_text()))
    print('END',label,results[label],flush=True)
    with (p/(label+'-suite-status.json')).open('x') as f:json.dump(results[label],f,indent=2)
    if not results[label].get('completed'):break
print('SUITE_FINISHED',flush=True)
