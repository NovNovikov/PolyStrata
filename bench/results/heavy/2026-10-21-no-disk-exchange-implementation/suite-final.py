import json,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for label in ['deepseek-integration-smoke','glm-exchange-final','glm-mmap']:
    print('START',label,flush=True)
    with (p/(label+'-console.txt')).open('x') as log:
        r=subprocess.run([sys.executable,str(p/'runner.py'),label],stdout=log,stderr=subprocess.STDOUT)
    result={'exit':r.returncode}
    path=p/(label+'-outcome.json')
    if path.exists():result.update(json.loads(path.read_text()))
    with (p/(label+'-suite-status.json')).open('x') as f:json.dump(result,f,indent=2)
    print('END',label,result,flush=True)
    if not result.get('completed'):break
print('FINAL_SUITE_FINISHED',flush=True)
