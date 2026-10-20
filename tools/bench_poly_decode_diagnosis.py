"""Fresh, isolated decode diagnosis using only public repository text and a public expert seed.

Does not read private logs, prompts, sessions or the user's expert profile. No ports are opened.
All outputs are exclusive-create snapshots; existing work/output directories are refused.
"""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import threading
import time

import psutil
from bench_poly_checkpoints import Engine, sha
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from serve.telemetry import gpu_reader


class Diagnosis(Engine):
    def __init__(self, config, args, home):
        self.snapshot = 0
        self.phase = 'load'
        self.hardware = []
        self.stop_sampling = threading.Event()
        super().__init__(config, args, home)
        self.process = psutil.Process(self.p.pid)
        self.gpu = gpu_reader()
        self.sampler = threading.Thread(target=self.sample_hardware, daemon=True)
        self.sampler.start()

    def sample_hardware(self):
        self.process.cpu_percent(None)
        while not self.stop_sampling.wait(0.5):
            try:
                m=self.process.memory_info()
                row=dict(t=time.monotonic(),phase=self.phase,process_cpu_percent=self.process.cpu_percent(None),
                         rss=m.rss,private=m.private,system_available=psutil.virtual_memory().available)
                if self.gpu.ok():
                    row['gpu']=self.gpu.read()
                    row['gpu']['sm_mhz']=self.gpu._uint('nvmlDeviceGetClockInfo',ctypes.c_uint(1))
                    row['gpu']['memory_mhz']=self.gpu._uint('nvmlDeviceGetClockInfo',ctypes.c_uint(2))
                self.hardware.append(row)
            except (psutil.Error,OSError):
                break

    def record(self):
        self.snapshot+=1
        data=dict(command=self.command,load_s=getattr(self,'load_s',None),requests=self.results,
                  gpu_process_memory=self.gpu_memory,hardware=self.hardware)
        path=self.home/('results.json' if self.p.poll() is not None else f'snapshot-{self.snapshot:03d}.json')
        with path.open('x',encoding='utf8') as f:json.dump(data,f,indent=2)
        with (self.home/f'protocol-{self.snapshot:03d}.txt').open('x',encoding='utf8') as f:f.write('\n'.join(self.lines)+'\n')

    def generate(self,label,ids,count):
        self.phase=label+'/prefill'
        start=time.monotonic()
        self.send(f'GEN {count} temperature=0 seed=42 checkpoints=2048,4096,8192,16384 '+','.join(map(str,ids)))
        tokens=[]; stamp=[]; resume=None; progress=start
        while True:
            line=self.line()
            if line.startswith('RESUME '):resume=int(line.split()[1])
            if line.startswith('T '):
                self.phase=label+'/decode';tokens.append(int(line.split()[1]));stamp.append(time.monotonic())
            if time.monotonic()-progress>10:
                print(label,line if line.startswith('PP ') else f'generated={len(tokens)}',flush=True)
                progress=time.monotonic()
            if line.startswith('DONE '):d=line.split();break
        assert len(tokens)==int(d[1]) and resume==int(d[8]),d
        n=len(tokens)
        result=dict(label=label,prompt_tokens=len(ids),requested=count,generated_ids=tokens,resume=resume,
                    prompt_ms=float(d[3]),decode_ms=float(d[4]),finish=d[5],guessed_right=int(d[6]),
                    guessed=int(d[7]),hits=int(d[9]),lookups=int(d[10]),read=int(d[14]),
                    decode_tps=(n-1)*1000/float(d[4]) if n>1 and float(d[4]) else None,
                    token_times=[round(t-stamp[0],6) for t in stamp])
        result['hit_rate']=result['hits']/max(1,result['lookups'])
        if n>=128:result['first128_arrival_tps']=127/(stamp[127]-stamp[0])
        self.results.append(result)
        self.phase=label+'/done';self.sample_memory(label);self.record()
        print(label,json.dumps({k:v for k,v in result.items() if k not in ('generated_ids','token_times')}),flush=True)
        return result

    def close(self):
        self.stop_sampling.set();self.sampler.join(timeout=2)
        super().close()


def prompts(config,root,target):
    tok=Tokenizer.from_dir(config['tokenizer'])
    template=ChatTemplate(Path(config['tokenizer'])/'chat_template.jinja')
    files=['docs/DETAILS.md','src/poly/engine.hpp','src/poly/cpu_experts.cpp','serve/frontend.py']
    sources={name:sha(root/name) for name in files}
    varied='\n\n'.join(f'Public source: {name}\n'+(root/name).read_text(encoding='utf8') for name in files)
    repetitive=''.join(f'{i:05d}: amber birch cobalt delta elm frost.\n' for i in range(15000))
    result={}
    for name,text in [('repetitive',repetitive),('varied',varied)]:
        content_ids=tok.encode(text)
        n=target-200
        for _ in range(8):
            content=tok.decode(content_ids[:n])
            request='Analyze the following public technical material. Explain its mechanisms, tradeoffs, and concrete failure modes in detail.\n\n'+content+'\n\nGive a detailed analysis grounded in the material.'
            rendered=template.render([dict(role='system',content='You are a technical reviewer. Analyze the supplied public material.'),dict(role='user',content=request)],enable_thinking=True)
            ids=tok.encode(rendered)
            if abs(len(ids)-target)<=2:break
            n+=target-len(ids)
        assert abs(len(ids)-target)<=8,(name,len(ids),target)
        result[name]=ids
    del tok,template;gc.collect()
    return result,sources


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prompt-tokens',type=int,default=22446)
    a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=False);a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    config=json.loads(a.config.read_text(encoding='utf-8-sig'));args=list(config['args'])
    seed=a.work/'expert-profile.txt';shutil.copyfile(a.profile,seed)
    args[args.index('--expert-profile')+1]=str(seed.resolve())
    # Keep the production checkpoint count, but isolate decode from disk parking in this diagnostic.
    args[args.index('--session-dir')+1]=''
    args+=['--dump-buffers']
    os.environ['STRATA_TRACE_DECODE_COSTS']='1';os.environ['STRATA_TRACE_DRAFT_POLICY']='1'
    os.environ.pop('STRATA_REFRESH_DRAFT_COSTS',None)
    inputs,sources=prompts(config,root,a.prompt_tokens)
    with (a.output/'inputs.json').open('x',encoding='utf8') as f:json.dump(inputs,f)
    provenance=dict(binary_sha256=sha(Path(config['exe'])),config_sha256=sha(a.config),
                    expert_seed_sha256=sha(a.profile),public_source_sha256=sources,
                    private_data_read=False,disk_parking_disabled=True,trace_costs=True,trace_policy=True,
                    source_sha256={name:sha(root/name) for name in ['src/poly/main.cpp','src/poly/engine.cpp','tools/bench_poly_decode_diagnosis.py']})
    with (a.output/'provenance.json').open('x',encoding='utf8') as f:json.dump(provenance,f,indent=2)
    e=Diagnosis(config,args,a.output/'engine')
    try:
        e.ready()
        for kind in ['repetitive','varied']:
            ids=inputs[kind]
            first=e.generate(kind+'-cold512',ids,512)
            e.generate(kind+'-continuation128',ids+first['generated_ids'],128)
            for repeat in range(1,3):e.generate(kind+f'-warm-repeat-{repeat}',ids,128)
    finally:e.close()
    log=(a.output/'engine/engine.stderr').read_text(encoding='utf8')
    costs=[dict(x.split('=',1) for x in line.split()) for line in re.findall(r'strata-poly: decode-costs (.*)',log)]
    assert len(costs)==len(e.results),'missing decode timers'
    summary=[]
    for request,cost in zip(e.results,costs):
        samples=[x for x in e.hardware if x['phase']==request['label']+'/decode']
        numeric={k:v for k,v in request.items() if k not in ('generated_ids','token_times')}
        numeric['costs']={k:float(v) for k,v in cost.items()}
        numeric['hardware_decode_samples']=len(samples)
        for key in ['process_cpu_percent','system_available']:
            if samples:numeric[key+'_median']=statistics.median(x[key] for x in samples)
        for key in ['util','power','sm_mhz','memory_mhz']:
            values=[x['gpu'][key] for x in samples if x.get('gpu',{}).get(key) is not None]
            if values:numeric['gpu_'+key+'_median']=statistics.median(values)
        summary.append(numeric)
    with (a.output/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2)
    print('DECODE_DIAGNOSIS_COMPLETED',flush=True)


if __name__=='__main__':main()
