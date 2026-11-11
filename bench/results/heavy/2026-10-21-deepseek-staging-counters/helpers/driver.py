"""Same-day DeepSeek comparison using launch JSON, opaque profile copies and public text."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HOME = Path(__file__).resolve().parent
# the share of the uncached routed experts the CPU computes under the batch, with the speculative
# copy-ahead off: with it on (a batch of 4096 tokens is over stage_ahead) the CPU share never applies.
CPU_SHARE = {'bat-cpu-0':'0.0','bat-cpu-50':'0.5','bat-cpu-75':'0.75'}
RUN_ORDER = ['bat-no-disk','bat-exchange','bat-mmap']
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('exchange_runner', Path(__file__).with_name('engine_runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HOME = HOME
spec2 = importlib.util.spec_from_file_location('public_analysis', Path(__file__).with_name('analysis.py'))
analysis = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(analysis)


class Run(runner.Run):
    def line(self):
        line = super().line()
        if hasattr(self, 'progress') and line.startswith(('RESUME ', 'PP ')):
            self.progress.append(dict(line=line, wall_s=time.monotonic()))
        return line

    def generate(self, number, ids, max_new=256):
        self.progress = []
        super().generate(number, ids, max_new)
        req = self.requests[-1]
        resume = next(x['wall_s'] for x in self.progress if x['line'].startswith('RESUME '))
        last = [x for x in self.progress if x['line'].startswith('PP ')][-1]
        req['resume_to_last_pp_s'] = last['wall_s'] - resume
        req['resume_to_last_pp_tps'] = req['read'] / req['resume_to_last_pp_s']
        req['pp'] = self.progress
        runner.save(self.home / f'request-{number}-timing.json',
                    {k:req[k] for k in ['read','resume','prompt_ms','prompt_tps','resume_to_last_pp_s','resume_to_last_pp_tps','pp']})
        if req['resume'] != 0 or req['read'] != 50000:
            raise RuntimeError('Full prefill was not measured: resume/read mismatch')


def inputs(config):
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    tok = Tokenizer.from_dir(config['tokenizer'])
    template = ChatTemplate(Path(config['tokenizer']) / 'chat_template.jinja')
    sources = [ROOT / x for x in ['docs/DETAILS.md','docs/GLM.md','docs/BACKENDS.md','README.md']]
    corpus = '\n\n'.join(p.read_text(encoding='utf-8') for p in sources)
    while len(tok.encode(corpus)) < 52000:
        corpus += '\n\n' + corpus
    result = []
    for tag in ['A', 'B']:
        text = (f'Independent engineering task {tag}. Read this public documentation, then explain the '
                'RAM/VRAM expert placement and caching mechanisms. Give a detailed technical answer.\n\n' + corpus)
        rendered = template.render([dict(role='user', content=text)], enable_thinking=False)
        all_ids = tok.encode(rendered)
        ids = all_ids[:49968] + all_ids[-32:]
        assert len(ids) == 50000
        result.append(ids)
    common = next((i for i,(a,b) in enumerate(zip(*result)) if a != b),50000)
    assert common < 32
    runner.save(HOME / 'public-inputs.json',dict(ids=result, tokens=[len(x) for x in result],
        common_prefix=common, sources={str(p.relative_to(ROOT)):runner.sha(p) for p in sources},
        tokenizer_template_sha256=runner.sha(Path(config['tokenizer'])/'chat_template.jinja')))
    del tok,template,corpus
    gc.collect()
    return result


def summarize():
    out = {}
    for label in RUN_ORDER:
        home = HOME / label
        if not (home/'results.json').exists(): continue
        d = json.loads((home/'results.json').read_text())
        rows = [json.loads(x) for x in (home/'samples.jsonl').read_text().splitlines()]
        log = (home/'engine.stderr').read_text()
        alloc = []
        for phase,payload in re.findall(r'memory\[([^]]+)\] card=0 VRAM allocations total=(.*)',log):
            alloc.append(dict(phase=phase,**{k:float(v) for k,v in re.findall(r'(total|dense|experts|KV|staging|rows|other)=([0-9.]+)','total='+payload)}))
        reqs=[]
        for r in d['requests']:
            q={k:v for k,v in r.items() if k not in ['ids','token_times','pp','io_before','io_after','cpu_before','cpu_after']}
            q['output_sha256']=hashlib.sha256(json.dumps(r['ids']).encode()).hexdigest()
            for phase in ['prefill','decode']:
                q[phase+'_samples']=analysis.summarize([x for x in rows if x['phase']==f"request-{r['number']}/{phase}"])
            reqs.append(q)
        out[label]=dict(load_s=d['load_s'],exit=d['exit'],command=d['command'],requests=reqs,
            allocations_mib=alloc,placement_lines=[x for x in log.splitlines() if 'expert placement:' in x],
            graph_lines=[x for x in log.splitlines() if 'graphs[pos=' in x],
            exchange_lines=[x for x in log.splitlines() if 'strata-poly: exchange:' in x],
            park_lines=[x for x in log.splitlines() if 'conversation cache: parked' in x],
            prefill_progress_file='protocol.txt')
    runner.save(HOME/'summary.json',out)
    for name,v in out.items():
        print('SUMMARY',name,'load_s',round(v['load_s'],2),
              [{k:round(r[k],3) if isinstance(r[k],float) else r[k] for k in ['read','resume','prompt_tps','resume_to_last_pp_tps','decode_tps','decode_hit_rate']} for r in v['requests']],flush=True)


def main():
    import psutil
    if any((p.info['name'] or '').lower()=='strata-poly.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another engine is running; leaving it untouched')
    cfg_path=ROOT/'strata-dsv4-q3_k_m.json'
    config=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    original=list(config['args'])
    profile=Path(original[original.index('--expert-profile')+1])
    watched={str(p):runner.sha(p) for p in [cfg_path,ROOT/'run-dsv4-q3_k_m.bat',profile,Path(config['exe'])]}
    seed=HOME/'initial-expert-profile.txt'
    if not seed.exists():
        with seed.open('xb') as out, profile.open('rb') as src: shutil.copyfileobj(src,out)
    else:
        frozen=json.loads((HOME/'handoff-manifest.json').read_text())
        assert runner.sha(seed)==frozen['profile_seed_sha256']
    runner.save(HOME/'manifest.json',dict(original_command=[config['exe'],'--serve']+original,
        hashes=watched,profile_policy='Identical opaque copies of the prepared statistics seed; original never written',
        native_protocol=True,outputs_per_request=256,prompt_tokens=50000,
        order=RUN_ORDER,exchange_max=1,cpu_share_variants=CPU_SHARE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    prompts=inputs(config)
    for label in RUN_ORDER:
        print('START',label,flush=True)
        work=HOME/(label+'-work');work.mkdir()
        seed_copy=work/'expert-profile.txt';shutil.copyfile(seed,seed_copy)
        args=list(original)
        for flag,value in [('--expert-profile',str(seed_copy)),('--session-dir',str(work/'sessions'))]:
            args[args.index(flag)+1]=value
        if label=='bat-exchange':args+=['--no-disk-expert-exchange']
        if label=='bat-mmap':args.remove('--no-disk')
        if label in CPU_SHARE:args+=['--stage-ahead','0','--cpu-share',CPU_SHARE[label]]
        args+=['--dump-buffers']
        expected=[config['exe'],'--serve']+original
        normalized=[config['exe'],'--serve']+list(args)
        normalized.remove('--dump-buffers')
        for flag in ['--expert-profile','--session-dir']:
            normalized[normalized.index(flag)+1]=expected[expected.index(flag)+1]
        if label=='bat-exchange':normalized.remove('--no-disk-expert-exchange')
        if label=='bat-mmap':expected.remove('--no-disk')
        if label in CPU_SHARE:      # the flags under measurement, not the configuration's
            i=normalized.index('--stage-ahead')
            del normalized[i:i+4]
            assert '--stage-ahead' not in expected and '--cpu-share' not in expected
        assert normalized==expected
        if runner.sha(seed_copy)!=runner.sha(seed):raise RuntimeError('Seed mismatch')
        config['test_env']={'STRATA_TRACE_STAGING':'1'}   # the staging counters
        run=Run(label,config,args,work)
        outcome={}
        try:
            run.ready(config)
            for i,ids in enumerate(prompts,1):
                run.generate(i,ids,256)
                if run.requests[-1]['generated'] < 32:
                    raise RuntimeError('Unexpectedly short answer: preserve data and diagnose input/logits before more model runs')
            outcome['completed']=True
        except Exception as e:
            outcome['error']=str(e)
            print('FAILED',label,str(e),flush=True)
        finally:
            run.close()
        runner.save(HOME/(label+'-outcome.json'),outcome)
        print('FINISHED',label,outcome,flush=True)
        if not outcome.get('completed'):break
    summarize()
    runner.save(HOME/'final-verification.json',dict(
        original_files_unchanged={p:runner.sha(p)==h for p,h in watched.items()},
        no_engine_running=not any((p.info['name'] or '').lower()=='strata-poly.exe' for p in psutil.process_iter(['name']))))


if __name__=='__main__':main()
