"""Sequential GLM comparison. Public prompt/profile, fresh stores, immutable result files."""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import psutil
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from serve.telemetry import gpu_reader, gpu_process_memory

HOME = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, data):
    with path.open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2)


def prepare(config):
    tok = Tokenizer.from_dir(config['tokenizer'])
    template = ChatTemplate(Path(config['tokenizer']) / 'chat_template.jinja')
    source = ROOT / 'docs/GLM.md'
    material = tok.encode(source.read_text(encoding='utf8'))
    n = 1800
    for _ in range(6):
        text = ('Read the following public documentation. Write a detailed engineering explanation of at least '
                '1200 words about how this mixture-of-experts engine places weights, uses RAM and VRAM, '
                'processes a prompt, maintains the expert cache and generates an answer. Explain the role of '
                'the draft model, memory bandwidth, cache warm-up, and the measurements needed to compare '
                'two configurations fairly. Discuss concrete failure cases and remedies. Do not merely '
                'repeat the text; explain the mechanisms and their tradeoffs.\n\n' + tok.decode(material[:n]))
        rendered = template.render([dict(role='system', content='You are a careful systems engineer.'),
                                    dict(role='user', content=text)], enable_thinking=False)
        ids = tok.encode(rendered)
        if abs(len(ids) - 2048) < 3:
            break
        n += 2048 - len(ids)
    save(HOME / 'input.json', dict(ids=ids, tokens=len(ids), rendered=rendered, source_sha256=sha(source),
                                  template_sha256=sha(Path(config['tokenizer']) / 'chat_template.jinja')))
    del tok, template
    gc.collect()
    return ids


class Run:
    def __init__(self, label, config, args, work):
        self.label = label
        self.home = HOME / label
        self.home.mkdir()
        self.phase = 'load'
        self.rows = []
        self.requests = []
        self.stop = threading.Event()
        self.err = (self.home / 'engine.stderr').open('x', encoding='utf8')
        self.protocol = (self.home / 'protocol.txt').open('x', encoding='utf8', buffering=1)
        self.samples = (self.home / 'samples.jsonl').open('x', encoding='utf8', buffering=1)
        env = {k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
        for name in ['STRATA_FREEZE_EXPERTS', 'STRATA_VERIFY_STAGE_CACHE', 'STRATA_PROFILE_DECODE',
                     'STRATA_REFRESH_DRAFT_COSTS', 'STRATA_ROUTE_TAIL_SKIP']:
            env.pop(name, None)
        env['STRATA_TRACE_DECODE_COSTS'] = '1'
        env['STRATA_TRACE_DRAFT_POLICY'] = '1'
        env.update(config.get('test_env', {}))
        env['PATH'] = os.pathsep.join(config.get('lib_dirs', []) + [env.get('PATH', '')])
        self.command = [config['exe'], '--serve'] + args
        self.started = time.monotonic()
        self.p = subprocess.Popen(self.command, cwd=config['cwd'], env=env, text=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        self.process = psutil.Process(self.p.pid)
        self.gpu = gpu_reader()
        self.q = queue.Queue()
        def read():
            for line in self.p.stdout:
                self.q.put(line.rstrip())
            self.q.put(None)
        threading.Thread(target=read, daemon=True).start()
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()
        self.last_progress = 0
        self.work = work
        save(self.home / 'command.json', dict(command=self.command, pid=self.p.pid,
             exe_sha256=sha(config['exe']), environment={k:env[k] for k in env if k.startswith('STRATA_')}))

    def sample(self):
        self.process.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)
        while not self.stop.is_set():
            try:
                m = self.process.memory_info()
                v = psutil.virtual_memory()
                row = dict(t=time.monotonic(), elapsed=time.monotonic()-self.started, phase=self.phase,
                           cpu_percent=self.process.cpu_percent(None), memory=m._asdict(),
                           threads=self.process.num_threads(), io=self.process.io_counters()._asdict(),
                           cpu_times=self.process.cpu_times()._asdict(),
                           system_ram=dict(total=v.total, available=v.available, used=v.used),
                           system_cpu_percent_per_core=psutil.cpu_percent(None, percpu=True),
                           system_disk=psutil.disk_io_counters()._asdict(),
                           process_gpu=gpu_process_memory(self.p.pid))
                if self.gpu.ok():
                    row['gpu'] = self.gpu.read()
                    row['gpu']['sm_mhz'] = self.gpu._uint('nvmlDeviceGetClockInfo', ctypes.c_uint(1))
                    row['gpu']['memory_mhz'] = self.gpu._uint('nvmlDeviceGetClockInfo', ctypes.c_uint(2))
                    reasons = ctypes.c_ulonglong()
                    try:
                        status = self.gpu.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.gpu.dev, ctypes.byref(reasons))
                        if status == 0: row['gpu']['throttle_reasons'] = reasons.value
                    except (AttributeError, OSError):
                        pass
                self.rows.append(row)
                self.samples.write(json.dumps(row) + '\n')
            except (psutil.Error, OSError):
                return
            self.stop.wait(.5)

    def line(self):
        while True:
            try:
                line = self.q.get(timeout=1)
                if line is None:
                    raise RuntimeError('engine exited: ' + str(self.p.wait()))
                self.protocol.write(line + '\n')
                if line.startswith(('ERR ', 'FATAL ', 'SERR ')):
                    raise RuntimeError(line)
                return line
            except queue.Empty:
                if time.monotonic() - self.last_progress > 10:
                    self.last_progress = time.monotonic()
                    print(self.label, self.phase, 'elapsed_s', round(time.monotonic()-self.started), flush=True)

    def ready(self, config):
        while not self.line().startswith('READY '):
            pass
        self.load_s = time.monotonic() - self.started
        wanted = {str(Path(config['args'][config['args'].index(flag)+1]).resolve()).lower()
                  for flag in ['--model', '--draft-model'] if flag in config['args']}
        opened = {str(Path(f.path).resolve()).lower() for f in self.process.open_files()}
        save(self.home / 'ready.json', dict(load_s=self.load_s, memory=self.process.memory_info()._asdict(),
             process_gpu=gpu_process_memory(self.p.pid), model_or_draft_handles_open=sorted(wanted & opened)))
        print(self.label, 'READY', round(self.load_s, 2), flush=True)

    def generate(self, number, ids, max_new=512):
        label = 'request-' + str(number)
        self.phase = label + '/prefill'
        sent = time.monotonic()
        io_before = self.process.io_counters()._asdict()
        cpu_before = self.process.cpu_times()._asdict()
        command = f'GEN {max_new} temperature=0 seed=42 ' + ','.join(map(str, ids)) + '\n'
        self.p.stdin.write(command)
        self.p.stdin.flush()
        tokens, stamps, resume = [], [], None
        progress = sent
        while True:
            line = self.line()
            if line.startswith('RESUME '): resume = int(line.split()[1])
            if line.startswith('T '):
                self.phase = label + '/decode'
                tokens.append(int(line.split()[1]))
                stamps.append(time.monotonic())
            if time.monotonic()-progress > 10:
                print(self.label, label, line if line.startswith('PP ') else 'generated=' + str(len(tokens)), flush=True)
                progress = time.monotonic()
            if line.startswith('DONE '):
                d = line.split()
                break
        elapsed = time.monotonic()-sent
        assert len(tokens) == int(d[1]) and resume == int(d[8])
        result = dict(number=number, prompt_tokens=len(ids), generated=len(tokens), requested=max_new,
                      ids=tokens, resume=resume, prompt_ms=float(d[3]), decode_ms=float(d[4]), finish=d[5],
                      drafts_accepted=int(d[6]), drafts_offered=int(d[7]), raw_done_gpu=int(d[9]),
                      raw_done_lookups=int(d[10]), read=int(d[14]), elapsed_s=elapsed,
                      ttft_s=stamps[0]-sent if stamps else None,
                      decode_tps=(len(tokens)-1)*1000/float(d[4]) if len(tokens)>1 else None,
                      prompt_tps=int(d[14])*1000/float(d[3]) if float(d[3]) else None,
                      token_times=[t-stamps[0] for t in stamps], stdin_bytes=len(command.encode())+1,
                      io_before=io_before, io_after=self.process.io_counters()._asdict(),
                      cpu_before=cpu_before, cpu_after=self.process.cpu_times()._asdict(),
                      process_gpu_after=gpu_process_memory(self.p.pid))
        self.phase = label + '/done'
        save(self.home / (label + '.json'), result)
        self.requests.append(result)
        print(self.label, label, json.dumps({k:result[k] for k in ['generated','prompt_ms','decode_ms','decode_tps','ttft_s','resume','drafts_accepted','drafts_offered']}), flush=True)

    def close(self):
        self.phase = 'shutdown'
        if self.p.poll() is None:
            self.p.stdin.write('QUIT\n')
            self.p.stdin.flush()
        self.p.wait()  # graceful only: never kill a long shutdown
        self.stop.set()
        self.sampler.join()
        self.err.close()
        self.protocol.close()
        self.samples.close()
        text = (self.home / 'engine.stderr').read_text(encoding='utf8')
        traces = [{k:float(v) for k,v in (x.split('=') for x in line.split())}
                  for line in re.findall(r'strata-poly: decode-costs (.*)', text)]
        for req, costs in zip(self.requests, traces):
            req['decode_costs'] = costs
            req['decode_hit_rate'] = costs['gpu_experts']/max(1,costs['gpu_experts']+costs['cpu_experts'])
        save(self.home / 'results.json', dict(command=self.command, load_s=getattr(self,'load_s',None),
             requests=self.requests, exit=self.p.returncode))


def native(run, command, error=False):
    run.p.stdin.write(command+'\n'); run.p.stdin.flush()
    while True:
        line=run.q.get(timeout=120)
        if line is None:raise RuntimeError('engine exited during '+command)
        run.protocol.write(line+'\n')
        if line.startswith(('EXCHANGED','ERR EXCHANGE')):
            if error != line.startswith('ERR EXCHANGE'):raise RuntimeError(line)
            print(command,line,flush=True)
            return line

def main():
    label=sys.argv[1]
    base=json.loads((ROOT/'strata-glm-3.0bit.json').read_text(encoding='utf-8-sig'))
    ids=json.loads((ROOT/'bench/results/2026-10-20-glm-mmap-vs-no-disk/input.json').read_text())['ids']
    if label.startswith('deepseek'):
        base=json.loads((ROOT/'strata-dsv4-q3_k_m.json').read_text(encoding='utf-8-sig'))
        # Same public technical request, rendered/tokenized with this architecture, no private data.
        tok=Tokenizer.from_dir(base['tokenizer'])
        template=ChatTemplate(Path(base['tokenizer'])/'chat_template.jinja')
        public=(ROOT/'docs/GLM.md').read_text(encoding='utf8')
        text=template.render([dict(role='user',content='Explain this public engine documentation in detail.\n'+public)],enable_thinking=False)
        ids=tok.encode(text)[:2050]
        del tok,template;gc.collect()
    if any((p.info['name'] or '').lower()=='strata-poly.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('another engine is running')
    if label=='glm-smoke-no-graphs':
        print('building failure-path hardening before smoke tests',flush=True)
        with (HOME/'build-6.txt').open('x') as log:
            built=subprocess.run(['cmd','/c',str(HOME/'build-tests.cmd')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if built.returncode:raise RuntimeError('hardening build failed')
        test_env=os.environ.copy()
        test_env['PATH']=os.pathsep.join(base.get('lib_dirs',[])+[test_env.get('PATH','')])
        with (HOME/'hardening-tests.txt').open('x') as log:
            for name in ['poly_exchange_test','poly_exchange_gpu_test','poly_no_disk_test','glm_conversations_test']:
                checked=subprocess.run([str(ROOT/'build-poly-cuda12'/(name+'.exe'))],env=test_env,stdout=log,stderr=subprocess.STDOUT)
                if checked.returncode:raise RuntimeError('hardening test failed: '+name)
    work=HOME/(label+'-work');work.mkdir(exist_ok=False)
    seed=work/'expert-profile.txt'
    public_seed=ROOT/('data/expert-profile-glm.txt' if not label.startswith('deepseek') else 'data/expert-profile-dsv4.txt')
    if public_seed.exists():shutil.copyfile(public_seed,seed)
    else:seed.write_text('') # empty public seed: engine uses uniform initial placement
    args=list(base['args'])
    for flag,value in [('--expert-profile',str(seed)),('--session-dir',str(work/'sessions'))]:
        if flag in args:args[args.index(flag)+1]=value
    if not label.startswith('deepseek'):args+=['--vram-reserve-mib','1024']
    base['test_env']={}
    if not label.startswith('glm-control') and label!='glm-mmap':args+=['--no-disk-expert-exchange']
    if label=='glm-mmap':args.remove('--no-disk')
    limit=re.search(r'max(\d+)',label)
    if limit:args+=['--no-disk-exchange-max',limit.group(1)]
    if 'smoke' in label:
        base['test_env']['STRATA_EXCHANGE_TEST_MODE']='1'
    if 'no-graphs' in label:args+=['--no-graphs','--no-draft','--window','1']
    args+=['--dump-buffers']
    save(HOME/(label+'-setup.json'),dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_ids=ids,initial_seed_sha256=sha(seed),binary_sha256=sha(base['exe'])))
    run=Run(label,base,args,work)
    outcome={}
    try:
        run.ready(base)
        if 'smoke' in label:
            for fault in [-1,1,2,3]:native(run,'EXCHANGE '+str(fault),True)
            native(run,'EXCHANGE')
            for i,n in enumerate([31,32,33],1):run.generate(i,ids[10*i:10*i+n],16)
            run.generate(4,ids,64)
            native(run,'EXCHANGE')
            run.generate(5,ids,32)
            novel=ids[:1]+ids[100:]+ids[1:100]
            if label.startswith('deepseek'):novel=novel*3
            run.generate(6,novel,32)
            run.generate(7,ids,32) # return to the parked public conversation after ownership changed
        else:
            for i in range(1,4):run.generate(i,ids,512)
        outcome['completed']=True
    except Exception as e:
        outcome['error']=str(e);print('FAILED',str(e),flush=True)
    finally:run.close()
    save(HOME/(label+'-outcome.json'),outcome)
    print('FINISHED',label,json.dumps(outcome),flush=True)

if __name__=='__main__':main()
