"""One model load, one integrated public-input diagnostic sequence. Never reruns a result directory.

The protocol/sampler is derived from the existing public no-disk benchmark runner.
All output creation is exclusive; existing experiments and other engines are left alone.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from serve.telemetry import gpu_reader, gpu_process_memory
HOME = None

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, data):
    with Path(p).open('x', encoding='utf8') as f: json.dump(data, f, indent=2)

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


def short_input(config, data):
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import Service
    from types import SimpleNamespace
    tok = Tokenizer.from_dir(config['tokenizer'])
    svc = Service.__new__(Service)
    svc.engine = SimpleNamespace(info={}, batch=0)
    svc.tok = tok
    svc.template = ChatTemplate(Path(config['tokenizer'])/'chat_template.jinja')
    svc.preserve_thinking = config.get('preserve_thinking') is True
    svc.effort_end = config.get('effort_position','start') == 'end'
    svc._read_tokenizer(tok)
    public = tok.encode(data['messages'][0][0]['content'])
    text = tok.decode(public[:800]) + '\nExplain the placement of experts in detail.\nNotes:'
    padding = 100
    for _ in range(12):
        ids = svc.encode_prompt([dict(role='user',content=text+' x'*padding)],None,{},store=False)
        if len(ids)==1000: break
        padding += 1000-len(ids)
    assert len(ids)==1000, ('short input tokens',len(ids))
    assert ids[0] == tok.special_ids['tokenizer.ggml.bos_token_id']
    assert ids[-1] == tok.special_tokens['<think>']
    return ids


def main():
    global HOME
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--profile-seed',type=Path,required=True)
    ap.add_argument('--run',action='store_true',help='explicitly authorize the single model load')
    args = ap.parse_args()
    if not args.run: raise SystemExit('No model started; use --run for the integrated diagnostic sequence')
    if any((p.info['name'] or '').lower()=='strata-poly.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Another engine is running; leaving it untouched')
    HOME = args.output.resolve()
    HOME.mkdir(exist_ok=True,parents=True)
    if (HOME/'combined').exists() or (HOME/'manifest.json').exists():
        raise RuntimeError('This experiment already exists; refusing to rerun or overwrite')
    config = json.loads((ROOT/'strata-dsv4-q3_k_m.json').read_text(encoding='utf-8-sig'))
    data = json.loads(args.inputs.read_text(encoding='utf8'))
    prompts = data['ids']
    assert len(prompts)==2 and all(len(x)==50000 for x in prompts)
    assert data['server_encoder_sha256']==sha(ROOT/'serve/server.py')
    assert data['tokenizer_template_sha256']==sha(Path(config['tokenizer'])/'chat_template.jinja')
    short = short_input(config,data)
    save(HOME/'public-short-input.json',dict(ids=short,tokens=len(short),method='actual Service.encode_prompt, complete template'))
    gc.collect()
    work = HOME/'work'
    work.mkdir()
    seed = work/'expert-profile.txt'
    with seed.open('xb') as f, args.profile_seed.open('rb') as src: shutil.copyfileobj(src,f)
    launch = list(config['args'])
    for flag,value in [('--expert-profile',str(seed)),('--session-dir',str(work/'sessions'))]:
        launch[launch.index(flag)+1]=value
    if '--dump-buffers' not in launch: launch += ['--dump-buffers']
    config['test_env']={'STRATA_TRACE_TRANSFERS':'1','STRATA_TRACE_STAGING':'1'}
    save(HOME/'manifest.json',dict(command=[config['exe'],'--serve']+launch,
        executable_sha256=sha(config['exe']),input_sha256=sha(args.inputs),profile_seed_sha256=sha(args.profile_seed),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sequence=['long-A 50000/256','long-B 50000/256','restore-A 50000/32','short 1000/128'],
        diagnostic_only=True,repeat_policy='one sequence, no automatic retry'))
    run=Run('combined',config,launch,work)
    outcome={}
    try:
        run.ready(config)
        run.generate(1,prompts[0],256)
        assert run.requests[-1]['read']==50000 and run.requests[-1]['resume']==0
        run.generate(2,prompts[1],256)
        assert run.requests[-1]['read']==50000 and run.requests[-1]['resume']==0
        run.generate(3,prompts[0],32)
        run.generate(4,short,128)
        outcome['completed']=True
    except Exception as e:
        outcome['error']=str(e)
        print('DIAGNOSTIC FAILED; no retry:',repr(e),flush=True)
    finally:
        run.close()
        save(HOME/'outcome.json',outcome)
    from analyze_poly_transfers import analyze
    analysis=analyze(HOME)
    save(HOME/'transfer-analysis.json',analysis)
    for row in analysis['dma']:print('DMA',json.dumps(row),flush=True)
    print('COMPLETE',json.dumps(outcome),'records',len(analysis['records']),flush=True)

if __name__=='__main__':main()
