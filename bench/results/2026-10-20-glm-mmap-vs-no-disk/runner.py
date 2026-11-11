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
        env = os.environ.copy()
        for name in ['STRATA_FREEZE_EXPERTS', 'STRATA_VERIFY_STAGE_CACHE', 'STRATA_PROFILE_DECODE',
                     'STRATA_REFRESH_DRAFT_COSTS', 'STRATA_ROUTE_TAIL_SKIP']:
            env.pop(name, None)
        env['STRATA_TRACE_DECODE_COSTS'] = '1'
        env['STRATA_TRACE_DRAFT_POLICY'] = '1'
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
                  for flag in ['--model', '--draft-model']}
        opened = {str(Path(f.path).resolve()).lower() for f in self.process.open_files()}
        save(self.home / 'ready.json', dict(load_s=self.load_s, memory=self.process.memory_info()._asdict(),
             process_gpu=gpu_process_memory(self.p.pid), model_or_draft_handles_open=sorted(wanted & opened)))
        print(self.label, 'READY', round(self.load_s, 2), flush=True)

    def generate(self, number, ids):
        label = 'request-' + str(number)
        self.phase = label + '/prefill'
        sent = time.monotonic()
        io_before = self.process.io_counters()._asdict()
        cpu_before = self.process.cpu_times()._asdict()
        command = 'GEN 512 temperature=0 seed=42 ' + ','.join(map(str, ids)) + '\n'
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
        result = dict(number=number, prompt_tokens=len(ids), generated=len(tokens), requested=512,
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


def main():
    config = json.loads((ROOT / 'strata-glm-3.0bit.json').read_text(encoding='utf-8-sig'))
    work = Path(r'L:\Temp\opencode\glm-mmap-vs-no-disk-20261020')
    work.mkdir(parents=True, exist_ok=False)
    ids = prepare(config)
    save(HOME / 'provenance.json', dict(binary_sha256=sha(config['exe']), config_sha256=sha(ROOT/'strata-glm-3.0bit.json'),
         bat_sha256=sha(ROOT/'run-glm-3.0bit.bat'), seed_sha256=sha(ROOT/'data/expert-profile-glm.txt'),
         mode_order=['mmap-control','bat-no-disk'], changed_only_mode_flag=True,
         common_isolation=['public profile copies','fresh session stores','direct engine protocol'],
         common_diagnostics=['--dump-buffers','STRATA_TRACE_DECODE_COSTS','STRATA_TRACE_DRAFT_POLICY'],
         logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False), private_data_read=False))
    outcomes = {}
    for label in ['mmap-control','bat-no-disk']:
        conflicts = [p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='strata-poly.exe']
        if conflicts:
            outcomes[label] = dict(skipped='another engine is running', pids=conflicts)
            break
        case = work / label
        case.mkdir()
        seed = case / 'expert-profile.txt'
        shutil.copyfile(ROOT / 'data/expert-profile-glm.txt', seed)
        args = list(config['args'])
        args[args.index('--expert-profile')+1] = str(seed)
        args[args.index('--session-dir')+1] = str(case / 'sessions')
        if label == 'mmap-control': args.remove('--no-disk')
        args += ['--dump-buffers']
        run = Run(label, config, args, case)
        try:
            run.ready(config)
            for number in range(1, 4): run.generate(number, ids)
            outcomes[label] = dict(completed=True)
        except Exception as e:
            outcomes[label] = dict(error=str(e))
            print(label, 'FAILED', str(e), flush=True)
        finally:
            run.close()
    save(HOME / 'outcomes.json', outcomes)
    print('GLM_COMPARISON_FINISHED', json.dumps(outcomes), flush=True)


if __name__ == '__main__':
    main()
