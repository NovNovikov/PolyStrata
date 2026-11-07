#!/usr/bin/env python
"""The MiMo big-prompt speed run: bench/dsv4_speed_bench.py's sequence on MiMo's stack.

Same four requests as the DeepSeek test replays (long-A 50000/256, long-B 50000/256,
restore-A 50000/32, short 1000/128, `GEN <n> temperature=0 seed=42 <ids>`), but the inputs are
built with MiMo's own tokenizer - the DeepSeek public-inputs.json holds DeepSeek ids, which mean
nothing to MiMo's vocabulary - from fixed deterministic texts: A and B share nothing but the BOS.
The engine gets raw ids, so no chat template sits inside the long prompts; the short 1000-token
input goes through the same Service.encode_prompt path the DeepSeek test rebuilds.

The launch is strata-mimo-mopd-q2k.json's args EXACTLY as they stand (the stack the user runs
daily, --compact-scratch included); only --session-dir and --expert-profile are redirected into
the run's work folder, and the originals are only read.

Why: small-prompt reads measure 31-44 tok/s on MiMo (2026-11-07 traces, bench/results/*compact-ab);
DeepSeek's October big-prompt run held 591-700 tok/s across 50000 tokens. Whether MiMo's read
path bends at scale or stays flat is the open question, so every PP line lands in request-N.json
under "pp" with its counters (cpu_ms, route_ms, experts=gpu/cpu).

Refuses to start while strata-poly.exe or the Qwen server (strata.exe --serve) runs.
Writes manifest.json / command.json / ready.json / protocol.txt / request-1..4.json / results.json
/ samples.jsonl / engine.stderr / run.log into a fresh bench/results/<stamp>-mimo-speed/ folder.
Nothing existing is overwritten.

Time: load ~3 minutes. If big prompts read at the small-prompt rate the run is ~50 minutes
(two 50000-token prefills at ~30 tok/s); if the batched path kicks in, ~10 minutes total.
"""
import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import psutil  # noqa: E402
from serve.telemetry import gpu_reader, gpu_process_memory  # noqa: E402

CONFIG = REPO / "strata-mimo-mopd-q2k.json"
LONG_TOKENS = 50000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Tee:
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def engines_running():
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            if name == "strata-poly.exe":
                found.append(("strata-poly.exe (MiMo)", p.info["pid"]))
            elif name == "strata.exe" and "--serve" in (p.info["cmdline"] or []):
                found.append(("strata.exe --serve (Qwen)", p.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def long_input(tok, salt, say):
    """50000 raw ids of deterministic text; the salt is in the first line, so two salts
    share nothing but the tokenizer's own head behaviour. Plain encode - no chat template."""
    text = f"Mimo big-prompt transcript {salt}. "
    body = lambda i: f"line {i}: expert {i * 7 % 97} of layer {i % 47} reads {i * 13 % 251} rows\n"
    i, ids = 0, []
    while len(ids) < LONG_TOKENS:
        text += "".join(body(j) for j in range(i, i + 500))
        i += 500
        ids = tok.encode(text)
    ids = ids[:LONG_TOKENS]
    assert len(ids) == LONG_TOKENS, (len(ids),)
    say(f"длинный вход {salt}: {len(ids)} токенов собраны детерминированно (salt в первой строке)")
    return ids


def short_input(config, say):
    """The 1000-token chat prompt, rebuilt through the same encode_prompt path as the DSV4 test."""
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import Service
    from types import SimpleNamespace
    tok = Tokenizer.from_dir(config["tokenizer"])
    svc = Service.__new__(Service)
    svc.engine = SimpleNamespace(info={}, batch=0)
    svc.tok = tok
    svc.template = ChatTemplate(Path(config["tokenizer"]) / "chat_template.jinja")
    svc.preserve_thinking = config.get("preserve_thinking") is True
    svc.effort_end = config.get("effort_position", "start") == "end"
    svc._read_tokenizer(tok)
    text = ("Объясни по строчкам, что значит каждая цифра в строках лога движка про reuse и "
            "drafts accepted, почему reuse ускоряет ответ и что делает MTP. ")
    padding = 100
    ids = None
    for _ in range(20):
        ids = svc.encode_prompt([dict(role="user", content=text + " x" * padding)], None, {}, store=False)
        if len(ids) == 1000:
            break
        padding = max(1, padding + 1000 - len(ids))
    assert ids is not None and len(ids) >= 1000, ("short input too short", None if ids is None else len(ids))
    if len(ids) != 1000:   # the template's tail stays; the filler gives way - a synthetic prefill input
        ids = ids[:999] + [ids[-1]]
        say("короткий вход: точного 1000 паузой не сложилось, срезано по шаблонному хвосту")
    say(f"короткий вход: {len(ids)} токенов текущим encode_prompt")
    return ids


def parse_pp(lines):
    out = []
    for ln in lines:
        if not ln.startswith("PP "):
            continue
        f = ln.split()
        row = {"at": int(f[1]), "total": int(f[2]), "ms": float(f[3]), "tps": float(f[4])}
        for kv in f[5:]:
            k, _, v = kv.partition("=")
            if k in ("cpu_ms", "route_ms"):
                row[k] = float(v)
            elif k == "experts":
                row["experts_gpu"], _, row["experts_cpu"] = v.partition("/")
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(REPO / "build-poly-cuda12" / "strata-poly.exe"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--kill-check", default="on", choices=["on", "off"])
    args = ap.parse_args()

    live = engines_running() if args.kill_check == "on" else []
    if live:
        print("ОСТАНОВЛЕНО: уже работают движки: " + ", ".join(f"{n} (PID {p})" for n, p in live))
        print("Нужны вся карта и вся RAM. Закрой серверы и запусти снова.")
        return 2

    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    launch = list(config["args"])          # as it stands - nothing stripped
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out = REPO / "bench" / "results" / f"{stamp}-mimo-speed"
    out.mkdir(parents=False, exist_ok=False)
    (out / "work" / "sessions").mkdir(parents=True)
    runlog = open(out / "run.log", "w", encoding="utf-8")
    say = Tee(runlog)
    say(f"# MiMo big-prompt speed run (последовательность dsv4_speed_bench на стеке {Path(args.config).name})")
    say("# последовательность: long-A 50000/256, long-B 50000/256, restore-A 50000/32, short 1000/128")

    launch[launch.index("--session-dir") + 1] = str(out / "work" / "sessions")
    prof = launch[launch.index("--expert-profile") + 1]
    shutil.copyfile(prof, out / "work" / "expert-profile.txt")   # the user's profile stays untouched
    launch[launch.index("--expert-profile") + 1] = str(out / "work" / "expert-profile.txt")

    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_dir(config["tokenizer"])
    say("сборка входов (токенизатор) ...")
    long_a = long_input(tok, "A", say)
    long_b = long_input(tok, "B", say)
    shared = next((k for k in range(min(len(long_a), len(long_b))) if long_a[k] != long_b[k]), 0)
    short = short_input(config, say)
    del tok
    with (out / "inputs.json").open("x", encoding="utf8") as f:
        json.dump({"long_a_head_sha256": hashlib.sha256(repr(long_a[:2000]).encode()).hexdigest(),
                   "long_b_head_sha256": hashlib.sha256(repr(long_b[:2000]).encode()).hexdigest(),
                   "long_shared_prefix_tokens": shared,
                   "long_a_tokens": len(long_a), "long_b_tokens": len(long_b),
                   "short_tokens": len(short)}, f)

    exe = str(Path(args.exe).resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["STRATA_TRACE_DECODE_COSTS"] = "1"
    env["PATH"] = os.pathsep.join(config.get("lib_dirs", []) + [env.get("PATH", "")])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                            text=True).stdout.strip()
    command = [exe, "--serve"] + launch
    with (out / "manifest.json").open("x", encoding="utf8") as f:
        json.dump({"command": command, "executable_sha256": sha(exe), "config": str(args.config),
                   "config_sha256": sha(args.config), "source_commit": commit,
                   "sequence": ["long-A 50000/256", "long-B 50000/256", "restore-A 50000/32",
                                "short 1000/128"]}, f, indent=2)

    stop_sampling = threading.Event()
    started = time.monotonic()
    phase = {"name": "load"}
    samples = open(out / "samples.jsonl", "w", encoding="utf-8")
    stderr_f = open(out / "engine.stderr", "w", encoding="utf8")
    say("запуск движка (загрузка ~3 минуты) ...")
    try:
        p = subprocess.Popen(command, cwd=str(REPO), env=env, text=True, bufsize=1,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_f,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError as e:
        say(f"ОСТАНОВЛЕНО: движок не запустился: {e}")
        return 1
    with (out / "command.json").open("x", encoding="utf8") as f:
        json.dump({"command": command, "pid": p.pid, "exe_sha256": sha(exe),
                   "environment": {k: env[k] for k in env if k.startswith("STRATA_")}}, f, indent=2)

    proc = psutil.Process(p.pid)
    gpu = gpu_reader()
    lines = queue.Queue()

    def reader():
        for ln in p.stdout:
            lines.put(ln.rstrip())
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def sampler():
        try:
            proc.cpu_percent(None)
            psutil.cpu_percent(None, percpu=True)
        except psutil.Error:
            pass
        while not stop_sampling.is_set():
            try:
                row = {"t": time.monotonic(), "elapsed": time.monotonic() - started,
                       "phase": phase["name"], "cpu_percent": proc.cpu_percent(None),
                       "memory": proc.memory_info()._asdict(), "threads": proc.num_threads(),
                       "system_ram": psutil.virtual_memory()._asdict(),
                       "process_gpu": gpu_process_memory(p.pid)}
                if gpu.ok():
                    row["gpu"] = gpu.read()
                samples.write(json.dumps(row) + "\n")
                samples.flush()
            except (psutil.Error, OSError):
                return
            stop_sampling.wait(0.5)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    protocol = open(out / "protocol.txt", "w", encoding="utf8", buffering=1)
    last_shown = [time.monotonic()]

    def line():
        while True:
            try:
                ln = lines.get(timeout=2)
            except queue.Empty:
                if p.poll() is not None:
                    raise RuntimeError(f"движок вышел (код {p.poll()}, см. engine.stderr)")
                if time.monotonic() - last_shown[0] > 30:
                    last_shown[0] = time.monotonic()
                    say(f"фаза {phase['name']}: движок молчит, всего прошло {time.monotonic() - started:.0f} с ...")
                continue
            if ln is None:
                raise RuntimeError("engine exited: " + str(p.wait()))
            protocol.write(ln + "\n")
            if ln.startswith(("ERR ", "FATAL ", "SERR ")):
                raise RuntimeError(ln)
            return ln

    def generate(number, ids, max_new):
        phase["name"] = f"request-{number}/prefill"
        sent = time.monotonic()
        p.stdin.write(f"GEN {max_new} temperature=0 seed=42 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        tokens, stamps, resume, pp, d = [], [], None, [], None
        while True:
            ln = line()
            if ln.startswith("RESUME "):
                resume = int(ln.split()[1])
            if ln.startswith("PP "):
                pp.append(ln)
                row = parse_pp([ln])[-1]
                say(f"  req{number} PP {row['at']}/{row['total']} {row['ms']:.0f} мс {row['tps']:.1f} ток/с"
                    + (f" cpu_ms={row.get('cpu_ms', 0):.0f}" if 'cpu_ms' in row else ""))
            if ln.startswith("T "):
                phase["name"] = f"request-{number}/decode"
                tokens.append(int(ln.split()[1]))
                stamps.append(time.monotonic())
            if ln.startswith("DONE "):
                d = ln.split()
                break
        result = {"number": number, "prompt_tokens": len(ids), "generated": len(tokens),
                  "requested": max_new, "ids": tokens, "resume": resume,
                  "prompt_ms": float(d[3]), "decode_ms": float(d[4]), "finish": d[5],
                  "drafts_accepted": int(d[6]), "drafts_offered": int(d[7]),
                  "reused": int(d[8]), "read": int(d[14]),
                  "prompt_tps": int(d[14]) * 1000 / float(d[3]) if float(d[3]) else None,
                  "decode_tps": (len(tokens) - 1) * 1000 / float(d[4]) if len(tokens) > 1 else None,
                  "ttft_s": stamps[0] - sent if stamps else None,
                  "pp": parse_pp(pp)}
        with (out / f"request-{number}.json").open("x", encoding="utf8") as f:
            json.dump(result, f, indent=2)
        rate = "н/д" if not result["prompt_tps"] else f"{result['prompt_tps']:.1f}"
        say(f"request-{number}: prompt {result['prompt_tokens']} (reused {resume}, read {result['read']}) "
            f"= {rate} ток/с | decode {result['decode_tps'] or 0:.2f} ток/с")
        return result

    requests, error, load_s = [], None, None
    try:
        while not line().startswith("READY "):
            pass
        load_s = time.monotonic() - started
        phase["name"] = "ready"
        say(f"READY за {load_s:.1f} с")
        with (out / "ready.json").open("x", encoding="utf8") as f:
            json.dump({"load_s": load_s, "memory": proc.memory_info()._asdict(),
                       "process_gpu": gpu_process_memory(p.pid)}, f, indent=2)
        requests.append(generate(1, long_a, 256))
        requests.append(generate(2, long_b, 256))
        requests.append(generate(3, long_a, 32))
        requests.append(generate(4, short, 128))
    except Exception as e:
        error = repr(e)
        say(f"ПРОГОН ОСТАНОВЛЕН на ошибке: {error}")
    finally:
        phase["name"] = "shutdown"
        if p.poll() is None:
            try:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
            except OSError:
                pass
        try:
            p.wait(timeout=300)   # a long graceful shutdown is not to be killed; 5 min is generous
        except subprocess.TimeoutExpired:
            say("движок не вышел за 300 с после QUIT - убиваю")
            p.kill()
            p.wait(timeout=30)
        stop_sampling.set()
        th.join()
        samples.close()
        protocol.close()
        stderr_f.close()

    text = (out / "engine.stderr").read_text(encoding="utf8", errors="replace")
    costs = [{k: float(v) for k, v in re.findall(r"([a-z_]+)=([0-9.]+)", ln)}
             for ln in re.findall(r"decode-costs (.*)", text)]
    for req, c in zip(requests, costs):
        req["decode_costs"] = c
    with (out / "results.json").open("x", encoding="utf8") as f:
        json.dump({"requests": requests, "load_s": load_s, "exit": p.returncode, "error": error},
                  f, indent=2, ensure_ascii=False)

    say("")
    say(f"загрузка: {load_s:.1f} с" if load_s else f"выход движка: {p.returncode}")
    for req in requests:
        extra = f", read {req['read']} ({req['prompt_ms'] / 1000:.1f} с)" if req["number"] == 3 else ""
        say(f"  request-{req['number']}: prompt {req['prompt_tps'] or 0:7.1f} ток/с, "
            f"decode {req['decode_tps'] or 0:5.2f} ток/с, reused {req['reused']}{extra}")
    say(f"папка: {out}")
    runlog.close()
    return 0 if (error is None and p.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
