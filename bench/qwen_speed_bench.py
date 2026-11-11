#!/usr/bin/env python
"""The Qwen memory/speed run of 2026-10-30 (2026-10-30-qwen-memory-ab-192220), one mode only: B.

Starts strata.exe --serve with the exact arguments of that run (bench/results/.../manifest.json's
config_args, the expert profile copied into the new work folder), sends the same public synthetic
input (input.json, 32675 tokens) as one `GEN 512 <ids>` line, times every token, and writes
command.json / ready.json / protocol.txt / request-1.json / results.json / samples.jsonl /
run.log into a fresh bench/results/<stamp>-qwen-speed-b/ folder. Nothing is ever overwritten.

The environment is that run's mode-1: STRATA_EMB_REUSE_ACCOUNT=1, STRATA_TRACE=1,
STRATA_TRACE_DECODE_COSTS=1, STRATA_TRACE_DRAFT_POLICY=1 and the config's three STRATA_ vars.

The engine needs the whole card and ~105 GiB of RAM: the script refuses to start while another
strata.exe --serve is running (the assistant's own server), so a run cannot OOM the machine.
`--kill-check off` skips that. `--emb 0` reproduces the run's mode A instead.

Compared to 2026-10-30: process_gpu.used_mib comes from nvidia-smi; the driver's WDDM counters
(asked / spilled / shared) are not collected here. Everything else matches.
"""
import argparse
import hashlib
import json
import os
import psutil
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "bench" / "results" / "2026-10-30-qwen-memory-ab-192220"


class Tee:
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_running():
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] != me and p.info["name"] == "strata.exe" and "--serve" in (p.info["cmdline"] or []):
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


GPU_LOCK = threading.Lock()
_gpu_cache = {"t": 0.0, "mib": None}


def gpu_used_mib(pid):
    """The process's VRAM per nvidia-smi, at most once every 2 s (the call itself costs ~100 ms)."""
    with GPU_LOCK:
        now = time.monotonic()
        if now - _gpu_cache["t"] < 2.0:
            return _gpu_cache["mib"]
        _gpu_cache["t"] = now
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) >= 2 and f[0].isdigit() and int(f[0]) == pid:
                    _gpu_cache["mib"] = int(f[1])
                    return _gpu_cache["mib"]
        except Exception:
            pass
        _gpu_cache["mib"] = None
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(REPO / "engine" / "strata.exe"))
    ap.add_argument("--emb", default="1", help="STRATA_EMB_REUSE_ACCOUNT for the run (1 = mode B)")
    ap.add_argument("--kill-check", default="on", choices=["on", "off"])
    ap.add_argument("--drop", default="", help="comma-separated flags to remove from the manifest "
                      "command with their values (an engine without them); pass it in BOTH phases "
                      "of an A/B so both engines get the same request")
    ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                    help="replace (or add) a flag's value in the manifest command, e.g. "
                         "--set=--prefill=auto:16384; repeatable")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                    help="add an environment variable for the engine on top of the run's fixed set, "
                         "e.g. --env=STRATA_NV_CHUNK=32; repeatable")
    ap.add_argument("--label", default="", help="suffix for the results folder, e.g. our-16k")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the folder, the engine command and command.json, then stop: no "
                         "engine, no requests, no GPU. The results go to bench/results/dry/.")
    args = ap.parse_args()

    other = engine_running() if (args.kill_check == "on" and not args.dry_run) else None
    if other:
        print(f"ОСТАНОВЛЕНО: strata.exe --serve уже работает (PID {other}) - сервер ассистента занимает")
        print("карту и ~105 ГиБ RAM, второй движок не влезет. Закрой окно сервера и запусти батник снова.")
        return 2

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    mode = "b" if args.emb == "1" else "a"
    out = REPO / "bench" / "results" / ("dry" if args.dry_run else "") / (
        f"{stamp}-qwen-speed-{mode}" + (f"-{args.label}" if args.label else ""))
    out.mkdir(parents=True, exist_ok=False)
    runlog = open(out / "run.log", "w", encoding="utf-8")
    say = Tee(runlog)

    say(f"# Qwen speed run, mode {mode.upper()} (STRATA_EMB_REUSE_ACCOUNT={args.emb})")
    say(f"# сравнение с прогоном 30.10: {REF_DIR.name}")

    manifest = json.loads((REF_DIR / "manifest.json").read_text(encoding="utf-8"))
    cmd = list(manifest["config_args"])
    for name in (d.strip() for d in args.drop.split(",") if d.strip()):
        while name in cmd:
            i = cmd.index(name)
            j = i + 2 if i + 1 < len(cmd) and not cmd[i + 1].startswith("--") else i + 1
            say(f"# --drop: убрано {' '.join(cmd[i:j])}")
            del cmd[i:j]
    for pair in args.set:
        name, _, value = pair.partition("=")
        if value == "":                       # булев флаг: добавить как есть, если ещё нет
            if name not in cmd:
                cmd.append(name)
        elif name in cmd:
            cmd[cmd.index(name) + 1] = value
        else:
            cmd += [name, value]
        say(f"# --set: {name}{' ' + value if value else ''}")
    if "--serve" not in cmd:                 # the manifest's config_args start with --pack; the 30.10
        cmd.insert(0, "--serve")             # command.json shows the engine was started with --serve
    cmd.insert(0, str(Path(args.exe).resolve()))
    cmd[cmd.index("--expert-profile") + 1] = str(out / "profile.bin")
    shutil.copyfile(REF_DIR / "initial-profile.bin", out / "profile.bin")

    input_ids = json.loads((REF_DIR / "input.json").read_text(encoding="utf-8"))
    ids = input_ids["ids"]

    env = dict(os.environ)
    env.update({
        "STRATA_TRACE_DECODE_COSTS": "1",
        "STRATA_TRACE_DRAFT_POLICY": "1",
        "STRATA_MMQ_KQ": "1",
        "STRATA_HC_Q8": "1",
        "STRATA_ROUTE_TAIL_SKIP": "0",
        "STRATA_EMB_REUSE_ACCOUNT": args.emb,
        "STRATA_TRACE": "1",
    })
    for pair in args.env:
        name, _, value = pair.partition("=")
        env[name] = value
        say(f"# --env: {name}={value}")

    (out / "command.json").write_text(json.dumps(
        {"command": cmd, "exe_sha256": sha256(args.exe),
         "environment": {k: v for k, v in env.items() if k.startswith("STRATA_")}},
        indent=2), encoding="utf-8")
    (out / "request-1.json").write_text(json.dumps(
        {"number": 1, "prompt_tokens": len(ids), "generated": 512, "requested": 512, "ids": ids}),
        encoding="utf-8")
    say(f"exe: {cmd[0]}")
    say(f"exe sha: {json.loads((out / 'command.json').read_text())['exe_sha256'][:16]}…")

    if args.dry_run:
        say("# --dry-run: команда собрана и записана, движок не запускался")
        runlog.close()
        return 0

    stop_sampling = threading.Event()
    samples = open(out / "samples.jsonl", "w", encoding="utf-8")
    phase = {"name": "spawn", "t0": time.monotonic()}

    def sampler():
        proc = None
        while not stop_sampling.is_set():
            try:
                if proc is None:
                    proc = psutil.Process(proc_pid)
                m = proc.memory_info()._asdict()
                samples.write(json.dumps({
                    "t": time.perf_counter(), "elapsed": time.monotonic() - phase["t0"],
                    "phase": phase["name"], "cpu_percent": proc.cpu_percent(),
                    "memory": m, "gpu_used_mib": gpu_used_mib(proc_pid)}) + "\n")
                samples.flush()
            except (psutil.NoSuchProcess, OSError):
                pass
            stop_sampling.wait(0.5)

    say("запуск движка (загрузка ~2 минуты) ...")
    t0 = time.monotonic()
    stderr_f = open(out / "engine.stderr", "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr_f, cwd=str(REPO), env=env, text=True,
                                encoding="utf-8", bufsize=1)
    except OSError as e:
        say(f"ОСТАНОВЛЕНО: движок не запустился: {e}")
        runlog.close()
        return 1
    proc_pid = proc.pid
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    proto = open(out / "protocol.txt", "w", encoding="utf-8")

    def logline(line):
        proto.write(f"{time.monotonic() - phase['t0']:12.3f}  {line}\n")
        proto.flush()

    ready = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        logline(line)
        if line.startswith("READY"):
            ready = line
            break
        if line.startswith("ERR"):
            say(f"движок отказался стартовать: {line}")
            stop_sampling.set()
            return 1
    if not ready:
        stop_sampling.set()
        th.join(timeout=2)
        samples.close()
        try:
            exit_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait()
        say(f"ОСТАНОВЛЕНО: движок закрыл stdout, не сказав READY (код {exit_code}).")
        say("первые строки engine.stderr:")
        for ln in (out / "engine.stderr").read_text(encoding="utf-8", errors="replace").splitlines()[:6]:
            say("  " + ln)
        runlog.close()
        return 1
    load_s = time.monotonic() - t0
    phase["name"] = "wait_prompt"
    say(f"READY за {load_s:.1f} с")
    mem = psutil.Process(proc_pid).memory_info()._asdict()
    (out / "ready.json").write_text(json.dumps(
        {"load_s": load_s, "memory": mem, "ready_line": ready}, indent=2), encoding="utf-8")

    gen = f"GEN 512 {','.join(str(int(t)) for t in ids)}\n"
    phase["name"] = "prompt"
    t_sent = time.monotonic()
    proc.stdin.write(gen)
    proc.stdin.flush()

    token_times = []
    done_fields = None
    resume = 0
    for line in proc.stdout:
        line = line.rstrip("\n")
        logline(line)
        if line.startswith("T "):
            now = time.monotonic()
            if not token_times:
                phase["name"] = "decode"
            token_times.append(now - t_sent)
        elif line.startswith("RESUME"):
            f = line.split()
            resume = int(f[1]) if len(f) > 1 and f[1].isdigit() else 0
        elif line.startswith("DONE"):
            done_fields = line.split()
            break
        elif line.startswith("ERR"):
            say(f"движок отказал на запросе: {line}")
            break
    t_done = time.monotonic()
    stop_sampling.set()
    th.join(timeout=2)
    samples.close()

    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    try:
        exit_code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        exit_code = proc.wait()
    phase["name"] = "shutdown"

    def num(i, cast=float):
        try:
            return cast(done_fields[i])
        except (TypeError, IndexError, ValueError):
            return None

    gen_n = len(token_times)
    ttft = token_times[0] if token_times else None
    prompt_ms = ttft * 1000 if ttft is not None else None
    decode_ms = (t_done - t_sent - ttft) * 1000 if ttft is not None else None
    request = {
        "number": 1, "prompt_tokens": len(ids), "generated": gen_n, "requested": 512,
        "resume": resume,
        "prompt_ms": prompt_ms, "decode_ms": decode_ms,
        "finish": done_fields[5] if done_fields and len(done_fields) > 5 else None,
        "drafts_accepted": num(6, int), "drafts_offered": num(7, int),
        "reused": num(8, int),
        "raw_done_gpu": num(9, int), "raw_done_lookups": num(10, int),
        "read": len(ids),
        "elapsed_s": t_done - t_sent, "ttft_s": ttft,
        "decode_tps": (gen_n - 1) * 1000.0 / decode_ms if decode_ms else None,
        "prompt_tps": len(ids) * 1000.0 / prompt_ms if prompt_ms else None,
        "engine_done_line": " ".join(done_fields) if done_fields else None,
        "token_times": [round(x - (ttft or 0.0), 7) for x in token_times],
    }
    (out / "results.json").write_text(json.dumps(
        {"command": cmd, "load_s": load_s, "requests": [request], "exit": exit_code}, indent=2),
        encoding="utf-8")

    say(f"загрузка           : {load_s:.1f} с")
    say(f"промпт {len(ids)} токенов : {prompt_ms / 1000:.1f} с = {request['prompt_tps']:.1f} ток/с"
        if prompt_ms else "промпт: нет данных")
    if decode_ms:
        say(f"генерация {gen_n} токенов: {decode_ms / 1000:.1f} с = {request['decode_tps']:.2f} ток/с")
    say(f"MTP принято        : {request['drafts_accepted']} из {request['drafts_offered']}")
    say(f"хит-рейт кеша      : "
        f"{100.0 * request['raw_done_gpu'] / request['raw_done_lookups']:.1f}%"
        if request["raw_done_gpu"] and request["raw_done_lookups"] else "хит-рейт: нет данных")
    say(f"выход движка       : {exit_code}")
    say("для сравнения, прогон 30.10 (тот же вход): mode A 40.56 ток/с, mode B 46.05 ток/с, загрузка 133.0/115.7 с")
    say(f"папка: {out}")
    runlog.close()
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
