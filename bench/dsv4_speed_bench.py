#!/usr/bin/env python
"""The DeepSeek V4 speed run of 2026-10-30 (2026-10-30-poly-upstream-speed-183438), replayed.

Starts strata-poly.exe --serve with the exact arguments of that run (its manifest.json's command,
--stage-mib 2848 - NOT today's staging-size check with 256), the same public inputs
(bench/results/2026-10-21-deepseek-dma-cost/public-inputs.json, sha 7f5bbf0c..., two 50000-token
prompts) and the same expert-profile seed (sha 1d1cb63f...), then replays the four-request
sequence long-A 50000/256, long-B 50000/256, restore-A 50000/32, short 1000/128 with
`GEN <n> temperature=0 seed=42 <ids>`, exactly as tools/bench_poly_transfer_audit.py does.

The short 1000-token input is rebuilt the same way (the config's tokenizer through
serve.server.Service.encode_prompt); it is not stored in the 2026-10-30 folder.

Writes manifest.json / command.json / ready.json / protocol.txt / request-1..4.json /
results.json / samples.jsonl / engine.stderr / run.log / public-short-input.json into a fresh
bench/results/<stamp>-poly-dsv4-speed/ folder.  Nothing existing is ever overwritten; the script
also refuses a folder that already has combined output, like the original does.

Refuses to start while strata-poly.exe or the Qwen server (strata.exe --serve) runs: the engine
needs the whole card and the machine's RAM.  `--kill-check off` skips that.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
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

REF_DIR = REPO / "bench" / "results" / "2026-10-30-poly-upstream-speed-183438"
INPUTS = REPO / "bench" / "results" / "2026-10-21-deepseek-dma-cost" / "public-inputs.json"
SEED = REPO / "bench" / "inputs" / "dsv4-expert-profile-seed-1d1cb63f.txt"
INPUT_SHA = "7f5bbf0c8ff198b7ceb15dd25824913b674fd389e5bb8f1696f4beb79ef56f53"
SEED_SHA = "1d1cb63f70f30e833063d97855a8f3c3a52d0dd125f858b2e0417a49c0c186e3"
REF_DECODE = {1: 15.38, 2: 16.76, 3: 18.89, 4: 18.2}   # the 2026-10-30 run, ток/с
REF_PROMPT = {1: 665.9, 2: 650.8}                      # the same, промпт

# the 2026-10-30 manifest's command, --serve and the exe removed; the two work paths are
# replaced with the new run's, like the original runner did
ARGS = ["--compact-scratch", "--compact-graphs", "--no-disk", "--coalesce-expert-dma",
        "--rows-vram", "--read-chunk", "32768", "--prompt-batch", "4096",
        "--stage-mib", "2848",
        "--model", "L:\\AI_pictures_generate\\text-generation-webui2\\user_data\\models\\DeepSeek-V4-Flash-0731-Q3_K_M-custom.gguf",
        "--max-context", "131072", "--expert-profile", "PLACEHOLDER",
        "--conversation-cache-mib", "0", "--conversation-cache-no-branch", "--prompt-cache", "6",
        "--conversation-cache-min-free-mib", "512", "--session-min-free-mib", "4096",
        "--session-dir", "PLACEHOLDER", "--session-max-mib", "16384",
        "--vram-reserve-mib", "256", "--no-disk-vram-margin-mib", "256",
        "--expert-swaps", "64", "--dump-buffers"]


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
                found.append(("strata-poly.exe", p.info["pid"]))
            elif name == "strata.exe" and "--serve" in (p.info["cmdline"] or []):
                found.append(("strata.exe --serve (Qwen)", p.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def short_input(config, data, say):
    """The same 1000-token prompt of tools/bench_poly_transfer_audit.py, rebuilt here."""
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
    public = tok.encode(data["messages"][0][0]["content"])
    text = tok.decode(public[:800]) + "\nExplain the placement of experts in detail.\nNotes:"
    padding = 100
    for _ in range(12):
        ids = svc.encode_prompt([dict(role="user", content=text + " x" * padding)], None, {}, store=False)
        if len(ids) == 1000:
            break
        padding += 1000 - len(ids)
    assert len(ids) == 1000, ("short input tokens", len(ids))
    assert ids[0] == tok.special_ids["tokenizer.ggml.bos_token_id"]
    assert ids[-1] == tok.special_tokens["<think>"]
    say(f"короткий вход: 1000 токенов собран текущим encode_prompt (в папке 30.10 он не хранился)")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(REPO / "build-poly-cuda12" / "strata-poly.exe"))
    ap.add_argument("--inputs", default=str(INPUTS))
    ap.add_argument("--profile-seed", default=str(SEED))
    ap.add_argument("--kill-check", default="on", choices=["on", "off"])
    args = ap.parse_args()

    live = engines_running() if args.kill_check == "on" else []
    if live:
        print("ОСТАНОВЛЕНО: уже работают движки: " + ", ".join(f"{n} (PID {p})" for n, p in live))
        print("DeepSeek-движку нужны вся карта и вся RAM, второй движок не влезет.")
        print("Закрой окно сервера ассистента (Qwen) и запусти батник снова.")
        return 2

    for path, want, what in ((args.inputs, INPUT_SHA, "вход"), (args.profile_seed, SEED_SHA, "seed профиля")):
        got = sha(path)
        if got != want:
            print(f"ОСТАНОВЛЕНО: {what} {path} имеет sha {got[:12]}…, а нужен {want[:12]}… (прогон 30.10).")
            print("Сравнивать будет не с чем; батник ничего не стал запускать.")
            return 2

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out = REPO / "bench" / "results" / f"{stamp}-poly-dsv4-speed"
    out.mkdir(parents=True, exist_ok=False)
    (out / "work" / "sessions").mkdir(parents=True)
    runlog = open(out / "run.log", "w", encoding="utf-8")
    say = Tee(runlog)

    say(f"# DeepSeek V4 speed run, повтор прогона {REF_DIR.name} (параметры 30.10, --stage-mib 2848)")
    say(f"# сегодня проверяли размер staging (--stage-mib 256) - здесь ЕГО НЕТ, параметры 30-го числа")

    config = json.loads((REPO / "strata-dsv4-q3_k_m.json").read_text(encoding="utf-8-sig"))
    data = json.loads(Path(args.inputs).read_text(encoding="utf8"))
    prompts = data["ids"]
    assert len(prompts) == 2 and all(len(x) == 50000 for x in prompts)
    if data.get("server_encoder_sha256") != sha(REPO / "serve" / "server.py"):
        say("заметка: вход закодирован прежним serve/server.py (sha в файле входа). Для двух длинных")
        say("входов это не важно - движку идут сырые id; короткий вход пересобирается текущим кодом.")

    seed = out / "work" / "expert-profile.txt"
    shutil.copyfile(args.profile_seed, seed)
    shutil.copyfile(args.inputs, out / "public-inputs.json")
    launch = list(ARGS)
    launch[launch.index("--expert-profile") + 1] = str(seed)
    launch[launch.index("--session-dir") + 1] = str(out / "work" / "sessions")

    say("сборка короткого входа 1000 токенов (токенизатор) ...")
    short = short_input(config, data, say)
    with (out / "public-short-input.json").open("x", encoding="utf8") as f:
        json.dump({"ids": short, "tokens": len(short),
                   "method": "actual Service.encode_prompt, complete template"}, f)
    gc.collect()

    exe = str(Path(args.exe).resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["STRATA_TRACE_DECODE_COSTS"] = "1"
    env["STRATA_TRACE_DRAFT_POLICY"] = "1"
    env["STRATA_TRACE_TRANSFERS"] = "1"
    env["STRATA_TRACE_STAGING"] = "1"
    env["PATH"] = os.pathsep.join(config.get("lib_dirs", []) + [env.get("PATH", "")])

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                            text=True).stdout.strip()
    with (out / "manifest.json").open("x", encoding="utf8") as f:
        json.dump({"command": [exe, "--serve"] + launch, "executable_sha256": sha(exe),
                   "input_sha256": sha(args.inputs), "profile_seed_sha256": sha(args.profile_seed),
                   "source_commit": commit,
                   "sequence": ["long-A 50000/256", "long-B 50000/256", "restore-A 50000/32",
                                "short 1000/128"],
                   "replays": REF_DIR.name,
                   "repeat_policy": "one sequence, no automatic retry"}, f, indent=2)

    command = [exe, "--serve"] + launch
    say(f"exe: {exe}")
    say(f"exe sha: {sha(exe)[:12]}…  коммит: {commit[:8] if commit else '?'}")

    stop_sampling = threading.Event()
    started = time.monotonic()
    phase = {"name": "load"}
    samples = open(out / "samples.jsonl", "w", encoding="utf-8")
    stderr_f = open(out / "engine.stderr", "w", encoding="utf8")
    say("запуск движка (загрузка ~2 минуты) ...")
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
    import queue
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
                m = proc.memory_info()
                v = psutil.virtual_memory()
                row = {"t": time.monotonic(), "elapsed": time.monotonic() - started,
                       "phase": phase["name"], "cpu_percent": proc.cpu_percent(None),
                       "memory": m._asdict(), "threads": proc.num_threads(),
                       "io": proc.io_counters()._asdict(), "cpu_times": proc.cpu_times()._asdict(),
                       "system_ram": {"total": v.total, "available": v.available, "used": v.used},
                       "system_cpu_percent_per_core": psutil.cpu_percent(None, percpu=True),
                       "system_disk": psutil.disk_io_counters()._asdict(),
                       "process_gpu": gpu_process_memory(p.pid)}
                if gpu.ok():
                    row["gpu"] = gpu.read()
                    row["gpu"]["sm_mhz"] = gpu._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(1))
                    row["gpu"]["memory_mhz"] = gpu._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(2))
                samples.write(json.dumps(row) + "\n")
                samples.flush()
            except (psutil.Error, OSError):
                return
            stop_sampling.wait(0.5)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    protocol = open(out / "protocol.txt", "w", encoding="utf8", buffering=1)
    last_progress = [time.monotonic()]

    def line():
        while True:
            try:
                ln = lines.get(timeout=1)
            except queue.Empty:
                if time.monotonic() - last_progress[0] > 10:
                    last_progress[0] = time.monotonic()
                    say(f"фаза {phase['name']}: движок молчит, прошло {time.monotonic() - started:.0f} с ...")
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
        io_before = proc.io_counters()._asdict()
        cpu_before = proc.cpu_times()._asdict()
        cmd = f"GEN {max_new} temperature=0 seed=42 " + ",".join(map(str, ids)) + "\n"
        p.stdin.write(cmd)
        p.stdin.flush()
        tokens, stamps, resume, d = [], [], None, None
        while True:
            ln = line()
            if ln.startswith("RESUME "):
                resume = int(ln.split()[1])
            if ln.startswith("T "):
                phase["name"] = f"request-{number}/decode"
                tokens.append(int(ln.split()[1]))
                stamps.append(time.monotonic())
            if ln.startswith("DONE "):
                d = ln.split()
                break
        assert len(tokens) == int(d[1]) and resume == int(d[8]), \
            f"request {number}: DONE vs tokens/RESUME mismatch"
        result = {"number": number, "prompt_tokens": len(ids), "generated": len(tokens),
                  "requested": max_new, "ids": tokens, "resume": resume,
                  "prompt_ms": float(d[3]), "decode_ms": float(d[4]), "finish": d[5],
                  "drafts_accepted": int(d[6]), "drafts_offered": int(d[7]),
                  "raw_done_gpu": int(d[9]), "raw_done_lookups": int(d[10]), "read": int(d[14]),
                  "elapsed_s": time.monotonic() - sent,
                  "ttft_s": stamps[0] - sent if stamps else None,
                  "decode_tps": (len(tokens) - 1) * 1000 / float(d[4]) if len(tokens) > 1 else None,
                  "prompt_tps": int(d[14]) * 1000 / float(d[3]) if float(d[3]) else None,
                  "token_times": [t - stamps[0] for t in stamps],
                  "stdin_bytes": len(cmd.encode()) + 1,
                  "io_before": io_before, "io_after": proc.io_counters()._asdict(),
                  "cpu_before": cpu_before, "cpu_after": proc.cpu_times()._asdict(),
                  "process_gpu_after": gpu_process_memory(p.pid)}
        with (out / f"request-{number}.json").open("x", encoding="utf8") as f:
            json.dump(result, f, indent=2)
        say(f"request-{number}: промпт {result['prompt_tokens']} принято {result['resume']} "
            f"сгенерировано {result['generated']} decode {result['decode_tps']:.2f} ток/с "
            f"(30.10: {REF_DECODE[number]})")
        return result

    requests, error = [], None
    try:
        while not line().startswith("READY "):
            pass
        load_s = time.monotonic() - started
        phase["name"] = "ready"
        say(f"READY за {load_s:.1f} с")
        with (out / "ready.json").open("x", encoding="utf8") as f:
            json.dump({"load_s": load_s, "memory": proc.memory_info()._asdict(),
                       "process_gpu": gpu_process_memory(p.pid)}, f, indent=2)
        requests.append(generate(1, prompts[0], 256))
        assert requests[-1]["read"] == 50000 and requests[-1]["resume"] == 0
        requests.append(generate(2, prompts[1], 256))
        assert requests[-1]["read"] == 50000 and requests[-1]["resume"] == 0
        requests.append(generate(3, prompts[0], 32))
        requests.append(generate(4, short, 128))
    except Exception as e:
        error = repr(e)
        say(f"ПРОГОН ОСТАНОВЛЕН на ошибке (без повтора): {error}")
    finally:
        phase["name"] = "shutdown"
        say("остановка движка (QUIT, ждём корректное завершение) ...")
        if p.poll() is None:
            try:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
            except OSError:
                pass
        p.wait()  # graceful only, like the original: never kill a long shutdown
        stop_sampling.set()
        th.join()
        samples.close()
        protocol.close()

    text = (out / "engine.stderr").read_text(encoding="utf8", errors="replace")
    import re
    traces = [{k: float(v) for k, v in (x.split("=") for x in ln.split())}
              for ln in re.findall(r"strata-poly: decode-costs (.*)", text)]
    for req, costs in zip(requests, traces):
        req["decode_costs"] = costs
        req["decode_hit_rate"] = costs["gpu_experts"] / max(1, costs["gpu_experts"] + costs["cpu_experts"])
    with (out / "results.json").open("x", encoding="utf8") as f:
        json.dump({"command": command, "load_s": load_s if requests else None,
                   "requests": requests, "exit": p.returncode,
                   "error": error}, f, indent=2)
    # сессионные дампы (work/sessions, ~350 MiB на файл при 50K токенов): числа уже в
    # results.json, KV-состояние воспроизводится тем же промптом - дамп не результат.
    dumps = [f for f in (out / "work" / "sessions").glob("*") if f.is_file()]
    freed = sum(f.stat().st_size for f in dumps)
    kept = 0
    for f in dumps:
        try:
            f.unlink()
        except OSError:
            kept += 1
    say(f"дампы сессий: удалено {len(dumps) - kept} из {len(dumps)}, {freed / 2**30:.1f} ГиБ"
        + (f", {kept} не тронуто" if kept else ""))

    say("")
    say(f"загрузка движка      : {load_s:.1f} с" if requests else f"выход движка: {p.returncode}")
    for req in requests:
        ref = REF_DECODE[req["number"]]
        mark = ""
        if req["decode_tps"]:
            mark = f" ({100 * req['decode_tps'] / ref - 100:+.1f}% к 30.10)"
        # prompt_tps lives in results.json but never reached the screen; on a restored request it counts
        # only the tokens actually read, so the reused count rides along or the number reads as a failure
        if req["prompt_ms"]:
            reused = f", reused {req['resume']}" if req.get("resume") else ""
            say(f"  request-{req['number']} prompt : {req['prompt_tps']:6.1f} ток/с "
                f"({req['prompt_ms'] / 1000:.1f} с{reused})")
        say(f"  request-{req['number']} decode  : {req['decode_tps']:.2f} ток/с{mark}")
    say(f"выход движка         : {p.returncode}")
    say(f"папка: {out}")
    runlog.close()
    return 0 if (error is None and p.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
