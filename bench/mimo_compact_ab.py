#!/usr/bin/env python
"""MiMo compact A/B: the same requests through the production stack once plain, once with
--compact-scratch, once with --compact-scratch --compact-graphs on top.

Like the other benches: one script drives the phases (one card), every phase starts the engine
from the config's own args (the daily no-disk + exchange + copy-cache stack) plus its flags, with
its OWN fresh session dir and its OWN copy of the expert profile; the originals are only read.
Both experiment flags are stripped from the config's args first, so a config used day-to-day
with them on cannot hand a flag to the plain phase too.

Per phase, temperature=0 seed=42 through the MiMo chat template (Service.encode_prompt):
  req 1 - an agent-style first turn, cold: the whole prompt read
  req 2 - that turn's answer plus a follow-up: the part before the answer reused, the tail read
  req 3.. - req 2 again, --repeats times: warm agent traffic for decode tok/s and expert hits
The phase's VRAM plan is read from the engine's own stderr: the memory[load] allocation line
(experts/staging/rows/other) and the no-disk placement line naming how many experts stand in
VRAM - the economy of the compact layout is the point of the comparison, not an afterthought.

Byte-identical answers across the phases are NOT expected even though the layouts claim bitwise
parity: a narrower scratch gives the expert planner more VRAM, a different set of experts goes
on the card, and the GPU/CPU sum order may flip a greedy near-tie. Both answers are valid; the
verdict is memory, decode tok/s and expert hit, request by request.

Refuses to start while strata-poly.exe or the Qwen server runs. Writes manifest.json,
command-<phase>.json, protocol-<phase>.txt, results.json, summary.txt, run.log into a fresh
bench/results/<stamp>-mimo-compact-ab/; nothing existing is overwritten.
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

CONFIG = REPO / "strata-mimo-mopd-q2k.json"
EXPERIMENT = ("--compact-scratch", "--compact-graphs")
FIRST_TURN = (
    "Ты — агент, работающий в терминале. Вот фрагмент лога движка:\n\n"
    "strata serve: prompt 6832 tokens = 6371 reused + 461 read in 10106 ms (45.6 tok/s), "
    "288 generated in 21913 ms (13.1 tok/s), drafts accepted 0 of 1\n"
    "strata serve: prompt 7167 tokens = 6831 reused + 336 read in 9974 ms (33.7 tok/s), "
    "427 generated in 33226 ms (12.9 tok/s), drafts accepted 1 of 3\n\n"
    "Объясни по строчкам, что значит каждая цифра в этих строках, почему reuse ускоряет ответ "
    "и что такое drafts accepted. Дай структурированный ответ со списком пунктов.\n"
)
FOLLOW_UP = "Перепиши тот же ответ короче, тезисно, списком из шести пунктов."


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


class Encoder:
    """The model's tokenizer and chat template, driven the way the server drives them."""

    def __init__(self, cfg, say):
        from strata_tokenizer import Tokenizer
        from serve.frontend import ChatTemplate
        from serve.server import Service
        from types import SimpleNamespace
        self.tok = Tokenizer.from_dir(cfg["tokenizer"])
        svc = Service.__new__(Service)
        svc.engine = SimpleNamespace(info={}, batch=0)
        svc.tok = self.tok
        svc.template = ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
        svc.preserve_thinking = cfg.get("preserve_thinking") is True
        svc.effort_end = cfg.get("effort_position", "start") == "end"
        svc._read_tokenizer(self.tok)
        self.svc = svc
        say(f"токенизатор: {cfg['tokenizer']}")

    def encode(self, messages):
        return self.svc.encode_prompt(messages, None, {}, store=False)

    def decode(self, ids):
        return self.tok.decode(ids)


def fresh_args(base_args, work, extra, keep_flags=False):
    """base_args with the session dir and expert profile redirected into `work` and the phase's
    flags appended. Both experiment flags are stripped from the config's args first - unless
    keep_flags, for phases that probe another model's config as it is actually run."""
    args = (list(base_args) if keep_flags
            else [a for a in base_args if a not in EXPERIMENT]) + list(extra)
    work.mkdir(parents=True, exist_ok=False)
    (work / "sessions").mkdir()
    args[args.index("--session-dir") + 1] = str(work / "sessions")
    prof = args[args.index("--expert-profile") + 1]
    shutil.copyfile(prof, work / "expert-profile.txt")     # the user's profile stays untouched
    args[args.index("--expert-profile") + 1] = str(work / "expert-profile.txt")
    return args


def run_phase(name, base_args, extra, enc, prompts, out, say, exe, env, keep_flags=False):
    """One engine start through the fixed sequence; returns (requests, error)."""
    say("")
    say(f"=== фаза {name}: запуск движка (загрузка 2-4 минуты, тишина в этот момент нормальна) ===")
    args = fresh_args(base_args, out / ("work-" + name), extra, keep_flags)
    command = [exe, "--serve"] + args
    with (out / f"command-{name}.json").open("x", encoding="utf8") as f:
        json.dump({"command": command, "exe_sha256": sha(exe)}, f, indent=2)
    stderr_f = open(out / f"engine-{name}.stderr", "w", encoding="utf8")
    t0 = time.monotonic()
    try:
        p = subprocess.Popen(command, cwd=str(REPO), env=env, text=True, bufsize=1,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_f,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError as e:
        say(f"движок не запустился: {e}")
        return [], repr(e)
    lines = queue.Queue()

    def reader():
        for ln in p.stdout:
            lines.put(ln.rstrip())
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    protocol = open(out / f"protocol-{name}.txt", "w", encoding="utf8", buffering=1)

    def line(what, timeout):
        # a live but wedged engine never closes stdout and never answers: each wait has its own
        # deadline, and the raise runs the finally (QUIT, then kill after 120 s)
        end = time.monotonic() + timeout
        while True:
            try:
                ln = lines.get(timeout=2)
            except queue.Empty:
                if p.poll() is not None:
                    raise RuntimeError(f"движок вышел раньше срока (код {p.poll()}, см. engine-{name}.stderr)")
                if time.monotonic() > end:
                    raise RuntimeError(f"нет ответа ({what}) больше {timeout} с при живом процессе - останавливаю фазу")
                continue
            if ln is None:
                raise RuntimeError(f"stdout движка закрылся (код {p.poll()}, см. engine-{name}.stderr)")
            protocol.write(ln + "\n")
            if ln.startswith(("ERR ", "FATAL ", "SERR ")):
                raise RuntimeError(ln)
            return ln

    def generate(number, ids, max_new):
        sent = time.monotonic()
        p.stdin.write(f"GEN {max_new} temperature=0 seed=42 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        toks, stamps = [], []
        while True:
            ln = line(f"DONE запроса {number}", 900)
            if ln.startswith("T "):
                toks.append(int(ln.split()[1]))
                stamps.append(time.monotonic())
            if ln.startswith("DONE "):
                d = ln.split()
                break
        # DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered>
        #      <reused> <experts in VRAM> <experts> <RAM blobs> <file blobs> <file MB> <read> <offloaded>
        r = {"number": number, "prompt_tokens": len(ids), "generated": int(d[1]),
             "prompt_ms": float(d[3]), "decode_ms": float(d[4]), "finish": d[5],
             "drafts_accepted": int(d[6]), "drafts_offered": int(d[7]), "reused": int(d[8]),
             "read": int(d[14]), "wall_s": time.monotonic() - sent, "ids": toks,
             "decode_tps": (int(d[1]) - 1) * 1000 / float(d[4]) if int(d[1]) > 1 and float(d[4]) else None,
             "prompt_tps": int(d[14]) * 1000 / float(d[3]) if float(d[3]) else None,
             "ttft_s": stamps[0] - sent if stamps else None}
        r["draft_accept"] = (r["drafts_accepted"] / r["drafts_offered"]) if r["drafts_offered"] else None
        acc = r["draft_accept"]
        rate = lambda x, spec: "н/д" if x is None else format(x, spec)   # a one-token answer has no decode rate
        say(f"  req {number}: read {r['read']}, reused {r['reused']}, +{r['generated']} ток. | "
            f"decode {rate(r['decode_tps'], '.2f')} ток/с | промпт {rate(r['prompt_tps'], '.0f')} ток/с | "
            f"accept {'н/д' if acc is None else f'{acc:.1%}'} ({r['drafts_accepted']}/{r['drafts_offered']})")
        return r

    requests, error = [], None
    try:
        while not line("READY", 1800).startswith("READY "):
            pass
        say(f"READY за {time.monotonic() - t0:.1f} с")
        requests.append(generate(1, prompts["req1"], prompts["max_new"]))
        # req2 is built once, from the first phase's real answer, and reused verbatim by the rest
        if prompts.get("req2") is None:
            answer = enc.decode(requests[0]["ids"])
            prompts["req2"] = enc.encode([dict(role="user", content=FIRST_TURN),
                                          dict(role="assistant", content=answer),
                                          dict(role="user", content=FOLLOW_UP)])
            say(f"промпт 2 собран: {len(prompts['req2'])} токенов (ответ req1 репликой ассистента)")
        requests.append(generate(2, prompts["req2"], prompts["max_new"]))
        for k in range(prompts["repeats"]):
            requests.append(generate(3 + k, prompts["req2"], prompts["max_new"]))
    except Exception as e:
        error = repr(e)
        say(f"фаза {name} остановлена на ошибке: {error}")
    finally:
        if p.poll() is None:
            try:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
            except OSError:
                pass
        try:
            p.wait(timeout=120)   # a wedged engine must not wedge the bench too
        except subprocess.TimeoutExpired:
            say(f"фаза {name}: движок не вышел за 120 с после QUIT - убиваю процесс")
            p.kill()
            p.wait(timeout=30)
        protocol.close()
        stderr_f.close()
    text = (out / f"engine-{name}.stderr").read_text(encoding="utf8", errors="replace")
    costs = [{k: float(v) for k, v in re.findall(r"([a-z_]+)=([0-9.]+)", ln)}
             for ln in re.findall(r"decode-costs (.*)", text)]
    for req, c in zip(requests, costs):
        if "gpu_experts" in c and "cpu_experts" in c:
            req["expert_hit_rate"] = c["gpu_experts"] / max(1.0, c["gpu_experts"] + c["cpu_experts"])
    # the phase's VRAM plan, in the engine's own words
    mem = {"vram_line": next((ln.strip() for ln in text.splitlines()
                              if "memory[load]" in ln and "VRAM allocations total" in ln), None),
           "ready_line": next((ln.strip() for ln in text.splitlines() if "no-disk ready" in ln), None)}
    if mem["vram_line"]:
        say(f"  [VRAM] {mem['vram_line']}")
    if mem["ready_line"]:
        say(f"  [no-disk] {mem['ready_line']}")
    return requests, error, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(REPO / "build-poly-cuda12" / "strata-poly.exe"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--phases", default="plain,scratch,both",
                    help="any of plain,scratch,both, comma separated")
    ap.add_argument("--kill-check", default="on", choices=["on", "off"])
    ap.add_argument("--keep-flags", default="off", choices=["on", "off"],
                    help="off (default): strip --compact-* from the config's args before every phase, "
                         "so the A/B's plain phase stays plain; on: run the config exactly as it is")
    args = ap.parse_args()

    live = engines_running() if args.kill_check == "on" else []
    if live:
        print("ОСТАНОВЛЕНО: уже работают движки: " + ", ".join(f"{n} (PID {p})" for n, p in live))
        print("A/B нужна вся карта целиком; закрой сервер MiMo (и Qwen) и запусти снова.")
        return 2

    all_phases = {
        "plain": [],
        "scratch": ["--compact-scratch"],
        "both": ["--compact-scratch", "--compact-graphs"],
        # one engine start, --trace-graphs on: the fixed ~5.5 s per new prompt size either
        # shows up as a block of graph-create lines inside it, or it does not - no third answer
        "trace": ["--trace-graphs"],
    }
    phases = [(n, all_phases[n]) for n in args.phases.split(",") if n]
    if any(n not in all_phases for n, _ in phases):
        print("неизвестная фаза; есть plain, scratch, both")
        return 2

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out = REPO / "bench" / "results" / f"{stamp}-mimo-compact-ab"
    out.mkdir(parents=False, exist_ok=False)
    runlog = open(out / "run.log", "w", encoding="utf-8")
    say = Tee(runlog)
    say(f"# MiMo compact A/B: прод-стек без compact-флагов, с --compact-scratch, с обоими;")
    say(f"# {args.gen_tokens} ток. на ответ, {args.repeats} повтора прогрева; фазы: {', '.join(n for n, _ in phases)}")
    say(f"# конфиг {args.config}; у каждой фазы свои session-dir и копия expert-profile (оригиналы не пишутся)")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                            text=True).stdout.strip()
    enc = Encoder(cfg, say)
    req1 = enc.encode([dict(role="user", content=FIRST_TURN)])
    say(f"промпт 1: {len(req1)} токенов")

    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["STRATA_TRACE_DECODE_COSTS"] = "1"
    env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
    exe = str(Path(args.exe).resolve())
    with (out / "manifest.json").open("x", encoding="utf8") as f:
        json.dump({"exe": exe, "exe_sha256": sha(exe), "source_commit": commit,
                   "config": str(args.config), "config_sha256": sha(args.config),
                   "gen_tokens": args.gen_tokens, "repeats": args.repeats,
                   "phases": {n: fl for n, fl in phases}}, f, indent=2)

    results, mems, error = {}, {}, None
    t_all = time.monotonic()
    try:
        # the first phase builds req2 from its real answer; every later phase runs that same prompt
        prompts = {"req1": req1, "req2": None, "max_new": args.gen_tokens, "repeats": args.repeats}
        for name, flags in phases:
            results[name], e, mems[name] = run_phase(name, cfg["args"], flags, enc, prompts, out, say, exe,
                                                      env, args.keep_flags == "on")
            if e:
                raise RuntimeError(e)
    except Exception as e:
        error = repr(e)
        say(f"ОСТАНОВЛЕНО: {error}")
    finally:
        summary = write_summary(results, mems, [n for n, _ in phases], say)
        with (out / "results.json").open("w", encoding="utf8") as f:
            json.dump({"results": results, "memory": mems, "error": error}, f, indent=2, ensure_ascii=False)
        say("")
        say(f"всего {time.monotonic() - t_all:.0f} с; папка: {out}")
        runlog.close()
    if summary:
        (out / "summary.txt").write_text(summary, encoding="utf-8")
    return 0 if error is None else 1


def write_summary(results, mems, order, say):
    """decode tok/s and expert hit per phase on the warm repeats, medians, plus the VRAM plan."""
    def warm(phase, key):
        return [r[key] for r in results.get(phase, []) if r.get("number", 0) >= 3]

    base = order[0]
    tps = {n: median(warm(n, "decode_tps")) for n in order}
    hit = {n: median(warm(n, "expert_hit_rate")) for n in order}
    acc = {n: median(warm(n, "draft_accept")) for n in order}
    lines = ["", "=== ИТОГ: тёплые повторы (req 3+), медианы ==="]
    for n in order:
        if tps[n] is None:
            continue
        d = f"   ({100 * (tps[n] / tps[base] - 1):+.1f}% к {base})" if n != base and tps[base] else ""
        h = f" | hit {hit[n]:.1%}" if hit.get(n) is not None else ""
        a = f" | accept {acc[n]:.1%}" if acc.get(n) is not None else ""
        lines.append(f"{n:8s}: decode {tps[n]:.2f} ток/с{d}{h}{a}")
    for n in order:
        m = mems.get(n) or {}
        if m.get("ready_line"):
            lines.append(f"{n:8s} VRAM: {m['ready_line']}")
        elif m.get("vram_line"):
            lines.append(f"{n:8s} VRAM: {m['vram_line']}")
    for ln in lines:
        say(ln)
    return "\n".join(lines) + "\n"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


if __name__ == "__main__":
    sys.exit(main())
