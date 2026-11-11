#!/usr/bin/env python
"""MiMo MTP A/B: the same requests through strata-poly once with the draft model and once without.

The two runs cannot overlap (one card) and must share inputs, so one script drives both. Each phase
starts the engine from the config's own arguments (strata-mimo-mopd-q2k.json, so the bench measures
the run the user actually does) with its OWN fresh session dir and its OWN copy of the expert
profile: the user's files are read only, never written. The draft phase runs first and its real
answer becomes the assistant turn of request 2, so BOTH phases see byte-identical prompts.

Per phase, temperature=0 seed=42 through the MiMo chat template (Service.encode_prompt):
  req 1 - an agent-style first turn, cold: the whole prompt read
  req 2 - that turn's answer plus a follow-up: the part before the answer reused, the tail read
  req 3.. - req 2 again, --repeats times: the accept rate and decode speed of warm agent traffic,
            and the draft policy's warm-up visible across the repeats
The draft phase keeps --draft-model; the no-draft phase drops it and adds --no-draft, so its
drafts_accepted is 0 and its decode ms is the plain per-token verifier cost. The comparison the user
wants is decode tok/s (draft) vs decode tok/s (no-draft): the first is MTP with the verifier, the
second is MTP removed.

Refuses to start while strata-poly.exe or the Qwen server runs. Writes manifest.json,
command-<phase>.json, protocol-<phase>.txt, requests-<phase>.json, results.json, summary.txt,
run.log into a fresh bench/results/<stamp>-mimo-mtp-ab/; nothing existing is overwritten.
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
from serve.telemetry import gpu_process_memory  # noqa: E402

CONFIG = REPO / "strata-mimo-mopd-q2k.json"
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


def fresh_args(base_args, work, drop_draft):
    """base_args with the session dir and expert profile redirected into `work`, and, for the
    no-draft phase, --draft-model removed and --no-draft added. The originals are only read."""
    args = list(base_args)
    work.mkdir(parents=True, exist_ok=False)
    (work / "sessions").mkdir()
    args[args.index("--session-dir") + 1] = str(work / "sessions")
    prof = args[args.index("--expert-profile") + 1]
    shutil.copyfile(prof, work / "expert-profile.txt")     # the user's profile stays untouched
    args[args.index("--expert-profile") + 1] = str(work / "expert-profile.txt")
    if drop_draft:
        i = args.index("--draft-model")
        del args[i:i + 2]
        args.append("--no-draft")
    return args


def run_phase(name, base_args, enc, prompts, out, say, exe, env):
    """One engine start through the fixed sequence; returns (requests, error)."""
    say("")
    say(f"=== фаза {name}: запуск движка (загрузка 2-4 минуты, тишина в этот момент нормальна) ===")
    args = fresh_args(base_args, out / ("work-" + name), drop_draft=(name != "draft"))
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

    def line():
        while True:
            try:
                ln = lines.get(timeout=2)
            except queue.Empty:
                if p.poll() is not None:
                    raise RuntimeError(f"движок вышел раньше срока (код {p.poll()}, см. engine-{name}.stderr)")
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
            ln = line()
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
        while not line().startswith("READY "):
            pass
        say(f"READY за {time.monotonic() - t0:.1f} с")
        requests.append(generate(1, prompts["req1"], prompts["max_new"]))
        # req2 is built once, from the draft phase's real answer, and reused verbatim by the no-draft
        # phase, so both phases run byte-identical prompts; the draft phase runs first for this reason
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
    return requests, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(REPO / "build-poly-cuda12" / "strata-poly.exe"))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--kill-check", default="on", choices=["on", "off"])
    args = ap.parse_args()

    live = engines_running() if args.kill_check == "on" else []
    if live:
        print("ОСТАНОВЛЕНО: уже работают движки: " + ", ".join(f"{n} (PID {p})" for n, p in live))
        print("A/B нужна вся карта целиком; закрой сервер MiMo (и Qwen) и запусти снова.")
        return 2

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out = REPO / "bench" / "results" / f"{stamp}-mimo-mtp-ab"
    out.mkdir(parents=False, exist_ok=False)
    runlog = open(out / "run.log", "w", encoding="utf-8")
    say = Tee(runlog)
    say(f"# MiMo MTP A/B: draft vs no-draft, {args.gen_tokens} ток. на ответ, {args.repeats} повтора прогрева")
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
                   "phases": ["draft", "nodraft"]}, f, indent=2)

    results, error = {}, None
    t_all = time.monotonic()
    try:
        # draft phase first: run_phase builds req2 from its real answer and stores it in the shared
        # dict, so the no-draft phase runs the byte-identical prompts (same req2, same seed, same greedy)
        prompts = {"req1": req1, "req2": None, "max_new": args.gen_tokens, "repeats": args.repeats}
        results["draft"], e1 = run_phase("draft", cfg["args"], enc, prompts, out, say, exe, env)
        if e1:
            raise RuntimeError(e1)
        results["nodraft"], e2 = run_phase("nodraft", cfg["args"], enc, prompts, out, say, exe, env)
        if e2:
            raise RuntimeError(e2)
    except Exception as e:
        error = repr(e)
        say(f"ОСТАНОВЛЕНО: {error}")
    finally:
        summary = write_summary(out, results, say)
        with (out / "results.json").open("w", encoding="utf8") as f:
            json.dump({"results": results, "error": error}, f, indent=2, ensure_ascii=False)
        # сессионные дампы - KV-состояние прочитанных промптов (сотни MiB на файл при 50K):
        # результат прогона - числа, уже записанные в results.json, состояние за ними
        # воспроизводится тем же промптом; без уборки день A/B кликов оставляет гигабайты.
        dumps = [f for d in out.glob("work-*/sessions") for f in d.glob("*") if f.is_file()]
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
        say(f"всего {time.monotonic() - t_all:.0f} с; папка: {out}")
        runlog.close()
    if summary:
        (out / "summary.txt").write_text(summary, encoding="utf-8")
    return 0 if error is None else 1


def write_summary(out, results, say):
    """decode tok/s draft vs no-draft on the warm repeats, plus the accept that explains it."""
    def warm(phase, key):
        rows = [r for r in results.get(phase, []) if r.get("number", 0) >= 3]
        return [r[key] for r in rows]

    lines = ["", "=== парность ответов (temperature=0: черновики не имеют права менять вывод) ==="]
    lines += parity(results)
    d_tps = median(warm("draft", "decode_tps"))
    n_tps = median(warm("nodraft", "decode_tps"))
    acc = median(warm("draft", "draft_accept"))
    lines.append("")
    lines.append("=== ИТОГ: тёплые повторы (req 3+), медиана ===")
    if d_tps and n_tps:
        lines.append(f"decode без MTP   : {n_tps:.2f} ток/с")
        lines.append(f"decode с MTP     : {d_tps:.2f} ток/с   ({100 * (d_tps / n_tps - 1):+.1f}%)")
        lines.append(f"accept (черновики): {acc:.1%}" if acc is not None else "accept: нет данных")
        # no break-even threshold claimed: it follows from each draft step's and the verifier's own cost,
        # which this run does not separate; the measured delta above is the whole claim
    else:
        lines.append("недостаточно данных (одна из фаз не дошла до тёплых повторов).")
    for ln in lines:
        say(ln)
    return "\n".join(lines) + "\n"


def parity(results):
    """Greedy speculative decoding must answer exactly what the draft-less run answers - a speedup on
    different tokens is no speedup. Returns one line per paired request, mismatch spelled out."""
    d = {r["number"]: r for r in results.get("draft", [])}
    n = {r["number"]: r for r in results.get("nodraft", [])}
    lines = []
    for num in sorted(set(d) & set(n)):
        a, b = d[num], n[num]
        if a["ids"] == b["ids"] and a["finish"] == b["finish"]:
            lines.append(f"req {num}: токены и finish совпадают ({len(a['ids'])} ток.)")
        else:
            diff = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y),
                        min(len(a["ids"]), len(b["ids"])))
            lines.append(f"req {num}: РАСХОЖДЕНИЕ - первый differing токен #{diff} "
                         f"(draft: {len(a['ids'])} ток., {a['finish']}; no-draft: {len(b['ids'])} ток., {b['finish']}); "
                         f"ускорение этой фазы недействительно, пока черновики меняют вывод")
    if not lines:
        lines.append("нет парных запросов (одна из фаз не дала ответов).")
    return lines


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


if __name__ == "__main__":
    sys.exit(main())
