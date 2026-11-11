#!/usr/bin/env python
"""Compare qwen_speed_bench runs - the control first, the variant last, a verdict printed.

python bench/qwen_ab_compare.py                 -> the two newest bench/results/*-qwen-speed-b
python bench/qwen_ab_compare.py RUN ...         -> named folders or glob patterns, control first
Reads results.json of each (command, load_s, requests[0]); prints a table of prompt tok/s,
decode tok/s and load seconds, and the last run's change on the first with a one-line verdict.
With 3+ runs it also marks the best prompt and the best decode row.
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(folder):
    p = os.path.join(folder, "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    if not r.get("requests"):
        return None
    req = r["requests"][0]
    return {"folder": os.path.basename(folder), "exe": os.path.basename(os.path.dirname(r["command"][0]))
            + "\\" + os.path.basename(r["command"][0]), "load_s": r.get("load_s"),
            "prompt_tps": req.get("prompt_tps"), "decode_tps": req.get("decode_tps"),
            "gen": req.get("generated"), "exit": r.get("exit")}


def resolve(arg):
    hits = sorted(glob.glob(arg))
    return hits if hits else ([arg] if os.path.isdir(arg) else [])


def main():
    args = sys.argv[1:]
    if args:
        runs, missing = [], []
        for a in args:
            hit = resolve(a)
            got = [r for r in (load(d) for d in hit) if r]
            if got:
                runs.append(got[-1])
            else:
                missing.append(a)
        for m in missing:
            print(f"пропуск (нет готового results.json): {m}")
        if len(runs) < 2:
            print("сравнять нечего: готовых прогонов меньше двух")
            return 1
    else:
        found = sorted(d for d in glob.glob(os.path.join(REPO, "bench", "results", "*-qwen-speed-b"))
                       if os.path.exists(os.path.join(d, "results.json")))
        if len(found) < 2:
            print(f"нужно два прогона, найдено {len(found)}: {found}")
            return 1
        runs = [load(d) for d in found[-2:]]
    print(f"{'':28} {'prompt ток/с':>12} {'decode ток/с':>13} {'загрузка':>9}  exe")
    for r in runs:
        m = re.search(r"-(\d{6})-qwen-speed-[ab]-(.+)$", r["folder"])   # HHMMSS + label identifies the run
        shown = (f"{m.group(1)} {m.group(2)}" if m else r["folder"])[:28]
        print(f"{shown:28} {r['prompt_tps'] or 0:12.1f} {r['decode_tps'] or 0:13.2f}"
              f" {r['load_s'] or 0:8.1f}с  {r['exe']}")
    if len(runs) >= 3:
        bp = max(runs, key=lambda r: r["prompt_tps"] or 0)
        bd = max(runs, key=lambda r: r["decode_tps"] or 0)
        print(f"лучший prompt: {bp['folder']}  {bp['prompt_tps']:.1f} ток/с")
        print(f"лучший decode: {bd['folder']}  {bd['decode_tps']:.2f} ток/с")
    a, b = runs[0], runs[-1]
    if a is not b and a["decode_tps"] and b["decode_tps"]:
        d = 100 * (b["decode_tps"] / a["decode_tps"] - 1)
        dp = (100 * (b["prompt_tps"] / a["prompt_tps"] - 1)
              if a["prompt_tps"] and b["prompt_tps"] else None)
        verdict = ("последний быстрее" if d > 2 else "последний медленнее" if d < -2
                   else "в пределах шума")
        line = f"вердикт: decode {a['decode_tps']:.2f} -> {b['decode_tps']:.2f} ({d:+.1f}%) - {verdict}"
        if dp is not None:
            line += f"; prompt {a['prompt_tps']:.0f} -> {b['prompt_tps']:.0f} ({dp:+.0f}%)"
        print(f"\n{line}")
    elif a is b:
        print("\nвердикт: это один и тот же прогон")
    else:
        print("\nвердикт: не вычислить - один из прогонов без decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
