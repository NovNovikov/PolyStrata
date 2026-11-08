"""Does the --decode-attn-warps flag reach the kernel from the user's config, without any env variable?

Two minutes, one click. The 7x of the decode attend was proven over STRATA_MIMO_ATTN_WARPS, and the
banner proved that path; the flag's own path - the config's args through the parser into the kernel -
no live run has shown yet. If it loses the setting somewhere, the server grinds 9 tokens a second with
a config that says auto and nothing complains. This starts the engine exactly as a normal launch does
(the config's args as they stand, the env stripped of STRATA_* so the kernel can only hear the flag),
sends one short prompt, takes a few decoded tokens and stops, then reads its own stderr copy:

    mimo2-attend: split kernel at 32 warps a block ... (--decode-attn-warps=auto)   - PASS
    the same line naming STRATA_MIMO_ATTN_WARPS                                      - the env leaked
    no line at all                                                                   - the flag never arrived

The user's session dir and expert profile are redirected into a temp folder, so the check touches no
state; the model load (~2 minutes) is the whole cost.
"""
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("msb", REPO / "bench" / "mimo_speed_bench.py")
msb = importlib.util.module_from_spec(spec)
sys.modules["msb"] = msb
spec.loader.exec_module(msb)

TMP = REPO / "bench" / "results" / f"{datetime.now():%Y-%m-%d-%H%M%S}-mimo-flag-check"


def main():
    live = msb.engines_running()
    if live:
        print("ОСТАНОВЛЕНО: уже работают движки: " + ", ".join(f"{n} (PID {p})" for n, p in live))
        print("Нужны вся карта и вся RAM. Закрой серверы и запусти снова.")
        return 2

    config = json.loads((REPO / "strata-mimo-mopd-q2k.json").read_text(encoding="utf-8-sig"))
    launch = list(config["args"])
    TMP.joinpath("sessions").mkdir(parents=True, exist_ok=True)
    launch[launch.index("--session-dir") + 1] = str(TMP / "sessions")
    prof = launch[launch.index("--expert-profile") + 1]
    shutil.copyfile(prof, TMP / "expert-profile.txt")          # the user's profile stays untouched
    launch[launch.index("--expert-profile") + 1] = str(TMP / "expert-profile.txt")
    if "--decode-attn-warps" not in launch:
        print("В конфиге нет --decode-attn-warps - проверять нечего, чини конфиг.")
        return 1

    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_dir(config["tokenizer"])
    ids = tok.encode("Сколько будет дважды два по пять? Отвечай коротко.")[:60]
    del tok
    print(f"запуск движка с конфигом как есть (подсказка из {len(ids)} токенов), загрузка ~2 минуты ...")

    exe = str(Path(config.get("exe", REPO / "build-poly-cuda12" / "strata-poly.exe")).resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["PATH"] = os.pathsep.join(config.get("lib_dirs", []) + [env.get("PATH", "")])
    err = open(TMP / "engine.stderr.log", "w", encoding="utf8", errors="replace")
    p = subprocess.Popen([exe, "--serve"] + launch, cwd=str(REPO), env=env, text=True, bufsize=1,
                         encoding="utf-8", errors="replace", stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)

    q: queue.Queue = queue.Queue()

    def reader():
        for ln in p.stdout:
            q.put(ln.rstrip())
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    p.stdin.write(f"GEN 16 temperature=0 seed=42 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    tokens, started, dead = 0, time.monotonic(), None
    while time.monotonic() - started < (420 if tokens == 0 else 120):
        try:
            ln = q.get(timeout=5)
        except queue.Empty:
            if p.poll() is not None:
                dead = f"движок вышел (код {p.poll()})"
                break
            continue
        if ln is None:
            dead = "у движка закрылся stdout"
            break
        if ln.startswith(("ERR ", "FATAL ", "SERR ")):
            dead = ln
            break
        if ln.startswith("T "):
            tokens += 1
        if ln.startswith("DONE "):
            break
    print(f"получено токенов: {tokens}" + (f", смерть: {dead}" if dead else ""))
    p.terminate()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
    err.close()

    text = (TMP / "engine.stderr.log").read_text(encoding="utf8", errors="replace")
    banners = [ln.strip() for ln in text.splitlines() if "mimo2-attend" in ln]
    print("строка(и) ядра: " + (" | ".join(banners) if banners else "НЕТ НИ ОДНОЙ"))
    if any("--decode-attn-warps" in b for b in banners):
        print("ВЕРДИКТ: флаг доехал - ядро запущено так, как велит конфиг.")
        return 0
    if any("STRATA_MIMO_ATTN_WARPS" in b for b in banners):
        print("ВЕРДИКТ: сработала переменная окружения, а не флаг - окружение протекло, проверять надо")
        print("        сам запуск, этот чекер окружение вычищает.")
        return 1
    print("ВЕРДИКТ: флаг НЕ доехал - ядро запустилось с прежними четырьмя warp'ами, сервер мелет медленно.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
