"""The split-K sweep on a card that must be IDLE - about a minute, no model load.

Run today with the assistant's Qwen server up, the same kernel path measured 36 ms where an idle card
gave 2.44 - fifteen times the time, on cells whose code had not changed.  A busy card does not fail a
kernel bench, it answers slowly and confidently, so this refuses to measure one unless no compute
process holds over 1000 MiB on it, and says why.  Everything else is poly_mimo2_attn_bench's decode
matrix; its output goes to a fresh bench/results folder, never over an earlier one.
"""
import importlib.util
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("msb", REPO / "bench" / "mimo_speed_bench.py")
msb = importlib.util.module_from_spec(spec)
sys.modules["msb"] = msb
spec.loader.exec_module(msb)


def card_users():
    """The engines themselves - their mere presence on the card wrecks long-kernel timing whatever
    nvidia-smi says: today's three sweeps ran at utilization 0% and 21 GB "free" and still measured
    7x slow on cells over a scheduling quantum, because the chat engine's resident context makes the
    driver preempt every long kernel between contexts.  nvidia-smi's memory column reads [N/A] per
    process under WDDM, so the old MiB filter could not fire; the process list is the truth."""
    return msb.engines_running()


def card_busy():
    """The card computing something right now.  Memory alone is not enough: the sweep of
    2026-11-08-092147 ran with 3.3 GB used and the whole card busy elsewhere, and every number came
    out 7-60x too slow with nothing in the memory column to say so.  Three samples over two
    seconds, any over 5% means wait."""
    seen = []
    for k in range(3):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout
            seen.append(int(out.strip().splitlines()[-1]))
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return [100]          # cannot read it: refuse rather than measure a lie
        if k < 2:
            time.sleep(1)
    return [u for u in seen if u > 5]


def main():
    busy = card_users()
    if busy:
        print("НА КАРТЕ ДВИЖКИ - измерять нельзя: их резидентный контекст preempt'ит длинные ядра,"
              " числа выйдут в разы медленнее правды (сегодня проверено трижды):")
        for name, pid in busy:
            print(f"  PID {pid}  {name}")
        print("Закрой оба сервера (Qwen-сервер чата тоже) и запусти снова.")
        return 2
    hot = card_busy()
    if hot:
        print(f"КАРТА СЧИТАЕТ ЧУЖОЕ (загрузка {max(hot)}% при трёх замерах) - числа выйдут в разы"
              " медленнее правды. Закрой всё, что грузит карту (игры, генерации, рендеры), и снова.")
        return 2
    out = REPO / "bench" / "results" / f"{datetime.now():%Y-%m-%d-%H%M%S}-attn-splits"
    out.mkdir(parents=True, exist_ok=False)
    exe = REPO / "build-poly-cuda12" / "poly_mimo2_attn_bench.exe"
    r = subprocess.run([str(exe), "60000", "decode"], capture_output=True, text=True, timeout=1800)
    (out / "sweep.txt").write_text(r.stdout + ("\n" + r.stderr if r.stderr else ""), encoding="utf-8")
    print(r.stdout)
    print(f"папка: {out}  выход: {r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
