"""Dry-run every phase of run-qwen-best-ab.bat and run-qwen-tuning-ab.bat, then assert the
assembled engine command from command.json. Run: python bench/test_qwen_bench_dry.py"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
UP = r"L:\AI_pictures_generate\strata-engine-0.1.58\strata.exe"
DF2 = r"L:\AI_pictures_generate\PolyStrata-data\dflash2\qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack"
D = ["--drop=--vram-reserve-mib,--session-stream-save"]

PHASES = [  # (label, extra args, expected substrings in command, forbidden substrings)
    ("our-8k", [], ["--serve", "--pack"], ["--vram-reserve-mib", "--session-stream-save", "--draft"]),
    ("our-16k", ["--set=--prefill=auto:16384"], ["--prefill auto:16384"], ["--draft"]),
    ("our-32k", ["--set=--prefill=auto:32768"], ["--prefill auto:32768"], ["--draft"]),
    ("up-auto", [f"--exe={UP}"], ["--serve", UP], ["--draft"]),
    ("up-32k", [f"--exe={UP}", "--set=--prefill=auto:32768"], ["--prefill auto:32768", UP], ["--draft"]),
    ("up-df2", [f"--exe={UP}", "--set=--draft=dflash2", f"--set=--draft-model={DF2}",
                "--set=--draft-fallback=none"], ["--draft dflash2", f"--draft-model {DF2}",
                "--draft-fallback none", "--mtp"], []),
    ("r4-sr0", ["--set=--prefill=auto:16384", "--set=--short-read=0"],
     ["--prefill auto:16384", "--short-read 0"],
     ["--prefill-head-alias", "--kv-grow", "--host-core", "--aux-cpus"]),
    ("r4-ha", ["--set=--prefill=auto:16384", "--set=--short-read=0", "--set=--prefill-head-alias"],
     ["--prefill auto:16384", "--short-read 0", "--prefill-head-alias"], ["--kv-grow"]),
    ("r4-ha-kg", ["--set=--prefill=auto:16384", "--set=--short-read=0", "--set=--prefill-head-alias",
                  "--set=--kv-grow"],
     ["--short-read 0", "--prefill-head-alias", "--kv-grow"], ["--host-core"]),
    ("r4-all", ["--set=--prefill=auto:16384", "--set=--short-read=0", "--set=--prefill-head-alias",
                "--set=--kv-grow", "--set=--host-core=sibling", "--set=--aux-cpus=auto"],
     ["--short-read 0", "--prefill-head-alias", "--kv-grow", "--host-core sibling",
      "--aux-cpus auto"], []),
    ("r4-sr512", ["--set=--prefill=auto:16384", "--set=--short-read=512"],
     ["--short-read 512"], ["--prefill-head-alias"]),
    ("r5-cpu", ["--set=--prefill=auto:16384", "--set=--short-read=512",
                "--set=--host-core=sibling", "--set=--aux-cpus=auto"],
     ["--short-read 512", "--host-core sibling", "--aux-cpus auto"], ["--prefill-head-alias"]),
    ("r5-kv", ["--set=--prefill=auto:16384", "--set=--kv-grow"],
     ["--prefill auto:16384", "--kv-grow"],
     ["--short-read 512", "--short-read 0", "--prefill-head-alias", "--host-core"]),
    ("r6-off", ["--set=--prefill=auto:16384"],
     ["--prefill auto:16384"], ["--kv-grow", "--short-read 512", "--prefill-head-alias"]),
    ("r6-on", ["--set=--prefill=auto:16384", "--set=--kv-grow"],
     ["--prefill auto:16384", "--kv-grow"],
     ["--short-read 512", "--prefill-head-alias", "--host-core"]),
    ("r6-er", ["--set=--prefill=auto:16384", "--set=--kv-grow", "--set=--kv-resident=0"],
     ["--kv-grow", "--kv-resident 0"], ["--short-read 512", "--prefill-head-alias"]),
]


def main():
    fails = 0
    for label, extra, want, forbidden in PHASES:
        r = subprocess.run([PY, str(REPO / "bench" / "qwen_speed_bench.py"), "--dry-run",
                            f"--label={label}", *D, *extra],
                           capture_output=True, text=True, encoding="utf-8", timeout=300)
        if r.returncode != 0:
            print(f"FAIL {label}: dry-run exit {r.returncode}\n{r.stdout[-500:]}{r.stderr[-500:]}")
            fails += 1
            continue
        folder = sorted((REPO / "bench" / "results" / "dry").glob(f"*-{label}"))
        if len(folder) < 1:
            print(f"FAIL {label}: dry-папка не найдена")
            fails += 1
            continue
        cmd = " ".join(json.loads((folder[-1] / "command.json").read_text(encoding="utf-8"))["command"])
        miss = [w for w in want if w not in cmd]
        pres = [f for f in forbidden if f in cmd]
        if miss or pres:
            print(f"FAIL {label}: нет {miss}, лишнее {pres}\n  cmd: {cmd[:400]}")
            fails += 1
        else:
            print(f"ok   {label}")
    print("ИТОГ:", "все фазы собирают правильные команды" if not fails else f"{fails} фаз с проблемой")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
