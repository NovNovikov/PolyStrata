#!/usr/bin/env python3
"""(third round 14) Mock-engine tests of tests/dsv41/short_prompt_check.py. No model, no GPU, no pack:
a scripted stand-in speaks the engine's --serve protocol (READY / T / DONE / ERR / QUIT), and the checks
grade the CHECKER: exit codes for every way a run can go wrong, the engine's FULL stderr surviving into
both the script's own output (where kuu_features_test.bat greps its witnesses) and the KUU_ENGINE_LOG
file, and the round-14 strictness itself - a DONE with too few tokens, an engine that exits non-zero,
an engine that dies before READY - each of which the second-round script waved through or deleted.

run:  python tests/dsv41/short_prompt_check_test.py     (exit 0 = all good)
"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "short_prompt_check.py")

MOCK = r"""
import os, sys
sc = os.environ.get("MOCK_SCENARIO", "ok")
def out(s):
    print(s, flush=True)
def err(s):
    sys.stderr.write(s + "\n"); sys.stderr.flush()
if sc == "noready":
    err("ds4: refusing to start (mock)\n")
    sys.exit(2)
err("ds4: routers kept in BF16 bits (mock witness)\n")
for i in range(40):
    err(f"ds4: mock load line {i}\n")
out("READY 1 1024")
err("ds4: engine up (mock)\n")
while True:
    l = sys.stdin.readline()
    if not l:
        break
    if l.startswith("QUIT"):
        err("ds4: farewell line after QUIT (mock)\n")
        sys.exit(3 if sc == "quit3" else 0)
    if l.startswith("GEN"):
        if sc == "err":
            out("ERR boom (mock)"); continue
        n = 15 if sc == "short" else 16
        for i in range(n):
            out(f"T {i} 42")
            if sc == "die" and i == 2:
                err("ds4: dying mid-request (mock)\n")
                os._exit(9)
            if sc == "badmem" and i == 5:
                err("ds4: CUDA error: an illegal memory access was encountered (mock)\n")
        out("DONE 42 stop")
"""

failures = 0
def check(ok, what):
    global failures
    if not ok:
        failures += 1
        print(f"  FAIL: {what}")

def main():
    tmp = tempfile.mkdtemp(prefix="spc-test-")
    mock_py = os.path.join(tmp, "mock_engine.py")
    with open(mock_py, "w") as f:
        f.write(MOCK)
    if os.name == "nt":
        shim = os.path.join(tmp, "mock_engine.bat")
        with open(shim, "w") as f:
            f.write(f'@echo off\r\n"{sys.executable}" "{mock_py}" %*\r\n')
    else:
        shim = os.path.join(tmp, "mock_engine.sh")
        with open(shim, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{mock_py}" "$@"\n')
        os.chmod(shim, 0o755)
    idsf = os.path.join(tmp, "ids.txt")
    with open(idsf, "w") as f:
        f.write(",".join(str(100 + i) for i in range(40)))

    def run(scenario):
        env = dict(os.environ, MOCK_SCENARIO=scenario, KUU_ENGINE_LOG=os.path.join(tmp, f"{scenario}.engine.log"))
        p = subprocess.run([sys.executable, SCRIPT, shim, "unused-pack", idsf],
                           capture_output=True, text=True, timeout=180, env=env,
                           encoding="utf-8", errors="replace")
        log = ""
        if os.path.exists(env["KUU_ENGINE_LOG"]):
            with open(env["KUU_ENGINE_LOG"], encoding="utf-8", errors="replace") as f:
                log = f.read()
        return p, log

    # (a) the clean run: exit 0, all three requests, and the engine's stderr WHOLE - the witness, all 40
    # load lines, the mid-run line and the post-QUIT farewell - in the caller's stream and in the file.
    p, log = run("ok")
    check(p.returncode == 0, f"ok scenario exits 0 (got {p.returncode}: {p.stdout[:200]})")
    check(p.stdout.count("request") == 3 and p.stdout.count("16 tokens") == 3,
          "ok scenario grades three full requests")
    check("routers kept in BF16" in p.stderr, "the engine witness reaches the script's stderr (the .bat greps there)")
    for s in ("routers kept in BF16", "mock load line 0", "mock load line 39", "engine up", "farewell line after QUIT"):
        check(s in log, f"the kept engine log carries: {s}")
    check(log.count("ds4: mock load line") == 40, "and carries every load line, not the tail or the head")
    check(os.path.exists(os.path.join(tmp, "ok.engine.log")), "the log file is KEPT, not unlinked")

    # (b) protocol failures grade as failures now
    p, _ = run("err")
    check(p.returncode == 1 and "request 1 ERR" in p.stdout, "an ERR line fails the run")
    p, _ = run("short")
    check(p.returncode == 1 and "expected 16" in p.stdout, "a DONE with 15 tokens fails (second-round script: silent)")
    p, _ = run("die")
    check(p.returncode == 1 and "engine died" in p.stdout, "a death mid-request fails")
    p, _ = run("quit3")
    check(p.returncode == 1 and "exited with code 3" in p.stdout, "a non-zero exit after QUIT fails (was ignored)")
    p, log = run("noready")
    check(p.returncode == 1 and "ended before READY" in p.stdout, "a death before READY fails")
    check("refusing to start" in log, "and its stderr is still kept for the postmortem")

    # (c) the poison lines in the engine's stderr fail a protocol-clean run, as before
    p, _ = run("badmem")
    check(p.returncode == 1 and "illegal memory access" in p.stdout, "an illegal-access stderr line fails the run")

    print(f"short_prompt_check_test: {'FAILURES ' + str(failures) if failures else 'all good'}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
