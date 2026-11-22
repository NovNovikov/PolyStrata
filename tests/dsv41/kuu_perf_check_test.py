#!/usr/bin/env python3
"""(perf round) Mock-engine tests of tests/dsv41/kuu_perf_check.py. No model, no GPU, no pack: a
scripted stand-in speaks the engine's --serve protocol and prints the witness lines the real engine
prints; the checks grade the CHECKER - every run-name, the three-way verdict for every witness shape
(activation / named reason / silence), the engine-trouble shapes (short DONE, ERR, death, no READY,
poison line, non-zero exit after QUIT), and the RESULT line the .bat collects being parseable.

run:  python tests/dsv41/kuu_perf_check_test.py     (exit 0 = all good)
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "kuu_perf_check.py")

MOCK = r"""
import os, sys
sc = os.environ.get("MOCK_SCENARIO", "ok")
wit = os.environ.get("MOCK_WIT", "none")
def out(s):
    print(s, flush=True)
def err(s):
    sys.stderr.write(s + "\n"); sys.stderr.flush()
if sc == "noready":
    err("ds4: refusing to start (mock)\n")
    sys.exit(2)
err("ds4: loaded in 2.5 s\n")
if wit != "none":
    for w in wit.split("||"):
        err("ds4: " + w + " (mock witness)\n")
out("READY 1 1024")
while True:
    l = sys.stdin.readline()
    if not l:
        break
    if l.startswith("QUIT"):
        sys.exit(3 if sc == "quit3" else 0)
    if l.startswith("GEN"):
        if sc == "err":
            out("ERR boom (mock)"); continue
        want = int(l.split()[1])
        n = want - 1 if sc == "short" else want
        for i in range(n):
            out(f"T {i} 42")
            if sc == "die" and i == 2:
                err("ds4: dying mid-request (mock)\n")
                os._exit(9)
            if sc == "badmem" and i == 5:
                err("ds4: CUDA error: an illegal memory access was encountered (mock)\n")
        out("DONE 42 stop")
"""
BANNER = "prompt path: the chunked pass from 96 tokens, windows of 16 below"
EXACT6 = ("routers kept in BF16 bits||DSpark not loaded||DSpark lazy: 3 rounds||requested slots of 8"
          "||next-layer read-ahead: 4 chunks||unmapped, their buffered handles closed")
EXACT5 = ("routers kept in BF16 bits||DSpark not loaded||DSpark lazy: 3 rounds||requested slots of 8"
          "||next-layer read-ahead: 4 chunks")
RES_RE = re.compile(r"^RESULT (\S+) (PASS|SKIP|FAIL) load=([\d.\-]+) ready=([\d.\-]+) warm=([\d.\-]+)"
                    r" prefill=([\d.\-]+) decode=(\S+) note=(.*)$", re.M)

def main():
    tmp = tempfile.mkdtemp(prefix="kpf-test-")
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
        f.write(",".join(str(100 + i) for i in range(512)))

    failures = 0
    def check(ok, what):
        nonlocal failures
        if not ok:
            failures += 1
            print(f"  FAIL: {what}")

    def run(name, wit, sc="ok"):
        env = dict(os.environ, MOCK_SCENARIO=sc, MOCK_WIT=wit, KUU_ENGINE_LOG=os.path.join(tmp, f"{name}-{sc}-{abs(hash(wit))}.engine.log"),
                   KUUPERF_GENS="2", KUUPERF_WARM="1", KUUPERF_TOKENS="8")
        p = subprocess.run([sys.executable, SCRIPT, shim, "unused-pack", name, idsf],
                           capture_output=True, text=True, timeout=180, env=env,
                           encoding="utf-8", errors="replace")
        m = RES_RE.search(p.stdout)
        return p, m

    # the witness table: activation PASS, named reason SKIP, silence FAIL - per run-name
    for name, wit, want in [
        ("baseline", BANNER, "PASS"),
        ("baseline", "none", "FAIL"),                       # set-and-silent is still the thing to catch
        ("defer3", "STRATA_DS4_DEFER=3 is armed", "PASS"),
        ("eg_ring", "requested slots of 8 slots", "PASS"),
        ("eg_ring", "Engram ring: 12.0 MB of pinned slots would not fit (1 of 8 got)", "SKIP"),
        ("eg_ring", "Engram ring armed: the chunked pass owns it", "SKIP"),
        ("eg_ring", "none", "FAIL"),
        ("lm_ahead", "next-layer read-ahead: 4 chunks ahead", "PASS"),
        ("lm_ahead", "read-ahead armed: the chunked pass owns it", "SKIP"),
        ("lm_ahead", "none", "FAIL"),
        ("store_base", "RAM store: about 400 slots in 7 size classes", "PASS"),
        ("uniform", "the store runs in 1 size classes", "PASS"),
        ("uniform", "RAM store: about 400 slots in 7 size classes", "FAIL"),   # built in classes: no took
        ("uniform", "none", "SKIP"),                                           # no store at all
        ("drop_shards", "2 expert file(s) unmapped, their buffered handles closed", "PASS"),
        ("drop_shards", "(local) STRATA_DS4_DROP_SHARDS armed but idle: no demand store", "SKIP"),
        ("drop_shards", "none", "FAIL"),
        ("all_exact", EXACT6, "PASS"),
        ("all_exact", EXACT5, "FAIL"),                       # drop_shards has no trace
    ]:
        p, m = run(name, wit)
        check(m is not None, f"{name}/{wit[:28]!r}: a parseable RESULT line (stdout: {p.stdout[-300:]!r})")
        if m:
            check(m.group(1) == name and m.group(2) == want,
                  f"{name}/{wit[:28]!r}: verdict {m.group(2)}, wanted {want}")
            want_exit = 1 if want == "FAIL" else 0
            check(p.returncode == want_exit, f"{name}/{wit[:28]!r}: exit {p.returncode}, wanted {want_exit}")
        if name == "all_exact" and m and want == "FAIL":
            check("no trace for: drop_shards" in m.group(8), f"all_exact names the missing trace: {m.group(8)}")

    # a healthy RESULT carries the load the engine reported and one decode number per measured GEN
    p, m = run("baseline", BANNER)
    check(m and m.group(3) == "2.5", f"the engine's own 'loaded in 2.5 s' lands in the RESULT load= ({m and m.group(3)})")
    check(m and m.group(7).count(",") == 1 and all(float(x) > 0 for x in m.group(7).split(",")),
          f"two measured GENs, two positive decode tok/s numbers ({m and m.group(7)})")

    # engine trouble outranks every witness: FAIL, exit 1, postmortem kept
    for sc, why in [("short", "a DONE one token short"), ("err", "an ERR line"),
                    ("die", "a death mid-request"), ("badmem", "an illegal-access line"),
                    ("quit3", "a non-zero exit after QUIT")]:
        p, m = run("baseline", BANNER, sc)
        check(m is not None and m.group(2) == "FAIL" and p.returncode == 1,
              f"{why} fails despite the witness ({sc}: {m and m.group(2)})")
    p, m = run("baseline", BANNER, "noready")
    check(m is not None and m.group(2) == "FAIL" and "died before READY" in m.group(8), "no READY fails with a RESULT line too")
    logf = None
    for f in os.listdir(tmp):
        if f.startswith("baseline-noready-") and f.endswith(".engine.log"):
            logf = os.path.join(tmp, f)
    check(logf and "refusing to start" in open(logf, encoding="utf-8", errors="replace").read(),
          "and the engine log is kept for the postmortem")

    # an unknown run-name fails loudly, not silently
    p, m = run("nosuchname", "none")
    check(m is not None and m.group(2) == "FAIL" and "unknown run-name" in m.group(8), "an unknown run-name fails")

    print(f"kuu_perf_check_test: {'FAILURES ' + str(failures) if failures else 'all good'}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
