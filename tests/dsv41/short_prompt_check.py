#!/usr/bin/env python3
"""short_prompt_check.py <strata-dsv41> <pack> <ids file (comma separated token ids)> [engine args...]

A short prompt (40 tokens: the windows path, below the chunked pass) with the expert cache on, default
sampler, twice, then a greedy one. Fails (exit 1) on any ERR line, an engine death, a DONE that did not
carry its 16 tokens, an engine that ignores QUIT or exits with a non-zero code, or an engine stderr line
with 'illegal memory access', 'router picked' or 'the move into VRAM failed'.
Found on HIP (p620): a swap copy out of a slot in the lent tail region read past the allocation (the RAM
store's put used vram_base + slot * unit instead of sat()), which only the windows path's swaps reached.

(third round 14) The engine's stderr IS the activation record its caller greps - kuu_features_test.bat
reads witness lines out of this script's output - so the FULL engine stderr now survives the run: every
line is teed to this script's own stderr as it arrives (which the .bat captures into its per-run log),
and kept in a file whose path comes from KUU_ENGINE_LOG (or a temporary of the run), never deleted. The
second-round script parked the engine stderr in a temporary it unlinked at the end, so the witnesses
grepped a file that only ever carried this script's own six prints - every feature run looked like a
silent no-op. Grading tightened with it: an engine that does not exit after QUIT, or exits non-zero, is
a failure, and a failure prints the engine log's last lines. Mock-engine tests of this script, no model
needed: tests/dsv41/short_prompt_check_test.py.
"""
import os, queue, subprocess, sys, tempfile, threading

TAIL = 50   # (engine log lines printed on a failure, the caller's first postmortem)

def dump_tail(logname):
    try:
        with open(logname, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    print(f"--- engine log, last {TAIL} of {len(lines)} lines ---")
    for l in lines[-TAIL:]:
        print(l.rstrip("\n"))

def main():
    exe, pack, idsf = sys.argv[1:4]
    rest = sys.argv[4:]
    ids = [int(x) for x in open(idsf).read().strip().split(",")][:40]
    logname = os.environ.get("KUU_ENGINE_LOG") or os.path.join(
        tempfile.gettempdir(), "short_prompt_check.engine.log")
    log = open(logname, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([exe, "--serve", "--pack", pack, "--max-context", "16384", "--eos", "1",
                          "--vision", "off", "--dspark", "off"] + rest,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1, encoding="utf-8", errors="replace")
    q = queue.Queue()
    err_done = queue.Queue()
    def pump_out():
        for l in p.stdout:
            q.put(l.rstrip("\n"))
        q.put(None)
    def pump_err():
        # (third round 14) every engine stderr line, verbatim, twice: into the log file that survives the
        # run and onto our own stderr, where kuu_features_test.bat's log (and its witness greps) sees it.
        for line in p.stderr:
            log.write(line)
            log.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
        p.stderr.close()
        err_done.put(None)
    threading.Thread(target=pump_out, daemon=True).start()
    threading.Thread(target=pump_err, daemon=True).start()
    def drained():
        try:
            err_done.get(timeout=15)
        except queue.Empty:
            pass
    bad = 0
    try:
        while True:
            l = q.get(timeout=3600)
            if l is None:
                print("engine ended before READY")
                drained(); dump_tail(logname)
                print("short prompt check: FAILED | engine log:", logname)
                return 1
            if l.startswith("READY"):
                break
        for k, keys in enumerate(("", "", "temperature=0")):
            p.stdin.write(f"GEN 16 {keys} {','.join(map(str, ids))}\n")
            p.stdin.flush()
            n = 0
            while True:
                l = q.get(timeout=900)
                if l is None:
                    print("request", k + 1, "engine died"); bad += 1; break
                if l.startswith("T "):
                    n += 1
                if l.startswith("ERR"):
                    print("request", k + 1, l[:120]); bad += 1; break
                if l.startswith("DONE"):
                    print(f"request {k + 1}: {n} tokens | {l[:80]}")
                    bad += n != 16
                    if n != 16:
                        print(f"request {k + 1}: expected 16 tokens, the DONE carried {n}")
                    break
            if bad:
                break
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
        except OSError:
            pass   # (the engine is already gone: the wait and its exit code grade that)
        killed = False
        try:
            p.wait(timeout=120)
        except Exception:
            p.kill(); p.wait(); killed = True
        drained()   # (the pipe closed with the engine: the file holds every line it ever wrote)
        if killed:
            print("engine did not exit after QUIT (killed after 120 s)"); bad += 1
        elif p.returncode != 0:
            print(f"engine exited with code {p.returncode}"); bad += 1
        for line in open(logname, encoding="utf-8", errors="replace"):
            if any(s in line for s in ("illegal memory access", "router picked", "the move into VRAM failed")):
                print("engine log:", line.strip()[:160]); bad += 1
        if bad:
            dump_tail(logname)
        print("short prompt check:", "FAILED" if bad else "ok", "| engine log:", logname)
        return 1 if bad else 0
    finally:
        log.close()

if __name__ == "__main__":
    sys.exit(main())
