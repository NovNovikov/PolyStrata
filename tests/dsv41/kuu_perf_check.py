#!/usr/bin/env python3
"""kuu_perf_check.py <exe> <pack> <run-name> <ids-file> [engine args...]

One real-load run of the engine on a real pack, for ONE feature's measurement: a long prompt (the ids
file, whole - kuu_perf_test.bat makes one of 32,768, the size the user's own benches run: the qwen A/B
benches take a 32,675-token prompt with GEN 512, the mimo bench --gen 1024/2048), KUUPERF_WARM warm-up
GENs (default 2), then KUUPERF_GENS measured GENs of KUUPERF_TOKENS tokens (defaults 5 and 512 - the
qwen bench's number; five because three could not see the machine's own drift).
Every GEN is timed where the protocol makes it measurable:
send -> first 'T ' is the prefill (the prompt landing), first 'T ' -> DONE is the decode, so
decode tok/s = (tokens-1) / decode_s. The engine's own cold load comes from its 'ds4: loaded in X s'
line. Every engine stderr line is kept (KUU_ENGINE_LOG) and teed, as in short_prompt_check.py.

The run-name picks the ACTIVATION witness, so a number is never reported without knowing whether the
switch actually reached its code: PASS printed the witness, SKIP printed a named reason it was idle,
FAIL is engine trouble or silence - the same three-way verdict as kuu_features_test.bat, in python this
time so the rules are testable (tests/dsv41/kuu_perf_check_test.py, mocks, no model).

The last line is one machine-readable RESULT the .bat collects:
  RESULT <name> PASS|SKIP|FAIL load=<s> ready=<s> warm=<tps> prefill=<s> decode=<tps,tps,...> note=<...>
prefill is the FIRST GEN's prompt read - the cold one; the later GENs reuse the KV and print their
0.1 s in the per-GEN lines only. Exit 0 for PASS and SKIP, 1 for FAIL. Running this needs the machine
to itself: it loads the whole pack (minutes, ~103 GB of RAM on this box). The user runs it; see
kuu_perf_test.bat.
"""
import os, queue, re, subprocess, sys, tempfile, threading, time

TAIL = 50   # (engine log lines printed on a failure, the first postmortem)
GEN_TMO = 1800     # one GEN incl. a 4K-token prefill: minutes, generous
READY_TMO = 3600   # (the pack load of this box: ~340 s; a cold page cache more)
POISON = ("illegal memory access", "router picked", "the move into VRAM failed")

# one run per parameter, as kuu_perf_test.bat calls them; (verdict, substring) tried in order,
# first hit decides; none hit -> FAIL "silent". The strings are the engine's own lines (ds4_model.cu).
WIT = {
    "baseline":   [("PASS", "ds4: prompt path")],   # the control: the banner is its whole check
    # A/B/A: every group runs its control twice, first and last (baseline ... baseline2, store_base ...
    # store_base2). The machine warms as the sweep walks - pack files heat the OS cache across runs and
    # the experts keep migrating inside one run - so the controls' own drift IS the noise estimate, and a
    # switch earns an effect only against the AVERAGE of its two controls. baseline2 is the same rules
    # and geometry as baseline, run last; the numbers differ only by when the machine was.
    "baseline2":  [("PASS", "ds4: prompt path")],
    "defer3":     [("PASS", "STRATA_DS4_DEFER=3 is armed")],
    "eg_ring":    [("PASS", "requested slots of"), ("SKIP", "would not fit"),
                   ("SKIP", "Engram ring armed")],
    # ring and read-ahead only DO something where the chunked pass streams experts from the files
    # (--mmap-experts on): in the copy-into-RAM geometry the layer-major path never runs and they
    # idle. The _mmap runs carry the switches into the geometry that has work for them; store_base
    # is their control.
    "ring_mmap":  [("PASS", "requested slots of"), ("SKIP", "would not fit"),
                   ("SKIP", "Engram ring armed")],
    "lm_ahead":   [("PASS", "next-layer read-ahead:"), ("SKIP", "read-ahead is off"),
                   ("SKIP", "read-ahead armed")],
    "ahead_mmap": [("PASS", "next-layer read-ahead:"), ("SKIP", "read-ahead is off"),
                   ("SKIP", "read-ahead armed")],
    "store_base": [("PASS", "RAM store: about")],
    "store_base2":[("PASS", "RAM store: about")],   # the streaming group's second control (A/B/A)
    "uniform":    [("PASS", "in 1 size classes"), ("FAIL", "RAM store:")],   # built but in classes: no took
    "drop_shards":[("PASS", "unmapped, their buffered handles closed"), ("SKIP", "0 expert"),
                   ("SKIP", "DROP_SHARDS armed but idle")],
    "all_exact":  None,   # the multi-trace check below
}
EXACT = {   # every switch of the exact set must leave a trace: its witness or its named reason
    "router_bf16": ("routers kept in BF16", "the switch does nothing"),
    "mtp_cap":     ("DS4_MTP_VRAM_GB: drafter", "no drafter is loaded", "DSpark not loaded"),
    "lazy_dspark": ("DSpark lazy:", "DSpark not loaded", "no drafter is loaded"),
    "eg_ring":     ("requested slots of", "would not fit", "Engram ring armed"),
    "lm_ahead":    ("next-layer read-ahead:", "read-ahead is off", "read-ahead armed"),
    "drop_shards": ("unmapped, their buffered handles closed", "DROP_SHARDS armed but idle"),
}

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
    exe, pack, name, idsf = sys.argv[1:5]
    rest = sys.argv[5:]
    if name not in WIT:
        print(f"RESULT {name} FAIL load=- ready=- warm=- prefill=- decode=- note=unknown run-name")
        print(f"run-names: {', '.join(sorted(WIT))}")
        return 1
    ids = [int(x) for x in open(idsf).read().strip().split(",")]
    gens = int(os.environ.get("KUUPERF_GENS", 5))   # (measured GENs: three could not see the machine's
    warm = int(os.environ.get("KUUPERF_WARM", 2))   #  own drift - decode moves run after run, 2026-11-22)
    ntok = int(os.environ.get("KUUPERF_TOKENS", 512))
    logname = os.environ.get("KUU_ENGINE_LOG") or os.path.join(
        tempfile.gettempdir(), f"kuu_perf_{name}.engine.log")
    log = open(logname, "w", encoding="utf-8", errors="replace")
    t_start = time.perf_counter()
    p = subprocess.Popen([exe, "--serve", "--pack", pack, "--max-context", "16384", "--eos", "1",
                          "--vision", "off", "--dspark", "off"] + rest,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1, encoding="utf-8", errors="replace")
    q, err_done = queue.Queue(), queue.Queue()
    load_s = [-1.0]   # the engine's own 'loaded in X s', the cold load this sweep pays eight times
    def pump_out():
        for l in p.stdout:
            q.put(l.rstrip("\n"))
        q.put(None)
    def pump_err():
        for line in p.stderr:
            log.write(line); log.flush()
            sys.stderr.write(line); sys.stderr.flush()
            m = re.search(r"loaded in ([\d.]+) s", line)
            if m:
                load_s[0] = float(m.group(1))
        p.stderr.close(); err_done.put(None)
    threading.Thread(target=pump_out, daemon=True).start()
    threading.Thread(target=pump_err, daemon=True).start()
    def drained():
        try:
            err_done.get(timeout=15)
        except queue.Empty:
            pass
    def result(verdict, prefill=-1.0, decodes=(), warm_tps=-1.0, note=""):
        print(f"RESULT {name} {verdict} load={load_s[0]:.1f} ready={ready_s[0]:.1f} "
              f"warm={warm_tps:.1f} prefill={prefill:.1f} "
              f"decode={','.join(f'{d:.1f}' for d in decodes) if decodes else '-'} note={note}")
    bad, ready_s = 0, [0.0]
    try:
        while True:
            l = q.get(timeout=READY_TMO)
            if l is None:
                print("engine ended before READY"); drained(); dump_tail(logname)
                result("FAIL", note="engine died before READY")
                return 1
            if l.startswith("READY"):
                ready_s[0] = time.perf_counter() - t_start
                break
        idstr = ",".join(map(str, ids))
        warm_tps, prefill, cold_prefill, decodes = -1.0, -1.0, -1.0, []
        for k in range(warm + gens):
            t_send = time.perf_counter(); t_first = None; n = 0
            p.stdin.write(f"GEN {ntok}  {idstr}\n"); p.stdin.flush()
            while True:
                l = q.get(timeout=GEN_TMO)
                if l is None:
                    print(f"gen {k + 1}: engine died"); bad += 1; break
                if l.startswith("T "):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
                if l.startswith("ERR"):
                    print(f"gen {k + 1}: {l[:120]}"); bad += 1; break
                if l.startswith("DONE"):
                    t_done = time.perf_counter()
                    if n != ntok:
                        print(f"gen {k + 1}: DONE carried {n} of {ntok} tokens"); bad += 1
                    prefill = (t_first - t_send) if t_first else -1.0
                    if k == 0:
                        cold_prefill = prefill   # RESULT reports the FIRST read (71.9 s on 32k, 2026-11-22),
                                                 # not the last GEN's 32767-reused 0.1 s the first version did
                    tps = (n - 1) / (t_done - t_first) if t_first and n > 1 else -1.0
                    if k < warm:
                        warm_tps = tps
                    else:
                        decodes.append(tps)
                    print(f"gen {k + 1} {'warm' if k < warm else 'meas'}: {n} tok | prefill {prefill:.1f} s"
                          f" | decode {tps:.1f} tok/s")
                    break
            if bad:
                break
        try:
            p.stdin.write("QUIT\n"); p.stdin.flush()
        except OSError:
            pass
        killed = False
        try:
            p.wait(timeout=120)
        except Exception:
            p.kill(); p.wait(); killed = True
        drained()
        if killed:
            print("engine did not exit after QUIT (killed after 120 s)"); bad += 1
        elif p.returncode != 0:
            print(f"engine exited with code {p.returncode}"); bad += 1
        eng_log = open(logname, encoding="utf-8", errors="replace").read()
        for s in POISON:
            if s in eng_log:
                print(f"engine log carries: {s}"); bad += 1
        if bad:
            dump_tail(logname)
            result("FAIL", cold_prefill, decodes, warm_tps, note="engine trouble")
            return 1
        if name == "all_exact":
            miss = [w for w, subs in EXACT.items() if not any(s in eng_log for s in subs)]
            if miss:
                result("FAIL", cold_prefill, decodes, warm_tps, note="no trace for:" + "".join(" " + m for m in miss))
                return 1
            result("PASS", cold_prefill, decodes, warm_tps, note="every switch of the exact set left its witness")
            return 0
        verdict, witness = "FAIL", ""
        for v, s in WIT[name]:
            if s in eng_log:
                verdict, witness = v, s
                break
        if verdict == "FAIL" and name == "uniform" and "RAM store:" not in eng_log:
            verdict, witness = "SKIP", "no RAM store was built: nothing to make uniform"
        note = f"witness: {witness}" if verdict != "FAIL" else "neither the witness nor a named reason in the log"
        result(verdict, cold_prefill, decodes, warm_tps, note=note)
        return 1 if verdict == "FAIL" else 0
    finally:
        log.close()

if __name__ == "__main__":
    sys.exit(main())
