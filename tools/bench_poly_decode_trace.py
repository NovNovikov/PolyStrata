"""Capture one public synthetic chat decode with Nsight Systems. Exclusive result directories."""
import argparse
import json
import os
from pathlib import Path
import shutil
from bench_poly_decode_diagnosis import Diagnosis, prompts
from bench_poly_checkpoints import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--nsys', type=Path)
    p.add_argument('--phase-probe', type=Path, help='Standalone instrumented executable; no Nsight needed')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    args = list(config['args'])
    seed = a.work / 'expert-profile.txt'
    shutil.copyfile(a.profile, seed)
    args[args.index('--expert-profile') + 1] = str(seed.resolve())
    args[args.index('--session-dir') + 1] = ''
    if a.phase_probe:
        config['exe'] = str(a.phase_probe.resolve())
        args[args.index('--no-disk-vram-margin-mib') + 1] = '512'
        args += ['--dump-buffers']
    inputs, sources = prompts(config, root, 22446)
    if a.phase_probe:
        os.environ.pop('STRATA_PROFILE_DECODE', None)
    else:
        os.environ['STRATA_PROFILE_DECODE'] = '1'
    os.environ['STRATA_TRACE_DECODE_COSTS'] = '1'
    os.environ.pop('STRATA_FREEZE_EXPERTS', None)
    # The trace measures execution costs; rehashing cache payloads belongs to correctness tests.
    # Base harness enables verification; leave it enabled so this trace has the same checked path.
    with (a.output / 'provenance.json').open('x', encoding='utf8') as f:
        json.dump(dict(binary_sha256=sha(Path(config['exe'])), public_source_sha256=sources,
                       profiler=str(a.nsys), phase_probe=str(a.phase_probe), private_data_read=False,
                       note='Run this Python driver under nsys; it owns the engine protocol pipes.'), f, indent=2)
    e = Diagnosis(config, args, a.output / 'engine')
    try:
        e.ready()
        e.generate('varied-profile128', inputs['varied'], 128)
    finally:
        # Allow the engine to finish its shutdown; never kill an experiment on a timeout.
        e.stop_sampling.set()
        e.sampler.join(timeout=2)
        if e.p.poll() is None:
            e.send('QUIT')
        e.p.wait()
        e.err.close()
        e.record()
        assert e.p.returncode == 0, e.p.returncode
    print('DECODE_TRACE_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
