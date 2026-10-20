"""Sequential, real-weight CPU ISA comparison; all outputs are exclusively created."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--stock', type=Path, required=True)
    p.add_argument('--vnni', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sweeps', type=int, default=12)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for label, binary in [('stock', a.stock), ('vnni', a.vnni)]:
        command = [str(binary.resolve()), a.model, str(a.sweeps), str((a.output / (label + '.bin')).resolve())]
        print(label, 'started', flush=True)
        start = time.monotonic()
        with (a.output / (label + '.csv')).open('x') as out, (a.output / (label + '.stderr')).open('x') as err:
            r = subprocess.run(command, stdout=out, stderr=err)
        provenance[label] = dict(command=command, exit=r.returncode, elapsed=time.monotonic() - start,
                                 binary_sha256=sha(binary))
        print(label, 'completed', r.returncode, flush=True)
        if r.returncode:
            break
    if all((a.output / (label + '.bin')).exists() for label in ['stock', 'vnni']):
        provenance['byte_identical_input_and_all_outputs'] = (a.output / 'stock.bin').read_bytes() == (a.output / 'vnni.bin').read_bytes()
        provenance['reference_hashes'] = {label: sha(a.output / (label + '.bin')) for label in ['stock', 'vnni']}
    with (a.output / 'provenance.json').open('x') as f:
        json.dump(provenance, f, indent=2)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
