"""Refuse a Windows Ninja/MSVC poly build whose C++ objects lost project-header dependencies.

Localized /showIncludes output can silently yield VALID entries with zero dependencies.
Run this after a complete build and before recording model benchmarks.
"""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--ninja', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    ninja = args.ninja.resolve()
    manifest = (build / 'build.ninja').read_text(encoding='utf8')
    objects = [path for path in re.findall(r'^build ([^\n:]+\.cpp\.obj):', manifest, re.M)
               if path.replace('\\', '/').startswith('CMakeFiles/strata-poly.dir/src/poly/')]
    if not objects:
        raise SystemExit('FAIL: no MSVC strata-poly C++ objects found')
    failures = []
    for path in objects:
        header = 'cpu_experts.hpp' if path.replace('\\', '/').endswith('/cpu_experts.cpp.obj') else 'engine.hpp'
        result = subprocess.run([str(ninja), '-t', 'deps', path], cwd=build,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf8', errors='replace', check=True).stdout
        count = re.search(r'#deps (\d+)', result)
        headers = [line.strip().replace('\\', '/') for line in result.splitlines()[1:] if line.strip()]
        if not count or int(count[1]) == 0 or not any(h.endswith('/src/poly/' + header) for h in headers):
            failures.append(path)
            print('FAIL:', path, 'missing recorded', header, 'dependency')
        else:
            print('PASS:', path, 'deps=' + count[1], header, 'tracked')
    if failures:
        raise SystemExit(f'{len(failures)} inconsistent object(s); do not benchmark this executable')
    print(f'PASS: {len(objects)} C++ objects track required project headers')


if __name__ == '__main__':
    main()
