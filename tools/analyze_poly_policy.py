"""Summarize numeric draft-policy traces from an explicitly selected synthetic benchmark run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def analyze(run):
    data = json.loads((run / 'engine/results.json').read_text(encoding='utf8'))
    blocks = []
    block = None
    for line in (run / 'engine/engine.stderr').read_text(encoding='utf8').splitlines():
        marker = 'strata-poly: draft-policy '
        if not line.startswith(marker):
            continue
        fields = line[len(marker):].split()
        if fields[0] == 'begin':
            block = {'rounds': []}
        elif fields[0] == 'end':
            if block is not None:
                blocks.append(block)
                block = None
        elif block is not None:
            values = dict(field.split('=', 1) for field in fields)
            costs = {}
            for entry in values.pop('costs').split(','):
                width, value = entry.split(':')
                cost, samples, age = map(float, value.split('/'))
                costs[int(width)] = dict(ms=cost, samples=samples, age=age)
            block['rounds'].append(dict(values, costs=costs))
    assert len(blocks) == len(data['requests']), (len(blocks), len(data['requests']))
    requests = []
    for request, block in zip(data['requests'], blocks):
        rounds = block['rounds']
        requests.append(dict(
            label=request['label'], decode_tps=request['decode_tps'],
            generated=len(request['generated_ids']), guessed_right=request['guessed_right'],
            selected=dict(Counter(int(r['chosen']) for r in rounds)),
            setup=dict(Counter(int(r['chosen']) for r in rounds if r['setup'] == '1')),
            first_costs=rounds[0]['costs'] if rounds else {},
            last_costs=rounds[-1]['costs'] if rounds else {},
            first_offer=int(rounds[0]['offered']) if rounds else 0,
            first_q=float(rounds[0]['q']) if rounds else 0,
        ))
    warm = [r['decode_tps'] for r in requests if r['label'].startswith('warm-repeat-')]
    return dict(run=str(run), warm_median=statistics.median(warm), requests=requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    # Preserve earlier analysis as well as the underlying experiment.
    with args.output.open('x', encoding='utf8') as output:
        json.dump(result, output, indent=2)
    print('warm_median', result['warm_median'])
    for r in result['requests']:
        print(r['label'], round(r['decode_tps'], 3), 'selected', r['selected'], 'setup', r['setup'])


if __name__ == '__main__':
    main()
