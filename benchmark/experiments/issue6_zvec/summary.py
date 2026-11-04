"""Summarize ivf_vs_zvec.py TSVs: equal-memory curves and the fastest config per codec at each recall@10 target."""
import csv
import sys
from collections import defaultdict

TARGETS = (0.80, 0.90, 0.95, 0.97, 0.98, 0.99)
FIELDS = ('codec', 'bits', 'lists', 'nprobe', 'index_bits', 'recall', 'qps', 'build_s')


def rows(paths):
    for path in paths:
        with open(path) as handle:
            for raw in csv.reader(handle, delimiter='\t'):
                row = dict(zip(FIELDS, raw))
                yield {**row, 'nprobe': int(row['nprobe']), 'index_bits': int(row['index_bits']),
                       'recall': float(row['recall']), 'qps': float(row['qps']), 'build_s': float(row['build_s'])}


def config(row):
    return f"{row['bits']} b/dim, nprobe {row['nprobe']}"


def equal_memory(table):
    by_level = defaultdict(list)
    for row in table:
        by_level[round(float(row['bits']))].append(row)
    print('| Level | Codec | Index bits/vec | Build s | ' + ' | '.join(f'nprobe {n}' for n in sorted({r['nprobe'] for r in table})) + ' |')
    print('|---|---|---:|---:|' + '---:|' * len({r['nprobe'] for r in table}))
    for level in sorted(by_level):
        codecs = defaultdict(list)
        for row in by_level[level]:
            codecs[row['codec']].append(row)
        for codec, cells in sorted(codecs.items()):
            cells.sort(key=lambda r: r['nprobe'])
            values = ' | '.join(f"{r['recall']:.3f} / {r['qps']:,.0f}" for r in cells)
            print(f"| {level} | {codec} | {cells[0]['index_bits']:,} | {cells[0]['build_s']:.0f} | {values} |")


def fastest_per_target(table):
    print('\n| R@10 target | Codec | Fastest config | Index bits/vec | R@10 | QPS 1t·b1 | Least-RAM config | Index bits/vec | R@10 | QPS |')
    print('|---|---|---|---:|---:|---:|---|---:|---:|---:|')
    for target in TARGETS:
        for codec in sorted({r['codec'] for r in table}):
            hits = [r for r in table if r['codec'] == codec and r['recall'] >= target]
            if not hits:
                print(f'| {target} | {codec} | - | | | | - | | | |')
                continue
            fast = max(hits, key=lambda r: r['qps'])
            small = min(hits, key=lambda r: (r['index_bits'], -r['qps']))
            print(f"| {target} | {codec} | {config(fast)} | {fast['index_bits']:,} | {fast['recall']:.3f} | "
                  f"{fast['qps']:,.0f} | {config(small)} | {small['index_bits']:,} | {small['recall']:.3f} | "
                  f"{small['qps']:,.0f} |")


def main():
    table = list(rows(sys.argv[1:]))
    equal_memory(table)
    fastest_per_target(table)


if __name__ == '__main__':
    main()
