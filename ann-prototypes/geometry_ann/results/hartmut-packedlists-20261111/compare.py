"""Summarizes the #53 A/B: storage and file size per variant; whether every serve returned the same ids and scores as
the first A serve; and per point, single and batch QPS of B over A, each from the mean of its two ABBA serves."""
import json
import sys
from pathlib import Path

import numpy as np

results = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
for dataset in ('arxiv', 'qwen'):
    builds = {v: json.loads((results / f'{dataset}-build-{v}/run.log').read_text().splitlines()[-1]) for v in 'AB'
              if (results / f'{dataset}-build-{v}/exit.json').exists()}
    serves = sorted(results.glob(f'{dataset}-serve-*'))
    if len(builds) < 2 or len(serves) < 4:
        continue
    a, b = builds['A'], builds['B']
    print(f"{dataset}: lists {a['lists']}, storage {a['storage_bytes'] / 2**20:.1f} -> {b['storage_bytes'] / 2**20:.1f} MiB "
          f"({b['storage_bytes'] / a['storage_bytes'] - 1:+.2%}), file {a['file_bytes'] / 2**20:.1f} -> "
          f"{b['file_bytes'] / 2**20:.1f} MiB, build {a['build_s']:.0f} / {b['build_s']:.0f} s")
    arrays = {s.name: np.load(s / 'serve.npz') for s in serves}
    reference = arrays[serves[0].name]
    for name, found in arrays.items():
        same = all(np.array_equal(found[key], reference[key]) for key in reference.files)
        print(f"  {name}: ids and scores {'identical' if same else 'DIFFER'} to {serves[0].name}")
    points = {s.name: json.loads((s / 'serve.json').read_text())['points'] for s in serves}
    for i, point in enumerate(points[serves[0].name]):
        by = {v: [points[s.name][i] for s in serves if s.name.endswith(v)] for v in 'AB'}
        mean = {v: {m: np.mean([p[m] for p in by[v]]) for m in ('single_qps', 'batch_qps')} for v in 'AB'}
        print(f"  k {point['k']:>3} nprobe {point['nprobe']:>4}: single {mean['A']['single_qps']:.0f} -> {mean['B']['single_qps']:.0f} "
              f"({mean['B']['single_qps'] / mean['A']['single_qps'] - 1:+.1%}, A spread "
              f"{abs(by['A'][0]['single_qps'] / by['A'][1]['single_qps'] - 1):.1%}), batch {mean['A']['batch_qps']:.0f} -> "
              f"{mean['B']['batch_qps']:.0f} ({mean['B']['batch_qps'] / mean['A']['batch_qps'] - 1:+.1%}, A spread "
              f"{abs(by['A'][0]['batch_qps'] / by['A'][1]['batch_qps'] - 1):.1%})")
