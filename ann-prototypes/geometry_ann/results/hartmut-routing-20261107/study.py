"""Run from the experiment root; freeze configurations on calibration before test replay."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=False)
FIELDS = ['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order']
TARGETS = [0.95, 0.99, 0.995]
MODELS = {
    'adaptive4': ['--variant', 'V2', '--split', 'kmeans', '--iterations', '4'],
    'fixed4': ['--variant', 'V0', '--children', '128', '--iterations', '4'],
    'fixed12': ['--variant', 'V0', '--children', '128', '--iterations', '12'],
}

def write_settings(path, rows):
    with path.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)

settings = []
for cap in [10000, 20000, 35000, 50000]:
    for route, budgets in [('graph', [512, 1024, 2048, 4096]), ('flat', [8256]), ('hierarchy', [8256])]:
        for budget in budgets:
            settings.append(dict(routing=route, bounds='none', weight=0, metadata_cap=budget,
                                 vector_cap=cap, scan_order='centroid'))
write_settings(OUT / 'calibration-settings.csv', settings)

def bench(model, split, settings, output):
    args = ['taskset', '-c', '2', str(ROOT / 'build/graph_bench'),
            '--base', 'data/base.fvecs', '--queries', f'data/{split}_queries.fvecs',
            '--truth', f'data/{split}_truth.ivecs', '--nq', '250' if split == 'calibration' else '1000',
            '--coarse', '64', '--leaves', '8192', '--target-size', '64', '--bounds', 'none',
            '--graph-degree', '8', '--settings', str(settings), *MODELS[model]]
    with output.open('x') as out, output.with_suffix('.log').open('x') as log:
        subprocess.run(args, stdout=out, stderr=log, check=True)

# Every choice is frozen before any held-out measurements.
decisions = []
for model in MODELS:
    calibration = OUT / f'{model}-calibration.csv'
    bench(model, 'calibration', OUT / 'calibration-settings.csv', calibration)
    rows = list(csv.DictReader(calibration.open()))
    chosen = {}
    for target in TARGETS:
        for route in ['graph', 'flat', 'hierarchy']:
            eligible = [r for r in rows if r['routing'] == route and float(r['recall10']) >= target]
            best = min(eligible, key=lambda r: float(r['p50_us'])) if eligible else None
            decisions.append(dict(model=model, target=target, routing=route, calibration=best))
            if best:
                chosen[tuple(best[k] for k in FIELDS)] = best
    write_settings(OUT / f'{model}-selected.csv', chosen.values())
(OUT / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')

for run in range(3):
    for model in (list(MODELS) if run % 2 == 0 else list(reversed(MODELS))):
        bench(model, 'test', OUT / f'{model}-selected.csv', OUT / f'{model}-heldout-{run}.csv')
    args = ['taskset', '-c', '2', 'uv', 'run', '--python', '3.12', 'source/compare_faiss.py',
            '--base', 'data/base.fvecs', '--queries', 'data/calibration_queries.fvecs',
            '--truth', 'data/calibration_truth.ivecs', '--nq', '250',
            '--nlist', '8192', '--sample', '327680', '--variant', 'ivf',
            '--budgets', '32,64,96,128,192,256',
            '--evaluation-queries', 'data/test_queries.fvecs', '--evaluation-truth', 'data/test_truth.ivecs',
            '--evaluation-output', str(OUT / f'faiss-heldout-{run}.csv'), '--evaluation-nq', '1000']
    with (OUT / f'faiss-calibration-{run}.csv').open('x') as out, (OUT / f'faiss-{run}.log').open('x') as log:
        subprocess.run(args, stdout=out, stderr=log, check=True)
