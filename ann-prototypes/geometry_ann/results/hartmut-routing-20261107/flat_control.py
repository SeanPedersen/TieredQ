"""Replay frozen flat settings with heap routing; three query repeats per immutable build."""
import csv
import subprocess
from pathlib import Path

MODELS = {
    'adaptive4': ['--variant', 'V2', '--split', 'kmeans', '--iterations', '4'],
    'fixed4': ['--variant', 'V0', '--children', '128', '--iterations', '4'],
    'fixed12': ['--variant', 'V0', '--children', '128', '--iterations', '12'],
}
for model in ["fixed4"]:
    variant = MODELS[model]
    with Path(f'results/{model}-selected.csv').open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r['routing'] == 'flat']
    settings = Path(f'results/{model}-flatheap-settings.csv')
    with settings.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows * 3)
    args = ['taskset', '-c', '2', 'flatheap/graph_bench', '--base', 'data/base.fvecs',
            '--queries', 'data/test_queries.fvecs', '--truth', 'data/test_truth.ivecs',
            '--nq', '1000', '--coarse', '64', '--leaves', '8192', '--target-size', '64',
            '--bounds', 'none', '--graph-degree', '8', '--settings', str(settings), *variant]
    with Path(f'results/{model}-flatheap.csv').open('x') as out, Path(f'results/{model}-flatheap.log').open('x') as log:
        subprocess.run(args, stdout=out, stderr=log, check=True)
