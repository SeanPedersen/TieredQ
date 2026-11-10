"""Aggregate repeats without selecting or retuning on held-out observations."""
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
FIELDS = ['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order']
METRICS = ['recall10', 'p50_us', 'p99_us', 'mean_vectors', 'mean_metadata',
           'mean_routing_us', 'mean_scanning_us', 'build_s', 'full_overhead_bytes',
           'build_peak_rss_bytes', 'graph_bytes']

def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))

summary = []
for decision in json.loads((ROOT / 'decisions.json').read_text()):
    model, chosen = decision['model'], decision['calibration']
    if chosen is None:
        summary.append(dict(model=model, target=decision['target'], routing=decision['routing'], status='NOT_REACHED_IN_CALIBRATION'))
        continue
    key = tuple(chosen[k] for k in FIELDS)
    heap_file = ROOT / f'{model}-flatheap.csv'
    use_heap = chosen['routing'] == 'flat' and heap_file.exists()
    if use_heap:
        observations = [row for row in records(heap_file) if tuple(row[k] for k in FIELDS) == key]
        old = [r for r in records(ROOT / f'{model}-heldout-0.csv') if tuple(r[k] for k in FIELDS) == key][0]
        for row in observations:
            for metric in ['recall10', 'mean_vectors', 'mean_metadata', 'mean_leaves', 'mean_skipped']:
                if row[metric] != old[metric]:
                    raise ValueError(f'Heap control changed {metric}')
    else:
        observations = [row for run in range(3) for row in records(ROOT / f'{model}-heldout-{run}.csv')
                        if tuple(row[k] for k in FIELDS) == key]
    if len(observations) != 3:
        raise ValueError(f'Expected exactly three repeats for {model}: {key}')
    if len({row['recall10'] for row in observations}) != 1:
        raise ValueError('Deterministic search changed recall across repeats')
    measured = {metric: median(float(row[metric]) for row in observations) for metric in METRICS}
    summary.append(dict(model=model, target=decision['target'], setting=dict(zip(FIELDS, key)),
                        calibration_recall=float(chosen['recall10']),
                        flat_control="heap" if use_heap else "full_sort" if chosen["routing"] == "flat" else None,
                        heldout_target_met=measured['recall10'] >= decision['target'], **measured))
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in summary:
    if 'p50_us' in row:
        print(row['model'], row['target'], row['setting']['routing'], row['recall10'],
              round(row['p50_us']), round(row['build_s'], 2), row['heldout_target_met'])

faiss = []
for budget in ['96', '192']:
    rows = [row for run in range(3) for row in records(ROOT / f'faiss-heldout-{run}.csv') if row['budget'] == budget]
    if len(rows) != 3:
        raise ValueError(f'Expected three frozen Faiss replays at nprobe={budget}')
    faiss.append(dict(budget=int(budget), **{k: median(float(row[k]) for row in rows) for k in
                 ['recall10', 'p50_us', 'p99_us', 'build_s', 'mean_distance_evaluations',
                  'serialized_overhead_bytes', 'process_peak_rss_bytes']}))
(ROOT / 'faiss-summary.json').write_text(json.dumps(faiss, indent=2) + '\n')
