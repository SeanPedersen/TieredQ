"""Experiment 3 summary: recall vs p50/vectors Pareto fronts of our index (stage B, all settings replayed on held-out)
and Faiss IVF-SQ8 (stage A; held-out at calibration-frozen budgets, calibration curves listed separately)."""
import csv, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
STAGE_A = ROOT.parent / 'TieredQ-issue44-finelists-20261110/results'
FAISS_32K = ROOT.parent / 'TieredQ-issue44-assignment-20261110/results/faiss-32k'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def pareto(points):
    """Points not beaten by another with recall >= and p50 <=."""
    front = []
    for p in sorted(points, key=lambda p: (p['p50_us'], -p['recall'])):
        if not front or p['recall'] > front[-1]['recall']:
            front.append(p)
    return front


ours = []
for run in sorted((ROOT / 'results').iterdir()):
    if not (run / 'heldout.csv').exists():
        continue
    memory = json.loads((run / 'memory.json').read_text())
    for r in rows(run / 'heldout.csv'):
        ours.append(dict(system=f"ours {run.name} {r['routing']}", leaves=int(r['leaves']), routing=r['routing'], budget=int(r['metadata_cap']),
                         cap=int(r['vector_cap']), patience=int(r['patience']), recall=float(r['recall10']), mean_vectors=float(r['mean_vectors']),
                         p99_vectors=int(r['p99_vectors']), routing_us=float(r['mean_routing_us']), p50_us=float(r['p50_us']),
                         p99_us=float(r['p99_us']), rounds=float(r.get('mean_routing_rounds') or 0), build_s=memory['build_s'],
                         serving_anon=memory['serving_anon']))
faiss = []
for run in [FAISS_32K, *sorted(p for p in STAGE_A.iterdir() if p.name.startswith('ivf'))] if STAGE_A.exists() else [FAISS_32K]:
    if not (run / 'heldout.csv').exists():
        continue
    memory = json.loads((run / 'memory.json').read_text()) if (run / 'memory.json').exists() else {}
    name = 'faiss ' + (run.name if run != FAISS_32K else 'ivf-32k')
    for r in rows(run / 'heldout.csv'):
        faiss.append(dict(system=name, budget=int(r['budget']), recall=float(r['recall10']), mean_vectors=float(r['mean_distance_evaluations']),
                          p50_us=float(r['p50_us']), p99_us=float(r['p99_us']), build_s=float(r['build_s']), serving_anon=memory.get('serving_anon', 0)))
summary = dict(ours=ours, faiss=faiss, ours_front=pareto(ours), faiss_front=pareto(faiss))
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['# Experiment 3: list count, routing and patience vs Faiss (Qwen 8.84M x 1024 SQ8, 750 held-out queries)', '',
         '## Pareto front, ours (all settings replayed on held-out)', '',
         '| System | Exact R / budget | Cap | W | Recall | Mean vec | p99 vec | Routing ms | Rounds | p50 ms | p99 ms | Build s | Anon GB |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in summary['ours_front']:
    lines.append(f"| {p['system']} | {p['budget']} | {p['cap']} | {p['patience']} | {p['recall']:.4f} | {p['mean_vectors']:.0f} | {p['p99_vectors']} | "
                 f"{p['routing_us'] / 1e3:.2f} | {p['rounds']:.2f} | {p['p50_us'] / 1e3:.1f} | {p['p99_us'] / 1e3:.1f} | {p['build_s']:.0f} | {p['serving_anon'] / 1e9:.2f} |")
lines += ['', '## Faiss IVF-SQ8 held-out (calibration-frozen nprobe)', '',
          '| System | nprobe | Recall | Mean vec | p50 ms | p99 ms | Build s | Anon GB |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for p in sorted(faiss, key=lambda p: (p['system'], p['budget'])):
    lines.append(f"| {p['system']} | {p['budget']} | {p['recall']:.4f} | {p['mean_vectors']:.0f} | {p['p50_us'] / 1e3:.1f} | {p['p99_us'] / 1e3:.1f} | "
                 f"{p['build_s']:.0f} | {p['serving_anon'] / 1e9:.2f} |")
lines += ['', '## All of ours by model', '']
for name in sorted({p['system'] for p in ours}):
    lines += [f'### {name}', '', '| Budget | Cap | W | Recall | Mean vec | p99 vec | Routing ms | Rounds | p50 ms | p99 ms |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in sorted((p for p in ours if p['system'] == name), key=lambda p: (p['budget'], p['patience'] == 0, p['patience'], p['cap'])):
        lines.append(f"| {p['budget']} | {p['cap']} | {p['patience']} | {p['recall']:.4f} | {p['mean_vectors']:.0f} | {p['p99_vectors']} | "
                     f"{p['routing_us'] / 1e3:.2f} | {p['rounds']:.2f} | {p['p50_us'] / 1e3:.1f} | {p['p99_us'] / 1e3:.1f} |")
    lines.append('')
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
