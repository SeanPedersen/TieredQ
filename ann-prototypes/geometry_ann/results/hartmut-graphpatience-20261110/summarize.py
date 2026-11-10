"""Graph routing (fixed caps and patience) vs flat routing with patience, on held-out queries at equal recall.
Flat reference curves come from hartmut-patience-20261110 (same index labels, codes and queries); values are interpolated
log-linearly in the cost along the monotone recall envelope of each flat curve."""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
FLAT = ROOT.parent / 'TieredQ-issue44-patience-20261110/results/lloyd-32k/heldout.csv'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def interpolate(recall, curve):
    """Cost a curve of (cost, recall) points needs for `recall`; None outside its measured envelope."""
    envelope, best = [], -1.0
    for cost, value in sorted(curve):
        if value > best:
            envelope.append((cost, value)); best = value
    for (c0, r0), (c1, r1) in zip(envelope, envelope[1:]):
        if r0 <= recall <= r1:
            share = (recall - r0) / (r1 - r0)
            return math.exp(math.log(c0) + share * (math.log(c1) - math.log(c0)))
    return None


def view(row):
    return dict(budget=int(row['metadata_cap']), cap=int(row['vector_cap']), patience=int(row['patience']),
                recall=float(row['recall10']), mean_vectors=float(row['mean_vectors']), p99_vectors=int(row['p99_vectors']),
                mean_leaves=float(row['mean_leaves']), routing_us=float(row['mean_routing_us']),
                p50_us=float(row['p50_us']), p99_us=float(row['p99_us']))


flat = [view(r) for r in rows(FLAT)]
flat_patience = [f for f in flat if f['patience']]
flat_fixed = [f for f in flat if not f['patience']]
references = {name: dict(vectors=[(f['mean_vectors'], f['recall']) for f in curve], p50=[(f['p50_us'], f['recall']) for f in curve])
              for name, curve in (('flat_patience', flat_patience), ('flat_fixed', flat_fixed))}
summary, lines = {}, ['# Graph routing + patience vs flat routing (Qwen 8.84M x 1024 SQ8, Lloyd 32K, 750 held-out queries)', '',
                      'Ratios > 1 mean graph is cheaper than flat+patience at equal held-out recall.', '']
for run in sorted((ROOT / 'results').iterdir()):
    if not (run / 'heldout.csv').exists():
        continue
    memory = json.loads((run / 'memory.json').read_text())
    entries = []
    for row in rows(run / 'heldout.csv'):
        entry = view(row)
        for name, reference in references.items():
            vectors = interpolate(entry['recall'], reference['vectors']); p50 = interpolate(entry['recall'], reference['p50'])
            entry[f'{name}_vector_ratio'] = vectors / entry['mean_vectors'] if vectors else None
            entry[f'{name}_p50_ratio'] = p50 / entry['p50_us'] if p50 else None
        entries.append(entry)
    entries.sort(key=lambda e: (e['patience'] == 0, e['budget'], e['patience'], e['cap']))
    summary[run.name] = dict(memory=memory, heldout=entries)
    lines += [f"## {run.name}", '', f"Build {memory['build_s']:.0f} s (graph {memory['graph_s']:.0f} s); serving anon {memory['serving_anon'] / 1e9:.2f} GB.", '',
              '| Budget | Cap | W | Recall | Mean vec | p99 vec | Routing ms | p50 ms | p99 ms | Vec vs flat+pat | p50 vs flat+pat |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for e in entries:
        fmt = lambda v: '' if v is None else f'{v:.2f}x'
        lines.append(f"| {e['budget']} | {e['cap']} | {e['patience']} | {e['recall']:.4f} | {e['mean_vectors']:.0f} | {e['p99_vectors']} | "
                     f"{e['routing_us'] / 1e3:.1f} | {e['p50_us'] / 1e3:.1f} | {e['p99_us'] / 1e3:.1f} | "
                     f"{fmt(e['flat_patience_vector_ratio'])} | {fmt(e['flat_patience_p50_ratio'])} |")
    lines.append('')
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
