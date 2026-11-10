"""Held-out scan counts, recall, build time and memory for ours vs Faiss; writes summary.json and REPORT.md."""
import csv, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
GB = 1e9


def ours(run):
    selection = list(csv.DictReader((run / 'heldout.csv.selection.csv').open()))
    heldout = {(r['routing'], r['metadata_cap'], r['vector_cap']): r for r in csv.DictReader((run / 'heldout.csv').open())}
    rows = []
    for s in selection:
        h = heldout[(s['routing'], s['metadata_cap'], s['vector_cap'])]
        rows.append(dict(target=s['target'], status=s['status'], setting=f"{s['routing']} m{s['metadata_cap']} cap{s['vector_cap']}",
                         recall=float(h['recall10']), clusters=float(h['mean_leaves']), vectors=float(h['mean_vectors']),
                         p50_ms=float(h['p50_us']) / 1e3, routing_ms=float(h['mean_routing_us']) / 1e3))
    return rows


def faiss(run):
    selection = json.loads((run / 'heldout.selection.json').read_text())['selection']
    heldout = {r['budget']: r for r in csv.DictReader((run / 'heldout.csv').open())}
    calibration = list(csv.DictReader((run / 'calibration.csv').open()))
    rows = []
    for target, chosen in selection.items():
        status = 'SELECTED'
        if chosen is None:  # mirror our protocol: report the best calibration budget as a diagnostic
            chosen = max(calibration, key=lambda r: (float(r['recall10']), -float(r['p50_us'])))
            status = 'NOT_REACHED_DIAGNOSTIC'
        h = heldout[str(chosen['budget'])]
        rows.append(dict(target=target, status=status, setting=f"nprobe{chosen['budget']}", recall=float(h['recall10']),
                         clusters=float(chosen['budget']), vectors=float(h['mean_distance_evaluations']),
                         p50_ms=float(h['p50_us']) / 1e3, routing_ms=None))
    return rows


summary = {}
for run in sorted((ROOT / 'results').iterdir()):
    if not (run / 'memory.json').exists() or not (run / 'heldout.csv').exists():
        continue
    rows = faiss(run) if run.name.startswith('faiss') else ours(run)
    summary[run.name] = dict(memory=json.loads((run / 'memory.json').read_text()), selections=rows)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

lines = ['# Leaf assignment vs Faiss IVF-SQ8 (Qwen 8.84M x 1024, INT8)', '',
         'Held-out: 750 queries; settings frozen on 250 calibration queries. Builds: 32 threads; search: 1 thread pinned.',
         'Clusters = mean leaves scanned (ours) or nprobe (Faiss). Vectors = mean vectors scanned / distance evaluations.', '',
         '| Model | Target | Status | Setting | Recall | Clusters | Vectors | p50 ms | Routing ms |', '|---|---:|---|---|---:|---:|---:|---:|---:|']
for name, result in summary.items():
    seen = set()
    for r in result['selections']:
        key = (r['target'], r['setting'])
        if key in seen:
            continue
        seen.add(key)
        routing = '—' if r['routing_ms'] is None else f"{r['routing_ms']:.2f}"
        lines.append(f"| {name} | {r['target']} | {r['status']} | {r['setting']} | {r['recall']:.4f} | {r['clusters']:.1f} | "
                     f"{r['vectors']:.0f} | {r['p50_ms']:.2f} | {routing} |")
lines += ['', '| Model | Build s | Reassign s | Moved vectors | Leaves | Build peak anon GB | Build peak file GB | Build HWM GB | Serving anon GB | Serving RSS GB |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name, result in summary.items():
    m = result['memory']
    lines.append(f"| {name} | {m['build_s']:.0f} | {m.get('reassign_s', 0):.0f} | {m.get('reassigned_vectors', '—')} | {m.get('leaves', m.get('nlist'))} | "
                 f"{m['build_peak_anon'] / GB:.2f} | {m['build_peak_file'] / GB:.2f} | {m['build_hwm'] / GB:.2f} | "
                 f"{m['serving_anon'] / GB:.2f} | {m['serving_rss'] / GB:.2f} |")
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
