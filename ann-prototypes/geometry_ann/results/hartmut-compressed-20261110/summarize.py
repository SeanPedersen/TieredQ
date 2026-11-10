"""Compressed vs exact FP32 flat routing on held-out queries: routing time, p50 and vectors at equal recall.
References: FP32 flat curves from hartmut-patience-20261110 (fixed caps and patience; same labels, codes, queries),
interpolated log-linearly in cost along their monotone recall envelopes. Same-build FP32 controls are listed too."""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
FLAT = ROOT.parent / 'TieredQ-issue44-patience-20261110/results/lloyd-32k/heldout.csv'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def interpolate(recall, curve):
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
    return dict(routing=row['routing'], rerank=int(row['metadata_cap']), cap=int(row['vector_cap']), patience=int(row['patience']),
                recall=float(row['recall10']), mean_vectors=float(row['mean_vectors']), p99_vectors=int(row['p99_vectors']),
                routing_us=float(row['mean_routing_us']), p50_us=float(row['p50_us']), p99_us=float(row['p99_us']))


reference = [view(r) for r in rows(FLAT)]
curves = {policy: dict(vectors=[(r['mean_vectors'], r['recall']) for r in reference if bool(r['patience']) == patience],
                       p50=[(r['p50_us'], r['recall']) for r in reference if bool(r['patience']) == patience])
          for policy, patience in (('fixed', False), ('patience', True))}
summary, lines = {}, ['# Compressed (TieredQ) vs FP32 flat routing (Qwen 8.84M x 1024 SQ8, Lloyd 32K, 750 held-out queries)', '',
                      'Ratios compare against the FP32 flat curve of the same policy at equal held-out recall (> 1: compressed is cheaper).', '']
for run in sorted((ROOT / 'results').iterdir()):
    if not (run / 'heldout.csv').exists():
        continue
    memory = json.loads((run / 'memory.json').read_text())
    entries = []
    for row in rows(run / 'heldout.csv'):
        e = view(row); curve = curves['patience' if e['patience'] else 'fixed']
        vectors = interpolate(e['recall'], curve['vectors']); p50 = interpolate(e['recall'], curve['p50'])
        e['vector_ratio'] = vectors / e['mean_vectors'] if vectors else None
        e['p50_ratio'] = p50 / e['p50_us'] if p50 else None
        entries.append(e)
    entries.sort(key=lambda e: (e['routing'], e['rerank'], e['patience'] == 0, e['patience'], e['cap']))
    summary[run.name] = dict(memory=memory, heldout=entries)
    lines += [f"## {run.name}", '', f"Build {memory['build_s']:.0f} s (router {memory['router_s']:.1f} s, table {memory['router_bytes'] / 2**20:.1f} MiB); "
              f"serving anon {memory['serving_anon'] / 1e9:.2f} GB.", '',
              '| Routing | Exact R | Cap | W | Recall | Mean vec | p99 vec | Routing ms | p50 ms | p99 ms | Vec vs FP32 | p50 vs FP32 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: '' if v is None else f'{v:.2f}x'
    for e in entries:
        lines.append(f"| {e['routing']} | {e['rerank']} | {e['cap']} | {e['patience']} | {e['recall']:.4f} | {e['mean_vectors']:.0f} | "
                     f"{e['p99_vectors']} | {e['routing_us'] / 1e3:.2f} | {e['p50_us'] / 1e3:.1f} | {e['p99_us'] / 1e3:.1f} | "
                     f"{fmt(e['vector_ratio'])} | {fmt(e['p50_ratio'])} |")
    lines.append('')
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
