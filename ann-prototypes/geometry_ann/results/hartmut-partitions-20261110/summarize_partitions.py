"""Summarize finer-partition graph_bench runs (calibration curves and frozen held-out selections) into REPORT-partitions.md."""
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])


def rows(path):
    return list(csv.DictReader(path.open()))


summary, lines = {}, ['# Finer partitions (Qwen 8.84M x 1024, INT8)', '',
                      'Calibration: 250 queries. Held-out: 750 disjoint queries, 3 repeats, settings frozen on calibration.',
                      'Baseline for comparison (64x128 = 8192 leaves, support study): flat 95.57% at 27.25 ms; graph 95.04% at 25.8 ms; best 97.65% at 61.5 ms.', '']
for run in sorted((ROOT / 'results').iterdir()):
    calibration = rows(run / 'calibration.csv')
    heldout = rows(run / 'heldout.csv')
    selection = rows(run / 'heldout.csv.selection.csv')
    memory = rows(run / 'heldout.csv.memory.csv')
    first = calibration[0]
    build = dict(leaves=int(first['leaves']), build_s=float(first['build_s']), clustering_s=float(first['clustering_s']),
                 graph_s=float(first['graph_s']), peak_build_gb=int(first['build_peak_rss_bytes']) / 1e9,
                 serving_gb=max(int(m['rss_bytes_after']) for m in memory) / 1e9,
                 swap_gb=max(int(m['swap_bytes_after']) for m in memory) / 1e9,
                 major_faults=sum(int(m['major_faults']) for m in memory))
    repeats = defaultdict(list)
    for r in heldout:
        repeats[(r['routing'], r['bounds'], r['weight'], r['metadata_cap'], r['vector_cap'])].append(r)
    if any(len(v) != 3 for v in repeats.values()):
        raise ValueError(f'{run.name}: incomplete held-out repeats')
    held = {k: dict(recall=statistics.mean(float(r['recall10']) for r in v),
                    p50_ms=statistics.median(float(r['p50_us']) for r in v) / 1e3,
                    routing_ms=statistics.median(float(r['mean_routing_us']) for r in v) / 1e3,
                    vectors=statistics.mean(float(r['mean_vectors']) for r in v)) for k, v in repeats.items()}
    summary[run.name] = dict(build=build, calibration=calibration, selection=selection,
                             heldout={'/'.join(k): v for k, v in held.items()})
    lines += [f'## {run.name}', '',
              f"Leaves {build['leaves']}, build {build['build_s'] / 60:.1f} min (clustering {build['clustering_s'] / 60:.1f}, graph {build['graph_s']:.0f} s), "
              f"peak build {build['peak_build_gb']:.2f} GB, serving RSS {build['serving_gb']:.2f} GB, swap {build['swap_gb']:.2f} GB, major faults {build['major_faults']}.", '',
              '| Routing | Metadata | Vector cap | Calib recall | Calib p50 ms | Routing ms | Mean vectors |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in calibration:
        lines.append(f"| {r['routing']} | {r['metadata_cap']} | {r['vector_cap']} | {float(r['recall10']):.4f} | {float(r['p50_us']) / 1e3:.2f} | "
                     f"{float(r['mean_routing_us']) / 1e3:.2f} | {float(r['mean_vectors']):.0f} |")
    lines += ['', '| Target | Status | Routing | Metadata | Cap | Held-out recall | p50 ms | Routing ms |', '|---:|---|---|---:|---:|---:|---:|---:|']
    for s in selection:
        h = held.get((s['routing'], s['bounds'], s['weight'], s['metadata_cap'], s['vector_cap']))
        if h:
            lines.append(f"| {s['target']} | {s['status']} | {s['routing']} | {s['metadata_cap']} | {s['vector_cap']} | "
                         f"{h['recall']:.4f} | {h['p50_ms']:.2f} | {h['routing_ms']:.2f} |")
    lines.append('')
(ROOT / 'summary-partitions.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'REPORT-partitions.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
