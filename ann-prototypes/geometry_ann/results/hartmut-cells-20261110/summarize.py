"""Validate cell-ceiling CSVs and write summary.json plus REPORT.md."""
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
COLUMNS = ['certified_clusters', 'certified_vectors', 'undecided_clusters', 'undecided_vectors',
           'pure_vectors_10', 'hybrid_vectors_10', 'mean_truth_rank', 'mean_truth_bound_ratio', 'bound_ns_per_leaf']


def p95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(.95 * (len(ordered) - 1)))]


def summarize(path):
    rows = list(csv.DictReader(path.open()))
    by_score = defaultdict(list)
    for row in rows:
        by_score[row['score']].append(row)
    queries = {row['query'] for row in rows}
    if any(len(v) != len(queries) for v in by_score.values()):
        raise ValueError(f'{path}: incomplete score rows')
    unsafe = sum(int(row['unsafe']) for row in rows)
    if unsafe:
        raise ValueError(f'{path}: {unsafe} unsafe bounds')
    centroid = by_score['centroid']
    result = dict(queries=len(queries),
                  oracle_vectors_10=dict(mean=statistics.mean(int(r['oracle_vectors_10']) for r in centroid),
                                         p95=p95([int(r['oracle_vectors_10']) for r in centroid])),
                  truth_clusters=statistics.mean(int(r['truth_clusters']) for r in centroid), scores={})
    for score, values in by_score.items():
        stats = {}
        for column in COLUMNS:
            numbers = [float(r[column]) for r in values]
            stats[column] = dict(mean=statistics.mean(numbers), median=statistics.median(numbers), p95=p95(numbers))
        stats['ranked'] = values[0]['ranked'] == '1'
        result['scores'][score] = stats
    return result


summary = {}
for model in sorted((ROOT / 'results').iterdir()):
    summary[model.name] = summarize(model / 'cells.csv')
    summary[model.name]['meta'] = json.loads((model / 'cells.csv.meta.json').read_text())
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

lines = ['# Cell-ceiling results', '',
         'Certified = leaves whose lower bound <= reference (max SQ8 distance of the 10 supplied true IDs):',
         'the leaves an exact search using that bound could not rule out. Means over queries; p95 in parentheses.',
         'Ranked columns scan whole leaves in bound order (ties by centroid distance) until 10/10 truth IDs are covered.',
         'Hull = convex hull of decoded members (ceiling for any hyperplane/convex metadata); undecided hulls',
         'hit the iteration budget. Bound ns are per leaf, measured inside a multi-threaded diagnostic.', '']
for model, result in summary.items():
    meta = result['meta']
    lines += [f'## {model}', '',
              f"Leaves {meta['leaves']}, build {meta['build_s']:.0f} s, slab build {meta['slab_build_s']:.1f} s "
              f"({meta['slab_bytes'] / 1e6:.1f} MB for {meta['directions']} directions), {result['queries']} queries. "
              f"Oracle vectors for 10/10: {result['oracle_vectors_10']['mean']:.0f} (p95 {result['oracle_vectors_10']['p95']}); "
              f"truth clusters {result['truth_clusters']:.2f}.", '',
              '| Score | Certified leaves | Certified vectors | Undecided leaves | Pure-order vectors | Hybrid vectors | Truth rank | Bound/ref | ns/leaf |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for score, s in result['scores'].items():
        f = lambda key, digits=0: f"{s[key]['mean']:.{digits}f} ({s[key]['p95']:.{digits}f})"
        ranked = s['ranked']
        lines.append(f"| {score} | {f('certified_clusters', 1)} | {f('certified_vectors')} | {f('undecided_clusters', 1)} | "
                     f"{f('pure_vectors_10') if ranked else '—'} | {f('hybrid_vectors_10') if ranked else '—'} | "
                     f"{s['mean_truth_rank']['mean']:.2f} | {s['mean_truth_bound_ratio']['mean']:.3f} | {s['bound_ns_per_leaf']['mean']:.0f} |")
    lines.append('')
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
