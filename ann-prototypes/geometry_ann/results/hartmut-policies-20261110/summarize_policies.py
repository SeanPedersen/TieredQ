"""Spill and patience policies vs fixed-cap centroid order; writes summary.json and REPORT.md."""
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
summary, lines = {}, ['# Scan policies (Qwen 8.84M x 1024, 64x128 leaves, 250 calibration queries)', '',
                      'Coverage = mean fraction of the 10 true IDs inside scanned leaves (recall upper bound for exact rerank).',
                      'Spill-tau: rows whose relative margin (d_alt - d_own)/d_own < tau are also stored in their nearest',
                      'alternative leaf among the 32 nearest centroids; caps count the extra copies. Patience-W stops after W',
                      'consecutive leaves (centroid order) leave the top-10 unchanged, capped at the largest budget.', '']
for model in sorted((ROOT / 'results').iterdir()):
    rows = list(csv.DictReader((model / 'cells.csv').open()))
    by = defaultdict(list)
    for r in rows:
        by[r['score']].append(r)
    caps = [c for c in rows[0] if c.startswith('coverage_')]
    fixed = {score: {c: statistics.mean(float(r[c]) for r in v) for c in caps}
             for score, v in by.items() if score == 'centroid' or score.startswith('spill-')}
    spill_fraction = {score: float(v[0]['spill_fraction']) for score, v in by.items() if score.startswith('spill-')}
    patience = {score: dict(vectors=statistics.mean(float(r['adaptive_vectors']) for r in v),
                            leaves=statistics.mean(float(r['adaptive_leaves']) for r in v),
                            coverage=statistics.mean(float(r['adaptive_coverage']) for r in v))
                for score, v in by.items() if score.startswith('patience-')}
    summary[model.name] = dict(fixed=fixed, spill_fraction=spill_fraction, patience=patience)
    lines += [f'## {model.name}', '', '| Policy | Extra vectors | ' + ' | '.join(c.removeprefix('coverage_') for c in caps) + ' |',
              '|---|---:|' + '---:|' * len(caps)]
    for score, values in fixed.items():
        extra = f"{spill_fraction.get(score, 0):.1%}"
        lines.append(f'| {score} | {extra} | ' + ' | '.join(f'{values[c]:.4f}' for c in caps) + ' |')
    lines += ['', '| Policy | Mean leaves | Mean vectors | Coverage |', '|---|---:|---:|---:|']
    for score, values in patience.items():
        lines.append(f"| {score} | {values['leaves']:.1f} | {values['vectors']:.0f} | {values['coverage']:.4f} |")
    lines.append('')
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
