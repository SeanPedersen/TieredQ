"""Summarise patience vs fixed caps on held-out queries: frozen calibration picks plus equal-recall ratios.
Equal recall: the fixed-cap held-out curve (monotone envelope) is interpolated log-linearly in vectors."""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
RUN = ROOT / 'results' / 'lloyd-32k'
FAISS = ROOT.parent / 'TieredQ-issue44-assignment-20261110/results/faiss-32k/heldout.csv'


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def setting_key(row):
    return int(row['vector_cap']), int(row['patience'])


def fixed_vectors_at(recall, curve):
    """Vectors a fixed cap needs for `recall`; None if outside the measured envelope."""
    envelope, best = [], -1.0
    for vectors, value in sorted(curve):
        if value > best:
            envelope.append((vectors, value)); best = value
    for (v0, r0), (v1, r1) in zip(envelope, envelope[1:]):
        if r0 <= recall <= r1:
            share = (recall - r0) / (r1 - r0)
            return math.exp(math.log(v0) + share * (math.log(v1) - math.log(v0)))
    return None


heldout = {setting_key(r): r for r in rows(RUN / 'heldout.csv')}
fixed_curve = [(float(r['mean_vectors']), float(r['recall10'])) for k, r in heldout.items() if k[1] == 0]
fixed_latency = sorted((float(r['mean_vectors']), float(r['p50_us'])) for k, r in heldout.items() if k[1] == 0)
memory = json.loads((RUN / 'memory.json').read_text())


def view(row):
    return dict(cap=int(row['vector_cap']), patience=int(row['patience']), recall=float(row['recall10']),
                mean_vectors=float(row['mean_vectors']), p99_vectors=int(row['p99_vectors']), max_vectors=int(row['max_vectors']),
                mean_leaves=float(row['mean_leaves']), p99_leaves=int(row['p99_leaves']), p50_us=float(row['p50_us']),
                p99_us=float(row['p99_us']), routing_us=float(row['mean_routing_us']), scanning_us=float(row['mean_scanning_us']),
                patience_stop_rate=float(row['patience_stop_rate']))


curves = []
for key, row in sorted(heldout.items(), key=lambda item: (item[0][1] == 0, item[0][1], item[0][0])):
    entry = view(row)
    if entry['patience']:
        equal = fixed_vectors_at(entry['recall'], fixed_curve)
        entry['fixed_vectors_at_equal_recall'] = equal
        entry['vector_ratio'] = equal / entry['mean_vectors'] if equal else None
    curves.append(entry)

picks = []
for sel in rows(RUN / 'heldout.csv.selection.csv'):
    if sel['status'] == 'FIXED':
        continue
    entry = view(heldout[setting_key(sel)])
    picks.append(dict(target=float(sel['target']), policy=sel['policy'], status=sel['status'],
                      calibration_recall=float(sel['calibration_recall']),
                      calibration_mean_vectors=float(sel['calibration_mean_vectors']), heldout=entry))

faiss = [dict(nprobe=int(r['budget']), recall=float(r['recall10']), mean_vectors=float(r['mean_distance_evaluations']),
              p50_us=float(r['p50_us']), p99_us=float(r['p99_us'])) for r in rows(FAISS)] if FAISS.exists() else []
summary = dict(memory=memory, curves=curves, picks=picks, faiss_32k=faiss)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

lines = ['# Patience stopping vs fixed caps (Qwen 8.84M x 1024 SQ8, flat Lloyd 32K, 750 held-out queries)', '',
         f"Build {memory['build_s']:.0f} s; serving anon {memory['serving_anon'] / 1e9:.2f} GB, file {memory['serving_file'] / 1e9:.2f} GB.", '',
         '## Frozen calibration picks (min mean vectors at the calibration target)', '',
         '| Target | Policy | Status | Cap | W | Held-out recall | Mean vec | p99 vec | Mean leaves | p50 ms | p99 ms |',
         '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in sorted(picks, key=lambda p: (p['target'], p['policy'])):
    h = p['heldout']
    lines.append(f"| {p['target']} | {p['policy']} | {p['status']} | {h['cap']} | {h['patience']} | {h['recall']:.4f} | "
                 f"{h['mean_vectors']:.0f} | {h['p99_vectors']} | {h['mean_leaves']:.1f} | {h['p50_us'] / 1e3:.1f} | {h['p99_us'] / 1e3:.1f} |")
for f in faiss:
    lines.append(f"| ref | Faiss IVF-SQ8 32K nprobe={f['nprobe']} | | | | {f['recall']:.4f} | {f['mean_vectors']:.0f} | | {f['nprobe']} | "
                 f"{f['p50_us'] / 1e3:.1f} | {f['p99_us'] / 1e3:.1f} |")
lines += ['', '## Held-out curves', '',
          '| Cap | W | Recall | Mean vec | p99 vec | Max vec | Mean leaves | p99 leaves | p50 ms | p99 ms | Fixed vec @ equal recall | Ratio |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in curves:
    equal = c.get('fixed_vectors_at_equal_recall')
    ratio = c.get('vector_ratio')
    lines.append(f"| {c['cap']} | {c['patience']} | {c['recall']:.4f} | {c['mean_vectors']:.0f} | {c['p99_vectors']} | {c['max_vectors']} | "
                 f"{c['mean_leaves']:.1f} | {c['p99_leaves']} | {c['p50_us'] / 1e3:.1f} | {c['p99_us'] / 1e3:.1f} | "
                 f"{'' if equal is None else f'{equal:.0f}'} | {'' if ratio is None else f'{ratio:.2f}x'} |")
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
