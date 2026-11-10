"""Wait for this experiment only, validate small artifacts, and write the final local report."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
while True:
 status=json.loads((root/'status.json').read_text())
 if status['stage'] in ['complete','failed']:break
 time.sleep(30)
with (root/'summary.log').open('w') as out:
 subprocess.run([sys.executable,str(root/'summarize.py'),str(root)],stdout=out,stderr=subprocess.STDOUT,check=True)
summary=json.loads((root/'summary.json').read_text())
lines=['# Geometry experiment results','',f"Experiment status: **{status['stage']}**.",'',
 'Full-corpus Qwen runs use 250 calibration / 750 held-out queries, seed42, CPU2.',
 'Rows below are frozen calibration selections. Targets are not held-out guarantees.',
 'Latency is the median p50 of three query repeats; GB are decimal process RSS.',
 'SIFT sanity results use 50 calibration queries only and are kept in summary.json.','',
 '| Model | Target | Route | Bounds | Weight | Recall | p50 ms | Build min | Serving GB | Peak build GB |',
 '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
for model,entry in summary.items():
 for p in entry.get('frozen_heldout_points',[]):
  lines.append(f"| {model} | {p['target']} ({p['status']}) | {p['routing']} | {p['bounds']} | {p['weight']} | {p['recall10']:.4f} | {p['p50_us']/1000:.3f} | {p['build_s']/60:.2f} | {p['serving_rss_bytes']/1e9:.3f} | {p['build_peak_rss_bytes']/1e9:.3f} |")
lines+=['','Calibration ranking diagnostics on first50 queries:','',
 '| Model | Bounds | Weight | Mean true-cluster rank | Bounds above reference |',
 '|---|---|---:|---:|---:|']
for model,entry in summary.items():
 for d in entry['geometry_diagnostics']:
  lines.append(f"| {model} | {d['bounds']} | {d['weight']} | {d['mean_truth_cluster_rank']:.2f} | {d['bounds_above_reference']:.2f} |")
lines+=['','See README.md for the bound construction, numerical contract, query protocol, and limitations.',
 'summary.json retains calibration curves and query paging counters. A zero unsafe-member',
 'count covers the diagnostic sample; exhaustive small tests and the mathematical bound',
 'provide independent checks. No issue comment or external publication was made.','']
(root/'REPORT.md').write_text('\n'.join(lines))
