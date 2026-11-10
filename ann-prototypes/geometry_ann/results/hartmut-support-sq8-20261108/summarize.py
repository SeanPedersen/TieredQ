"""Validate support/oracle artifacts and summarize frozen selections without retuning."""
import csv,json,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean,median
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent

def read(p):
 with p.open() as f:return list(csv.DictReader(f))

def p95(values):return sorted(values)[max(0,(95*len(values)+99)//100-1)]

def key(r):return tuple(r[k] for k in ['routing','bounds','weight','metadata_cap','vector_cap','scan_order'])
report={}; lines=['# Edge supports and oracle coverage','',
 'Geometry and coverage diagnostics use the first 50 calibration queries.',
 'Oracle costs cover the supplied FP32 top-10 IDs by scanning whole clusters.',
 'They are optimistic coverage diagnostics, not guaranteed SQ8 search recall.',
 'Ranked costs scan a complete cluster prefix without skipping oversized clusters.',
 'Witness ordering is diagnostic only: its real-vector evaluations are not part',
 'of the timed search configurations, and no witness latency win is claimed.','']
for folder in sorted((root/'results').iterdir()):
 if not folder.is_dir() or not (folder/'exit.json').exists():continue
 if json.loads((folder/'exit.json').read_text())['exit_code']:raise ValueError(f'{folder}: failed stage')
 cal=read(folder/'calibration.csv');diag=read(folder/'geometry-diagnostics.csv')
 for r in cal:
  if int(r['max_vectors'])>int(r['vector_cap']) or int(r['max_metadata'])>int(r['metadata_cap']):raise ValueError('hard cap exceeded')
 groups=defaultdict(list)
 for r in diag:
  if int(r['unsafe_members']):raise ValueError('unsafe geometry')
  for n in [9,10]:
   if int(r[f'oracle_vectors_{n}'])>int(r[f'ranked_vectors_{n}']):raise ValueError('oracle consistency')
  groups[(r['bounds'],r['weight'],r['witness_ordering'])].append(r)
 diagnostic=[]
 for (mode,weight,witness),rows in groups.items():
  point=dict(bounds=mode,weight=float(weight),witness_ordering=bool(int(witness)),nq=len(rows))
  for k in ['truth_clusters','positive_bounds','bounds_above_reference','mean_truth_cluster_rank','max_truth_cluster_rank','mean_bound_ratio','oracle_vectors_9','oracle_vectors_10','ranked_vectors_9','ranked_vectors_10','witness_evaluations']:
   values=[float(r[k]) for r in rows];point[k]=dict(mean=mean(values),p50=median(values),p95=p95(values))
  diagnostic.append(point)
 entry=dict(calibration=cal,diagnostics=diagnostic)
 lines += [f'## {folder.name}','','| Score | Weight | Witness | Mean true-cluster rank | Mean oracle vectors for 10 | Mean ranked vectors for 10 | p95 ranked vectors for 10 |',
 '|---|---:|---|---:|---:|---:|---:|']
 for d in diagnostic:
  lines.append(f"| {d['bounds']} | {d['weight']} | {d['witness_ordering']} | {d['mean_truth_cluster_rank']['mean']:.2f} | {d['oracle_vectors_10']['mean']:.0f} | {d['ranked_vectors_10']['mean']:.0f} | {d['ranked_vectors_10']['p95']:.0f} |")
 held=folder/'heldout.csv'
 if held.exists():
  rows=read(held);memory=read(Path(str(held)+'.memory.csv'));pairs=list(zip(rows,memory,strict=True))
  for r,m in pairs:
   if any(r[k]!=m[k] for k in ['repeat','routing','metadata_cap','vector_cap']):raise ValueError('memory alignment')
   if int(r['max_vectors'])>int(r['vector_cap']) or int(r['max_metadata'])>int(r['metadata_cap']):raise ValueError('held-out hard cap exceeded')
  points=[]
  for selection in read(Path(str(held)+'.selection.csv')):
   obs=[(r,m) for r,m in pairs if key(r)==key(selection)]
   if len(obs)!=3 or {r['repeat'] for r,m in obs}!={'0','1','2'}:raise ValueError('incomplete repeats')
   if len({r['recall10'] for r,m in obs})!=1:raise ValueError('nondeterministic recall')
   point=dict(selection)
   for k in ['recall10','p50_us','p99_us','mean_vectors','mean_pruned','mean_positive_bounds','mean_projection_improvements','build_s','build_peak_rss_bytes','full_overhead_bytes','active_bounds_payload_bytes','support_build_s']:
    point[k]=median(float(r[k]) for r,m in obs)
   point.update(serving_rss_bytes=median(int(m['rss_bytes_after']) for r,m in obs),swap_bytes=max(int(m['swap_bytes_after']) for r,m in obs),major_faults=[int(m['major_faults']) for r,m in obs])
   points.append(point)
  entry['heldout']=points
  lines+=['','Selections frozen independently for each bounds family. Recall differs; no exact matched-recall claim.','',
   '| Target | Bounds | Weight | Status | Recall | p50 ms | Build min | Serving GB | Peak build GB |',
   '|---:|---|---:|---|---:|---:|---:|---:|---:|']
  for p in points:
   lines.append(f"| {p['target']} | {p['bounds']} | {p['weight']} | {p['status']} | {p['recall10']:.4f} | {p['p50_us']/1000:.3f} | {p['build_s']/60:.2f} | {p['serving_rss_bytes']/1e9:.3f} | {p['build_peak_rss_bytes']/1e9:.3f} |")
 report[folder.name]=entry
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Validated',', '.join(report))
