"""Summarize completed calibration curves and frozen held-out points, without retuning."""
import csv,json,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean,median
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
report={}
keys=['routing','bounds','weight','metadata_cap','vector_cap','scan_order']
for folder in sorted((root/'results').iterdir()):
 if not folder.is_dir() or not (folder/'exit.json').exists():continue
 if json.loads((folder/'exit.json').read_text())['exit_code']!=0:raise ValueError(folder)
 cal=read(folder/'calibration.csv')
 for r in cal:
  if int(r['max_vectors'])>int(r['vector_cap']) or int(r['max_metadata'])>int(r['metadata_cap']):raise ValueError('hard cap exceeded')
 diagnostics=read(folder/'geometry-diagnostics.csv'); grouped=defaultdict(list)
 for r in diagnostics:
  if int(r['unsafe_members']):raise ValueError('unsafe geometry bound')
  grouped[(r['bounds'],r['weight'])].append(r)
 shape=[]
 for (mode,weight),rows in grouped.items():
  shape.append(dict(bounds=mode,weight=float(weight),nq=len(rows),
    **{k:mean(float(r[k]) for r in rows) for k in ['positive_bounds','bounds_above_reference','mean_truth_cluster_rank','max_truth_cluster_rank','mean_bound_ratio']}))
 entry=dict(calibration=cal,geometry_diagnostics=shape)
 held=folder/'heldout.csv'
 if held.exists():
  rows=read(held);memory=read(Path(str(held)+'.memory.csv'))
  if len(rows)!=len(memory):raise ValueError('memory observation count')
  # Both files are emitted in the same loop. Correlate by ordinal, because
  # historical memory CSVs omit mode/weight and can share a route/cap key.
  pairs=list(zip(rows,memory,strict=True)); selected=[]
  for r,m in pairs:
   for k in ['repeat','routing','metadata_cap','vector_cap']:
    if r[k]!=m[k]:raise ValueError('memory observation ordering')
  for s in read(Path(str(held)+'.selection.csv')):
   obs=[(r,m) for r,m in pairs if all(r[k]==s[k] for k in keys)]
   if len(obs)!=3 or {r['repeat'] for r,m in obs}!={'0','1','2'}:raise ValueError('incomplete repeats')
   if len({r['recall10'] for r,m in obs})!=1:raise ValueError('nondeterministic recall')
   for r,m in obs:
    if int(r['max_vectors'])>int(r['vector_cap']) or int(r['max_metadata'])>int(r['metadata_cap']):raise ValueError('heldout hard cap')
   point=dict(s)
   point.update({k:median(float(r[k]) for r,m in obs) for k in ['recall10','p50_us','p99_us','mean_vectors','mean_pruned','mean_positive_bounds','build_s','build_peak_rss_bytes','full_overhead_bytes','active_bounds_payload_bytes','shared_basis_s','shared_ranges_s']})
   point.update(serving_rss_bytes=median(int(m['rss_bytes_after']) for r,m in obs),
     swap_bytes=max(int(m['swap_bytes_after']) for r,m in obs),major_faults=[int(m['major_faults']) for r,m in obs])
   selected.append(point)
  entry['frozen_heldout_points']=selected
 report[folder.name]=entry
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for name,entry in report.items():
 print(name)
 for r in entry['calibration']:
  if r['routing']=='flat':print(' calibration',r['bounds'],r['weight'],r['vector_cap'],r['recall10'],r['p50_us'])
 for r in entry.get('frozen_heldout_points',[]):print(' heldout',r['target'],r['routing'],r['bounds'],r['weight'],r['recall10'],r['p50_us'],r['status'])
