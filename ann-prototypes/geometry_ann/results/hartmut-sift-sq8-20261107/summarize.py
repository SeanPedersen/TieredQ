"""Aggregate frozen held-out points; never choose settings using held-out recall/latency."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
args=parser.parse_args(); root=args.root.resolve()
results=root/'results' if (root/'results').is_dir() else root
manifest_path=root/'data/manifest.json' if (root/'data/manifest.json').exists() else root/'dataset-manifest.json'
manifest=json.loads(manifest_path.read_text())

def read(path):
    with path.open() as file:return list(csv.DictReader(file))

def mid(rows,key):return median(float(r[key]) for r in rows)

fields=['routing','bounds','weight','metadata_cap','vector_cap','scan_order']
summary=[]
for model in ['adaptive4','fixed4','fixed12']:
    path=results/f'{model}-heldout.csv'
    if not path.exists():continue
    rows=read(path); memory=read(Path(str(path)+'.memory.csv'))
    for selected in read(Path(str(path)+'.selection.csv')):
        key=tuple(selected[f] for f in fields)
        observations=[r for r in rows if tuple(r[f] for f in fields)==key]
        if len(observations)!=3 or {r['repeat'] for r in observations}!={'0','1','2'}:
            raise ValueError(f'Incomplete query repeats: {model}: {key}')
        if len({r['recall10'] for r in observations})!=1:raise ValueError('Recall varied across deterministic repeats')
        for r in observations:
            if int(r['max_vectors'])>int(r['vector_cap']) or int(r['max_metadata'])>int(r['metadata_cap']):
                raise ValueError('Hard query cap exceeded')
        mem=[r for r in memory if (r['routing'],r['metadata_cap'],r['vector_cap'])==(selected['routing'],selected['metadata_cap'],selected['vector_cap'])]
        if len(mem)!=3:raise ValueError('Missing memory repeats')
        metrics={k:mid(observations,k) for k in ['recall10','p50_us','p99_us','mean_vectors','mean_metadata','build_s',
                    'raw_bytes','original_fp32_bytes','full_overhead_bytes','build_peak_rss_bytes','graph_bytes']}
        metrics.update(serving_rss_bytes=mid(mem,'rss_bytes_after'),swap_bytes=max(int(r['swap_bytes_after']) for r in mem),
                       major_faults=[int(r['major_faults']) for r in mem])
        summary.append(dict(model=model,target=selected['target'],selection_status=selected['status'],
                            setting=dict(zip(fields,key)),**metrics,
                            estimated_owned_index_bytes=metrics['raw_bytes']+metrics['full_overhead_bytes']))

for reference in ['faiss-sq8','faiss-fp32']:
    path=results/f'{reference}-heldout.csv'
    if not path.exists():continue
    rows=read(path); selection=json.loads(path.with_suffix('.selection.json').read_text())
    for target,selected in selection['selection'].items():
        if selected is None:
            summary.append(dict(model=reference,target=target,selection_status='NOT_REACHED_IN_CALIBRATION'))
            continue
        budget=str(selected['budget']); observations=[r for r in rows if r['budget']==budget]
        if len(observations)!=3:raise ValueError(f'Missing Faiss repeats: {reference}/{budget}')
        metrics={k:mid(observations,k) for k in ['recall10','p50_us','p99_us','mean_distance_evaluations','build_s',
                    'raw_bytes','original_fp32_bytes','serialized_overhead_bytes','process_peak_rss_bytes','serving_rss_bytes']}
        summary.append(dict(model=reference,target=target,selection_status='SELECTED',nprobe=int(budget),**metrics,
                            swap_bytes=max(int(r['swap_bytes']) for r in observations),
                            major_faults=[int(r['major_faults']) for r in observations]))
    for budget in selection.get('diagnostic_budgets',[]):
        observations=[r for r in rows if int(r['budget'])==budget]
        if len(observations)!=3:raise ValueError('Missing diagnostic repeats')
        summary.append(dict(model=reference,target='diagnostic',selection_status='NOT_REACHED_DIAGNOSTIC',nprobe=budget,
                            **{k:mid(observations,k) for k in ['recall10','p50_us','build_s','mean_distance_evaluations','serving_rss_bytes']}))

for point in summary:
    if 'recall10' in point:
        try: point['heldout_target_met']=point['recall10']>=float(point['target'])
        except ValueError: point['heldout_target_met']=None
report=dict(dataset_shape=manifest['base_shape'],quantization=manifest['quantization'],
            preparation_seconds=manifest.get('preparation_seconds'),
            sq8_exhaustive_diagnostic=read(results/'sq8-oracle.csv'),
            fp32_exhaustive_diagnostic=read(results/'fp32-oracle.csv') if (results/'fp32-oracle.csv').exists() else [],points=summary)
(results/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for p in summary:
    if 'p50_us' in p:
        route=p.get('setting',{}).get('routing','ivf')
        print(p['model'],p['target'],route,p['selection_status'],p['recall10'],round(p['p50_us']),round(p['serving_rss_bytes']/1e6,1))
    else: print(p['model'],p['target'],p['selection_status'])
