"""Cheap progress snapshot; no dataset/index loads or changes to running jobs."""
import csv
import datetime
import json
from pathlib import Path
import sys

w=Path(sys.argv[1]).resolve()
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),lanes={})
for bits in (1,2,4):
    rows=list(csv.DictReader((w/f'match-b{bits}/cells.csv').open()))
    tq=[r for r in rows if r['codec']=='tieredq']
    refs=w/f'reference-build-b{bits}/cells.csv'
    lane=dict(reference_checks=len(list(csv.DictReader(refs.open()))) if refs.exists() else 0,
              tieredq_cells=len(tq),equal_recall_points=len(list((w/f'equal-recall/b{bits}').glob('*.json'))))
    if tq:lane['last_cell']={key:tq[-1][key] for key in ['dataset','requested_code_bits','seed','recall10']}
    result['lanes'][bits]=lane
result['timing_cells']=len(list((w/'timing').glob('cell-*.json')))
result['exit_codes']={p.name:p.read_text().strip() for p in w.glob('*exit-code')}
with (w/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps(result,indent=2))
