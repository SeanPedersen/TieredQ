"""Summarize raw measurements and choose serving candidates without evaluation labels."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1])
def read(name):
    path=root/name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def grouped(rows,keys,fields):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys,key),n=len(values),**{field:statistics.mean(r[field] for r in values) for field in fields}) for key,values in groups.items()]

summary=dict(allocation_quality=grouped(read('allocation-quality.jsonl'),('policy','budget'),('recall10','bytes_per_vector')),
             allocation_timing=grouped(read('allocation-timing.jsonl'),('policy','budget','batch','threads'),('p50_ms','p95_ms','qps')),
             lookup_timing=grouped(read('lookup-timing.jsonl'),('grouping','budget','batch','threads'),('p50_ms','p95_ms','qps')),
             wide_isolated_timing=grouped(read('wide-isolated-timing.jsonl'),('budget','grouping','batch','threads'),('p50_ms','p95_ms','qps')),
             cross_isolated_timing=grouped(read('cross-isolated-timing.jsonl'),('dataset','bits','grouping','batch','threads'),('p50_ms','p95_ms','qps')),
             wide_timing=grouped(read('wide-timing.jsonl'),('budget','grouping','batch','threads'),('p50_ms','p95_ms','qps')),
             cross_timing=grouped(read('cross-timing.jsonl'),('dataset','bits','grouping','batch','threads'),('p50_ms','p95_ms','qps')),
             row_counts={name:len(read(name)) for name in ('lookup-quality.jsonl','lookup-timing.jsonl','allocation-quality.jsonl','allocation-timing.jsonl','costs.jsonl','cross-quality.jsonl','cross-timing.jsonl','phases.jsonl')})
candidates=read('costs-isolated.jsonl') or read('costs-controlled.jsonl') or read('costs.jsonl')
constrained=[]
for memory in [c['bytes_per_vector'] for c in candidates if c['policy']=='marginal']:
    for latency in (0.75,1.0,1.5,2.0,3.0):
        feasible=[c for c in candidates if c['bytes_per_vector']<=memory and c['base_single8_p50_ms']<=latency]
        current=[c for c in feasible if c['policy']=='marginal']
        if not current:
            continue
        constrained.append(dict(memory_ceiling=memory,latency_ceiling_ms=latency,
                           current=min(current,key=lambda c:c['base_reconstruction_mse']),
                           selected=min(feasible,key=lambda c:c['base_reconstruction_mse'])))
summary['base_only_constrained_selection']=constrained
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['row_counts']))
for r in constrained:
    if abs(r['memory_ceiling']-41.32)<.01 and r['latency_ceiling_ms']==1.0:
        print(json.dumps(r))
