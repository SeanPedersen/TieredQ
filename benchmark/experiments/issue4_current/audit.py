"""Fail review preparation on missing seeds, resource overshoots or timing/provenance drift."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
from matrix import NAMES,primary_cells,primary_matches,primary_equal

w=Path(sys.argv[1]).resolve()
metrics=dict(disk='disk_bits_per_vector',ram='ram_loaded_bits_per_vector',peak='ram_peak_bits_per_vector')
all_rows={}
matched=[]
counts=dict(reference_checks=0,matches=0,equal_recall=0,timing_cells=0)
for bits in (1,2,4):
    directory=w/f'match-b{bits}'
    rows=primary_cells(w,bits)
    keyed={(r['dataset'],r['codec'],int(r['requested_code_bits']) if r['codec']=='tieredq' else bits,int(r['seed'])):r for r in rows}
    all_rows.update(keyed)
    fresh=list(csv.DictReader((w/f'reference-build-b{bits}/cells.csv').open()))
    assert len(fresh)==12,(bits,len(fresh))
    for r in fresh:
        if r['dataset'] not in NAMES:
            continue
        old=keyed[r['dataset'],r['codec'],bits,0]
        for field in ('disk_bytes','recall10','companion_recall10'):
            if old.get(field):assert float(old[field])==float(r[field]),(bits,field,old,r)
        counts['reference_checks']+=1
    new_refs=[r for r in rows if r['dataset']=='glove1m' and r['codec'] in ('rabitq','qdrant')]
    assert len(new_refs)==6,(bits,len(new_refs))
    for r in new_refs:
        assert int(r['base_n'])==1000000 and int(r['d'])==200
        assert float(r['disk_bytes'])>0 and 0<=float(r['recall10'])<=1
        counts['reference_checks']+=1
    for codec in ('rabitq','qdrant'):
        matches=primary_matches(w,bits,codec)
        assert len(matches)==18
        for row in matches:
            counts['matches']+=1
            if row['code_bits'] is None:
                assert row['tieredq'] is None
                continue
            selected=[keyed[row['dataset'],'tieredq',int(row['code_bits']),seed] for seed in (0,1,2)]
            actual=max(float(r[metrics[row['metric']]]) for r in selected)
            assert actual<=row['target_bits_per_vector']+1e-8,(row,actual)
            assert abs(st.mean(float(r['recall10']) for r in selected)-row['tieredq']['recall10']['mean'])<1e-12
            matched.append((row['dataset'],int(row['code_bits'])))
    eq_rows=primary_equal(w,bits)
    assert len(eq_rows)==(18 if bits==4 else 12),(bits,len(eq_rows))
    for row in eq_rows:
        assert {int(r['seed']) for r in row['memory_rows']}=={0,1,2}
        mean=st.mean(float(r['recall10']) for r in row['memory_rows'])
        assert abs(mean-row['recall10_mean'])<1e-12
        assert row['reached']==(mean>=row['reference_recall10'])
        counts['equal_recall']+=1
        matched.append((row['dataset'],int(row['code_bits'])))
plan=json.loads((w/'timing/plan.json').read_text())
actual_plan={(ds,code) for ds,codec,bits,code in plan if codec=='tieredq'}
assert set(matched)<=actual_plan
schedule_differences=[]
for ds,codec,bits,code in plan:
    name=f'cell-{ds}-{codec}-{bits}'+(f'-c{code}' if code else '')+'.json'
    path=w/'timing'/name
    rows=json.loads(path.read_text())
    assert {(r['threads'],r['batch']) for r in rows}=={(1,1),(1,32),(8,1),(8,32)}
    expected=all_rows[ds,codec,code or bits,0]
    for row in rows:
        assert row['queries']==1000 and row['seed']==0 and row['k']==10 and not row['rerank']
        assert row['source_revision']==(w/'source/SOURCE_REVISION').read_text().strip()
        assert row['cpu_affinity']==list(range(8)),row['cpu_affinity']
        if codec=='tieredq':assert row['variant']=='fine' and row['fine_tiers_requested']
        assert abs(float(row['effective_bytes_per_vector'])-float(expected['disk_bytes'])/int(expected['base_n']))<1e-8
        assert row['qps']>0 and row['p95_ms']>=row['p50_ms']>0
        if abs(row['recall10']-float(expected['recall10']))>1e-10:
            schedule_differences.append(dict(cell=name,threads=row['threads'],batch=row['batch'],
                                            recall10=row['recall10'],memory_recall10=float(expected['recall10'])))
    profile=json.loads((w/'preparation'/name).read_text())
    assert profile['available'] and profile['search_ids_equal'] and len(profile['schedules'])==4
    assert profile['queries']==1000 and profile['base_n']==int(expected['base_n'])
    counts['timing_cells']+=1
# Canonical accuracy is the same 8t/b32 schedule used by memory measurement and confirmation.
assert not [r for r in schedule_differences if r['threads']==8 and r['batch']==32],schedule_differences
frozen=json.loads((w/'frozen-native.json').read_text())
assert hashlib.file_digest(Path(frozen['module']).open('rb'),'sha256').hexdigest()==frozen['sha256']
result=dict(status='passed',counts=counts,schedule_recall_differences=schedule_differences)
(w/'review-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
