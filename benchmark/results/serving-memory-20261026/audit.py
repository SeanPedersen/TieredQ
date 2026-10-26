"""Audit full-data memory identities and unchanged-budget persisted/search outputs."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
before=json.loads((root/'before.json').read_text())
after=json.loads((root/'after.json').read_text())
quality=json.loads((root/'quality.json').read_text())
assert before['data_manifest']==after['data_manifest']==quality['data_manifest']
expected={1536,1695,3072,5344,6144,11488,12288}
for result in (before,after,quality):
    assert result['affinity']==list(range(8))
    assert (result['seed'],result['queries'],result['warmups'],result['repeats'])==(0,1000,3,3)
    for point in result['records']:
        a=point['allocation']
        assert a['count']==100000
        assert sum(a['model'].values())==a['model_total']
        assert sum(a['cache'].values())==a['cache_total']
        assert a['model_total']+a['cache_total']==a['resident_bytes']
        assert point['bytes_per_vector']==max(a['resident_bytes'],a['serialized_bytes'])/a['count']
        assert point['build_s']>point['encode_s']>0
        recalls={t['recall10'] for t in point['timings']}
        assert len(recalls)<=1
        for t in point['timings']:
            assert t['qps']>0 and t['p95_ms']>=t['p50_ms']>0
b={p['code_bits']:p for p in before['records']}
a={p['code_bits']:p for p in after['records']}
assert b.keys()==a.keys()==expected
for bits in expected:
    assert b[bits]['index_sha256']==a[bits]['index_sha256']
    assert b[bits]['diagnostics']['tiers']==a[bits]['diagnostics']['tiers']
    if b[bits]['timings']:
        assert b[bits]['ids_sha256']==a[bits]['ids_sha256']
        assert b[bits]['timings'][0]['recall10']==a[bits]['timings'][0]['recall10']
    if bits in (1536,1695):
        assert a[bits]['bytes_per_vector']<b[bits]['bytes_per_vector']
    else:
        assert a[bits]['allocation']==b[bits]['allocation']
q=quality['records'][0]
assert q['code_bits']>1695 and q['bytes_per_vector']<=quality['anchor_bytes']
dual=json.loads((root/'dual-budget.json').read_text())
rq=json.loads((root/'rabitq-memory.json').read_text())
selected=dual['selected']
old_selected=json.loads((root/'dual-budget-baseline.json').read_text())
assert dual['affinity']==rq['affinity']==list(range(8))
assert selected['ids_sha256']==old_selected['ids_sha256']
assert selected['allocation']['serialized_bytes']==old_selected['allocation']['serialized_bytes']
assert old_selected['allocation']['resident_bytes']>dual['limits']['ram_bytes']
assert all(selected['memory'][key]<=limit for key,limit in dual['limits'].items())
for point in after['records']+quality['records']:
    m=point['separate_memory_metrics']
    assert sum(m['ram_components'].values())==m['ram_bytes']
    assert m['disk_bytes']==point['allocation']['serialized_bytes']
    assert m['ram_bytes']==point['allocation']['resident_bytes']+56
for point in rq['records']:
    m=point['memory']
    assert sum(m['ram_components'].values())==m['ram_bytes']
    assert m['disk_bits_per_vector']==m['disk_bytes']*8/100000
    assert m['ram_bits_per_vector']==m['ram_bytes']*8/100000
assert selected['timings'][0]['recall10']>rq['records'][0]['timings'][0]['recall10']
assert before['truth_sha256']==after['truth_sha256']==quality['truth_sha256']
assert after['binary_sha256']==quality['binary_sha256']==dual['binary_sha256']
for p in before['records']+after['records']:
    if p['timings']: assert p['reloaded_scores_and_ids_equal']
report={'status':'pass','unchanged_budget_count':len(expected),'identical_persisted_indexes':len(expected),'identical_full_query_id_arrays':3,'primary_saved_bytes':b[1695]['allocation']['resident_bytes']-a[1695]['allocation']['resident_bytes'],'quality_budget':q['code_bits'],'quality_bytes':q['bytes_per_vector'],'quality_recall':q['timings'][0]['recall10']}
report.update(dual_budget_code_bits=selected['code_bits'],dual_budget_memory=selected['memory'],dual_budget_recall=selected['timings'][0]['recall10'],rabitq_recall=rq['records'][0]['timings'][0]['recall10'])
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
