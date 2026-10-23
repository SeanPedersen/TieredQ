"""Audit the requested fixed-seed issue #4 experiment artifacts."""
import csv, hashlib, itertools, json, math, pathlib, sys
root=pathlib.Path(sys.argv[1]).resolve(); repo=root.parent.parent
rows=list(csv.DictReader((root/'results.csv').open()))
key=lambda r:(r['dataset'],r['codec'],int(r['target_bits']),int(r['threads']),int(r['batch']),int(r['seed']))
expected=set(itertools.product(('sift','glove','openai'),('tieredq','rabitq'),(1,2,4,8),(1,8),(1,32),(0,)))
assert len(rows)==96 and len({key(r) for r in rows})==96 and {key(r) for r in rows}==expected
for r in rows:
    assert int(r['queries'])==1000
    assert (int(r['d']),int(r['base_n']))=={'sift':(128,1000000),'glove':(200,100000),'openai':(1536,100000)}[r['dataset']]
    assert r['config']=={'tieredq':'fast-native','rabitq':'flat-fht-native-bounds-hacc'}[r['codec']]
    for field in ('effective_bits_per_dim','effective_bytes_per_vector','build_s','encode_vps','qps','p50_ms','p95_ms'):
        assert math.isfinite(float(r[field])) and float(r[field])>0,(key(r),field)
    assert 0<=float(r['recall10'])<=1 and float(r['p95_ms'])>=float(r['p50_ms'])
    assert math.isclose(float(r['effective_bits_per_dim']),float(r['effective_bytes_per_vector'])*8/int(r['d']))
for cell in itertools.product(('sift','glove','openai'),('tieredq','rabitq'),(1,2,4,8)):
    group=[r for r in rows if key(r)[:3]==cell]
    for field in ('recall10','effective_bytes_per_vector','build_s','encode_vps'):
        assert len({r[field] for r in group})==1,(cell,field)
meta=json.loads((root/'meta.json').read_text())
assert meta['status']=='complete' and not meta['validation_only']
assert meta['cpu_affinity']==list(range(8))
assert meta['rabitq_search']==dict(container='direct-flat',implicit_row_ids=True,high_accuracy=True,extra_bit_bounds=True)
sys.path.insert(0,str(repo/'benchmark'))
from build_policy import audit_compile_commands
assert audit_compile_commands(meta['compiler_commands'])
manifest=json.loads((root/'validated-source-sha256.json').read_text())
for name,digest in manifest.items(): assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
for name,digest in meta['source_sha256'].items(): assert manifest[name]==digest,name
for name,digest in meta['harness_sha256'].items(): assert manifest['benchmark/'+name]==digest,name
profile=list(csv.DictReader((root/'preprocessing.csv').open()))
assert len(profile)==96
assert {(r['dataset'],r['codec'],int(r['bits']),int(r['threads']),int(r['batch'])) for r in profile}=={k[:5] for k in expected}
for r in profile: assert 0<=float(r['preprocessing_fraction'])<=1
from PIL import Image
for name in ('recall-storage','qps-recall','latency-recall'):
    with Image.open(root/(name+'.png')) as im: im.verify()
assert len(json.loads((root/'summary.json').read_text())['operating_points'])==12
result=dict(status='passed',rows=len(rows),codec_configurations=24,seed=0,matrix_wall_s=meta['matrix_wall_s'],profiling_wall_s=meta['profiling_wall_s'],compiler_translation_units=len(meta['compiler_commands']),validated_source_files=len(manifest),profile_rows=len(profile),preprocessing_above_10_percent=sum(float(r['preprocessing_fraction'])>.1 for r in profile))
(root/'completeness-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
