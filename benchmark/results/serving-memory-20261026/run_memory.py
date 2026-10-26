"""Focused seed-0 OpenAI A/B: identical budgets first, storage-selected quality follow-up second."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

parser=argparse.ArgumentParser()
parser.add_argument('variant',choices=('before','after'))
parser.add_argument('--quality',action='store_true')
args=parser.parse_args()
bench=Path.cwd()
sys.path.insert(0,str(bench))
from data import prepare
from measure import measure
from native_build import build_native

# Another session may start work after the initial job check. Preserve it and wait.
while True:
    jobs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    competing=[line for line in jobs if 'python' in line and 'learned_codebooks/run.py' in line and 'tailscaled' not in line]
    if not competing: break
    print('Waiting for existing unrelated benchmark:',competing,flush=True)
    time.sleep(15)

output=bench.parent/'evidence' 
output.mkdir(exist_ok=True)
meta={'variant':args.variant,'affinity':sorted(os.sched_getaffinity(0)),'seed':0,'queries':1000,'warmups':3,'repeats':3}
meta['compiler_commands']=build_native().compiler_commands
import codec_native
meta['binary_sha256']=hashlib.sha256(next((bench/'build-native').glob('codec_native*.so')).read_bytes()).hexdigest()
base,queries,truth=prepare(bench/'data','openai')
manifest=json.loads((bench/'data/openai-100000-1000-v2.json').read_text())
for name,value in [('base',base),('queries',queries)]:
    assert hashlib.sha256(value.tobytes()).hexdigest()==manifest[f'{name}_sha256']
assert base.shape==(100000,1536) and queries.shape==(1000,1536)
meta['data_manifest']=manifest
meta['truth_sha256']=hashlib.sha256(truth.tobytes()).hexdigest()
meta['uname']=subprocess.check_output(['uname','-a'],text=True).strip()
meta['cpuinfo']=subprocess.check_output(['lscpu'],text=True)
meta['source_sha256']={str(p.relative_to(bench.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src','include','tests','benchmark/native') for p in (bench.parent/folder).rglob('*') if p.is_file()}
records=[]
start=time.perf_counter()

def build(bits):
    codec_native.set_threads(8)
    tick=time.perf_counter()
    index=codec_native.Index(base,math.ceil(bits/1536),0,True,False,True,bits)
    build_s=time.perf_counter()-tick
    path=output/f'index-{bits}.bin'
    allocation=index.allocations(str(path))
    memory=index.memory_metrics(str(path)) if hasattr(index,'memory_metrics') else None
    assert sum(allocation['cache'].values())==allocation['cache_total']
    assert sum(allocation['model'].values())==allocation['model_total']
    measured=max(allocation['resident_bytes'],allocation['serialized_bytes'])/len(base)
    record={'code_bits':bits,'build_s':build_s,'encode_s':index.encode_seconds,'encode_vps':len(base)/index.encode_seconds,'bytes_per_vector':measured,'allocation':allocation,'separate_memory_metrics':memory,'diagnostics':index.diagnostics(),'index_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'timings':[]}
    print('Built',bits,'bytes/vector',measured,'seconds',build_s,flush=True)
    return index,record

def timing(index,record,settings):
    bits=record['code_bits']
    codec_native.set_threads(8)
    record['reloaded_scores_and_ids_equal']=index.verify_reload(str(output/f'index-{bits}.bin'),queries)
    assert record['reloaded_scores_and_ids_equal']
    for threads,batch in settings:
        metric=measure(index,queries,truth,threads,batch)
        record['timings'].append({'threads':threads,'batch':batch}|metric)
        print('Measured',bits,threads,batch,metric,flush=True)
    ids=np.empty((len(queries),10),dtype=np.uint32)
    index.search(queries,ids,8)
    np.save(output/f'ids-{bits}.npy',ids)
    record['ids_sha256']=hashlib.sha256(ids.tobytes()).hexdigest()

def save():
    meta.update(records=records,wall_s=time.perf_counter()-start)
    (output/('quality.json' if args.quality else 'results.json')).write_text(json.dumps(meta,indent=2))

with threadpool_limits(limits=1):
    if not args.quality:
        for bits in (1695,5344,11488,1536,3072,6144,12288):
            index,record=build(bits)
            if bits in (1695,5344,11488):
                settings=[(8,32),(8,1)]
                if bits==1695: settings.append((1,1))
                timing(index,record,settings)
            records.append(record);save()
            del index
    else:
        # Storage only selects a larger code budget under the original low-memory anchor.
        limit=404.0696
        bits=1920
        upper=2144
        lower=1695
        attempts=[]
        best=None
        for _ in range(6):
            index,record=build(bits)
            narrow=all(t['bits']<=4 for t in record['diagnostics']['tiers'])
            # Keep this follow-up on the affected layout; packed paths have no saving from this fix.
            old_width=math.ceil((record['diagnostics']['coded_dimensions']+512)/16)*16
            old_bytes=record['bytes_per_vector']+(old_width-512)*1536*4/len(base) if narrow else record['bytes_per_vector']
            attempts.append({'code_bits':bits,'narrow':narrow,'bytes_per_vector':record['bytes_per_vector'],'old_layout_bytes_per_vector':old_bytes,'build_s':record['build_s'],'allocation':record['allocation']})
            if not narrow:
                upper=bits-1
                bits=(lower+upper)//2
                del index
                continue
            if record['bytes_per_vector']<=limit:
                best=(index,record)
                if old_bytes>limit: break
                lower=bits
            bits=max(lower+1,min(upper,bits+math.floor((limit-record['bytes_per_vector'])*8)))
            del index
        if best is None: raise RuntimeError('No within-memory quality point')
        index,record=best
        timing(index,record,[(8,32),(8,1),(1,1)])
        records.append(record);meta.update(calibration=attempts,anchor_bytes=limit,selection_policy='First measured narrow point that fits after the change and exceeds the anchor with the old projection; storage and routing only, no recall used.');save()
print('Finished',args.variant,'quality' if args.quality else 'unchanged budgets',meta['wall_s'],flush=True)
