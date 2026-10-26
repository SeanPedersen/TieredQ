"""Spend the remaining low-anchor budget, selecting by disk and serving RAM before recall."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

bench=Path.cwd();sys.path.insert(0,str(bench))
from data import prepare
from measure import measure
import codec_native
output=bench.parent/'evidence'
reference=json.loads((output/'rabitq-memory.json').read_text())['records'][0]['memory']
limits={key:reference[key] for key in ('disk_bytes','ram_bytes')}
base,queries,truth=prepare(bench/'data','openai')
attempts=[];seen=set();best=None;bits=2048
meta={'affinity':sorted(os.sched_getaffinity(0)),'seed':0,'queries':1000,'warmups':3,'repeats':3,'limits':limits,'selection_policy':'Start at 2048 code bits; at most six storage-only adjustments. Keep both disk and serving RAM under native RaBitQ 2-bit limits. Accept first point within 0.5% of a binding limit, without recall. Routing remains model-selected.'}
with threadpool_limits(limits=1):
    for _ in range(6):
        if bits in seen: break
        seen.add(bits)
        codec_native.set_threads(8)
        tick=time.perf_counter()
        index=codec_native.Index(base,math.ceil(bits/1536),0,True,False,True,bits)
        build_s=time.perf_counter()-tick
        path=output/f'dual-budget-{bits}.bin'
        memory=index.memory_metrics(str(path))
        allocation=index.allocations(str(path))
        candidate={'code_bits':bits,'build_s':build_s,'encode_s':index.encode_seconds,'encode_vps':len(base)/index.encode_seconds,'memory':memory,'allocation':allocation,'diagnostics':index.diagnostics()}
        attempts.append(candidate)
        feasible=all(memory[key]<=limit for key,limit in limits.items())
        print('Storage calibration:',candidate,flush=True)
        if feasible:
            best=(index,candidate,path)
            if max(memory[key]/limit for key,limit in limits.items())>=0.995: break
        gap=min((limit-memory[key])/len(base) for key,limit in limits.items())
        delta=math.floor(gap*8)
        bits=max(1,bits+(delta if delta else -1))
        del index
    if best is None: raise RuntimeError('No dual-budget point fits')
    index,selected,path=best
    assert index.verify_reload(str(path),queries)
    selected['timings']=[]
    for threads,batch in ((8,32),(8,1),(1,1)):
        metric=measure(index,queries,truth,threads,batch)
        selected['timings'].append({'threads':threads,'batch':batch}|metric)
        print('Measured',selected['code_bits'],threads,batch,metric,flush=True)
    ids=np.empty((1000,10),dtype=np.uint32);index.search(queries,ids,8)
    np.save(output/'ids-dual-budget.npy',ids)
    selected['ids_sha256']=hashlib.sha256(ids.tobytes()).hexdigest()
    selected['index_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
meta.update(attempts=attempts,selected=selected,binary_sha256=hashlib.sha256(next((bench/'build-native').glob('codec_native*.so')).read_bytes()).hexdigest())
(output/'dual-budget.json').write_text(json.dumps(meta,indent=2))
