"""Measure exact-code-budget OpenAI fast indexes chosen by serving storage alone."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
BENCH=HERE/'benchmark'
sys.path.insert(0,str(BENCH))
import numpy as np
from threadpoolctl import threadpool_limits
from data import prepare
from measure import measure
from run import FIELDS

DIMENSIONS=1536
MAX_CALIBRATIONS=6
STORAGE_TOLERANCE=0.005
ANCHORS=((2,404.0696,1792),(4,788.0696,5344),(8,1556.0696,11488))


def storage(index,root):
    with tempfile.TemporaryDirectory(dir=root) as temp:
        return index.storage(str(Path(temp)/'index.bin'))/100000


def main():
    import codec_native
    from native_build import build_native
    output=HERE/'output-aligned';output.mkdir(exist_ok=False)
    native=build_native()
    metadata={'status':'running','commit':(HERE/'SOURCE_REVISION').read_text().strip(),'protocol':'OpenAI only; seed 0; fast mode only; three warmup queries and three timing repeats; original RaBitQ anchors reused from completed current core run','cpu_affinity':sorted(os.sched_getaffinity(0)),'compiler_commands':native.compiler_commands,'source_sha256':json.loads((HERE/'source-sha256.json').read_text()),'anchors':ANCHORS,'storage_tolerance':STORAGE_TOLERANCE,'max_calibrations':MAX_CALIBRATIONS,'selection_policy':'Use measured serving storage only; accept first under-anchor point within 0.5%; adjust integer total-code-bit budget by remaining byte gap; no recall used to select budgets.'}
    metadata['binary_sha256']=hashlib.sha256(next(native.directory.glob('codec_native*.so')).read_bytes()).hexdigest()
    base,queries,truth=prepare(BENCH/'data','openai')
    manifest=json.loads((BENCH/'data/openai-100000-1000-v2.json').read_text())
    for key,array in [('base',base),('queries',queries)]:
        assert hashlib.sha256(array.tobytes()).hexdigest()==manifest[f'{key}_sha256']
    metadata['data_manifest']=manifest
    calibrations=[];selections=[];selected_wall=0
    (output/'meta.json').write_text(json.dumps(metadata,indent=2))
    total_start=time.perf_counter()
    with threadpool_limits(limits=1), (output/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS+['target_code_bits','rabitq_anchor_bits','rabitq_anchor_bytes'])
        writer.writeheader()
        for anchor_bits,limit,code_bits in ANCHORS:
            seen=set();best=None
            for attempt in range(MAX_CALIBRATIONS):
                if code_bits in seen: break
                seen.add(code_bits)
                codec_native.set_threads(8)
                cell_start=time.perf_counter()
                index=codec_native.Index(base,math.ceil(code_bits/DIMENSIONS),0,True,False,True,code_bits)
                build_s=time.perf_counter()-cell_start
                measured=storage(index,output)
                calibration={'anchor_bits':anchor_bits,'attempt':attempt+1,'target_code_bits':code_bits,'nominal_bits_per_dimension':code_bits/DIMENSIONS,'bytes_per_vector':measured,'build_s':build_s,'tiers':index.diagnostics()['tiers']}
                calibrations.append(calibration)
                (output/'calibration.json').write_text(json.dumps(calibrations,indent=2))
                print('Storage calibration:',json.dumps(calibration),flush=True)
                if measured<=limit and (best is None or measured>best[0]): best=(measured,index,code_bits,build_s,cell_start)
                if measured<=limit and (limit-measured)/limit<=STORAGE_TOLERANCE: break
                delta=math.floor((limit-measured)*8)
                code_bits=max(1,min(DIMENSIONS*8,code_bits+(delta if delta else -1)))
            if best is None: raise RuntimeError(f'No within-memory point for anchor {anchor_bits}')
            measured,index,code_bits,build_s,cell_start=best
            timing_start=time.perf_counter()
            common=dict(dataset='openai',d=DIMENSIONS,base_n=len(base),queries=len(queries),metric='cosine',codec='tieredq',config='fast-native-exact-code-budget',target_bits=code_bits/DIMENSIONS,effective_bits_per_dim=measured*8/DIMENSIONS,effective_bytes_per_vector=measured,build_s=build_s,encode_vps=len(base)/index.encode_seconds,rerank=False,seed=0,target_code_bits=code_bits,rabitq_anchor_bits=anchor_bits,rabitq_anchor_bytes=limit)
            for threads in (1,8):
                for batch in (1,32):
                    metrics=measure(index,queries,truth,threads,batch)
                    writer.writerow(common|metrics|dict(threads=threads,batch=batch));stream.flush()
                    print(f'Measured code_bits={code_bits}, threads={threads}, batch={batch}: {metrics}',flush=True)
            selected_wall+=build_s+time.perf_counter()-timing_start
            selections.append(common|{'diagnostics':index.diagnostics()})
            del index,best
    metadata.update(status='complete',total_wall_s=time.perf_counter()-total_start,selected_build_and_search_wall_s=selected_wall,selections=selections,calibration_attempts=len(calibrations))
    (output/'meta.json').write_text(json.dumps(metadata,indent=2))
    print('Completed:',metadata['total_wall_s'],'seconds',flush=True)


if __name__=='__main__':
    main()
