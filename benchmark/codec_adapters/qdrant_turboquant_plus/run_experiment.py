"""Storage-only discrete calibration, validation, then the single-seed scheduling matrix."""
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from adapters import Qdrant, TieredQ, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measure import recall

OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
MODELS = OUT/'models'
MODELS.mkdir(exist_ok=True)
TOLERANCE = 0.005
MAX_ATTEMPTS = 24
SCHEDULE = [(t,b) for t in (1,8) for b in (1,32)]


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def predictions(index, queries, threads, batch):
    ids = np.empty((len(queries),10), dtype=np.uint32)
    for start in range(0,len(queries),batch):
        index.search(queries[start:start+batch], ids[start:start+batch], threads)
    return ids


def validate(index, queries, label):
    probes = np.ascontiguousarray(queries[:65])
    expected = predictions(index, probes, 1, 1)
    assert np.all(np.diff(np.sort(expected,axis=1),axis=1)>0)
    for threads,batch in SCHEDULE:
        assert np.array_equal(expected,predictions(index,probes,threads,batch)), (label,threads,batch)
    if isinstance(index,Qdrant):
        for query in probes[:3]:
            assert index.lib.tq_validate(index.pointer,query.ctypes.data), (label,"native scalar/batch scores")
    before = index.details()
    index.reload()
    assert before == index.details(), (label,'reload storage')
    assert np.array_equal(expected,predictions(index,probes,8,32)), (label,'reload predictions')
    return dict(label=label,query_count=len(probes),serial_batch_thread_ids_equal=True,reload_ids_equal=True,storage_equal=True)


def measure(index, queries, truth, threads, batch, reference, label):
    out = np.empty((len(queries),10),dtype=np.uint32)
    for i in range(3):
        index.search(queries[i:i+1],out[:1],threads)
    durations=[]
    for repeat in range(3):
        for start in range(0,len(queries),batch):
            stop=min(start+batch,len(queries));tick=time.perf_counter()
            index.search(queries[start:stop],out[start:stop],threads)
            durations.append(dict(repeat=repeat,start=start,count=stop-start,seconds=time.perf_counter()-tick))
        assert np.array_equal(out,reference),(label,threads,batch,repeat,'scheduling variation')
    latencies=np.repeat([r['seconds']*1000 for r in durations],[r['count'] for r in durations])
    save(f'timings-{label}-{threads}-{batch}.json',durations)
    np.save(OUT/f'ids-{label}-{threads}-{batch}.npy',out)
    return dict(recall10=recall(truth,out),qps=len(queries)*3/sum(r['seconds'] for r in durations),
                p50_ms=float(np.percentile(latencies,50)),p95_ms=float(np.percentile(latencies,95)))


def main():
    assert sorted(os.sched_getaffinity(0)) == list(range(8))
    with np.load(Path(__file__).resolve().parents[2]/'data/gemini-prepared.npz') as data:
        base,queries,truth=(data[k] for k in ('base','queries','truth'))
    assert base.shape==(99000,768) and queries.shape==(1000,768) and truth.shape==(1000,10)
    validations=[];anchors=[];history=[];selected=[]
    metadata=dict(status='building-anchors',seed=0,affinity=sorted(os.sched_getaffinity(0)),
      selection_rule='Recall never evaluated in calibration. First measured under-anchor point within 0.5%; otherwise largest measured feasible serving byte total after bounded secant/bisection, integer budgets 1..8*d. Every proposed layout is built and measured.',
      tolerance=TOLERANCE,max_attempts=MAX_ATTEMPTS)
    save('meta.json',metadata)
    for bits in (1,2,4):
        index=Qdrant(base,bits,MODELS/f'qdrant-{bits}')
        details=index.details();validations.append(validate(index,queries,f'qdrant-{bits}'))
        anchors.append(dict(anchor_bits=bits,storage=details,build_s=index.build_s,encode_vps=len(base)/index.encode_seconds))
        index.close();save('anchors.json',anchors);save('validation.json',validations)
        print('ANCHOR',anchors[-1],flush=True)
    tick=time.perf_counter()
    for anchor in anchors:
        limit=anchor['storage']['serving_bytes'];maximum=8*base.shape[1]
        measured={};budget=min(maximum,max(1,math.floor((limit/len(base)-24)*8)))
        for attempt in range(MAX_ATTEMPTS):
            start=time.perf_counter();path=MODELS/f'calibration-{anchor["anchor_bits"]}-{budget}.bin'
            index=TieredQ(base,budget,path);details=index.details()
            record=dict(anchor_bits=anchor['anchor_bits'],requested_code_bits=budget,storage=details,
                        build_s=index.build_s,encode_vps=len(base)/index.encode_seconds,wall_s=time.perf_counter()-start)
            measured[budget]=record;history.append(record);index.close();save('calibration.json',history)
            print('CALIBRATION',anchor['anchor_bits'],budget,details['serving_bytes'],limit,flush=True)
            size=details['serving_bytes']
            if size<=limit and ((limit-size)/limit<=TOLERANCE or budget==maximum):break
            under=[b for b,r in measured.items() if r['storage']['serving_bytes']<=limit]
            lower=max(under,default=0)
            upper=min((b for b,r in measured.items() if b>lower and r['storage']['serving_bytes']>limit),default=maximum+1)
            proposal=max(lower+1,min(upper-1,budget+math.floor((limit-size)*8/len(base))))
            if proposal in measured:
                untried=[b for b in range(lower+1,upper) if b not in measured]
                if not untried:break
                proposal=untried[len(untried)//2]
            budget=proposal
        feasible=[r for r in measured.values() if r['storage']['serving_bytes']<=limit]
        if not feasible:raise RuntimeError(f'No feasible layout for anchor {anchor}')
        selected.append(max(feasible,key=lambda r:r['storage']['serving_bytes']));save('selections.json',selected)
    metadata['calibration_wall_s']=time.perf_counter()-tick;metadata['status']='validating-selected'
    save('meta.json',metadata)
    indexes=[]
    for a,s in zip(anchors,selected,strict=True):
        bits=a['anchor_bits']
        q=Qdrant.__new__(Qdrant);q.path=MODELS/f'qdrant-{bits}';q.dim=768;q.pointer=q.lib.tq_load(str(q.path).encode())
        indexes.append((q,a,a['build_s'],a['encode_vps'],None))
        t=TieredQ(base,s['requested_code_bits'],MODELS/f'selected-tieredq-{bits}.bin',validate_reload=True)
        assert t.details()==s['storage'],('selected rebuild differs',bits)
        validations.append(validate(t,queries,f'tieredq-{bits}'))
        indexes.append((t,a,t.build_s,len(base)/t.encode_seconds,s['requested_code_bits']))
    save('validation.json',validations)
    # All adapters pass validation before any timing starts.
    rows=[];search_start=time.perf_counter()
    for index,anchor,build_s,encode_vps,budget in indexes:
        bits=anchor['anchor_bits'];label=f'{index.codec}-{bits}';details=index.details()
        reference=predictions(index,queries,1,1)
        np.save(OUT/f'ids-reference-{label}.npy',reference)
        for threads,batch in SCHEDULE:
            row=dict(codec=index.codec,anchor_bits=bits,requested_code_bits=budget,
                     learned_code_bits=details.get('learned_code_bits',bits*768),base_n=len(base),queries=len(queries),seed=0,
                     bytes_per_vector=details['serving_bytes']/len(base),disk_bytes=details['disk_bytes'],resident_bytes=details['resident_bytes'],
                     anchor_bytes_per_vector=anchor['storage']['serving_bytes']/len(base),
                     percent_below_anchor=100*(anchor['storage']['serving_bytes']-details['serving_bytes'])/anchor['storage']['serving_bytes'],
                     build_s=build_s,encode_vps=encode_vps,threads=threads,batch=batch)
            row.update(measure(index,queries,truth,threads,batch,reference,label));rows.append(row)
            with (OUT/'results.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            print('MEASURED',row,flush=True)
        index.close()
    metadata.update(status='complete',search_matrix_wall_s=time.perf_counter()-search_start,
                    selected_build_s=sum(x[2] for x in indexes),rows=len(rows))
    save('meta.json',metadata)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
