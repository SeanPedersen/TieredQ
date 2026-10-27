"""Validate scheduling identities and retain complete predictions and raw call durations."""
import time
import math
import numpy as np
import json

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")
SCHEDULE=((1,1),(1,32),(8,1),(8,32))


def predictions(index, queries, threads=8, batch=32):
    ids=np.empty((len(queries),10),dtype=np.uint32)
    for start in range(0,len(queries),batch):
        index.search(queries[start:start+batch],ids[start:start+batch],threads)
    return ids


def recall(truth, ids):
    return float(np.mean([len(set(a)&set(b))/10 for a,b in zip(truth,ids,strict=True)]))


def validate(index, queries):
    probes=np.ascontiguousarray(queries[:65])
    expected=predictions(index,probes,1,1)
    assert np.all(np.diff(np.sort(expected,axis=1),axis=1)>0)
    for threads,batch in SCHEDULE:
        assert np.array_equal(expected,predictions(index,probes,threads,batch))
    if hasattr(index,'reload'):
        before=index.details()
        for q in probes[:3]:
            assert index.lib.tq_validate(index.pointer,q.ctypes.data)
        index.reload()
        assert before == index.details()
        assert np.array_equal(expected,predictions(index,probes))
    return dict(probes=65,scheduling_identity=True,reload_and_scalar_batch=hasattr(index,'reload'))


def evaluate(index, queries, truth, common, directory):
    validation=validate(index,queries)
    save(directory/f'validation-seed{common["seed"]}.json',validation)
    expected=predictions(index,queries)
    np.save(directory/f'ids-seed{common["seed"]}.npy',expected)
    result=[]
    for threads,batch in SCHEDULE:
        values=dict(recall10=recall(truth,expected),qps=math.nan,p50_ms=math.nan,p95_ms=math.nan)
        if common['seed']==0:
            output=np.empty_like(expected)
            for i in range(3):
                index.search(queries[i:i+1],output[:1],threads)
            timings=[]
            for repeat in range(3):
                for start in range(0,len(queries),batch):
                    stop=min(start+batch,len(queries));tick=time.perf_counter()
                    index.search(queries[start:stop],output[start:stop],threads)
                    timings.append(dict(repeat=repeat,start=start,count=stop-start,seconds=time.perf_counter()-tick))
                assert np.array_equal(output,expected)
                np.save(directory/f'ids-t{threads}-b{batch}-repeat{repeat}.npy',output)
            save(directory/f'timings-t{threads}-b{batch}.json',timings)
            latencies=np.repeat([r['seconds']*1000 for r in timings],[r['count'] for r in timings])
            values.update(qps=len(queries)*3/sum(r['seconds'] for r in timings),
                          p50_ms=float(np.percentile(latencies,50)),p95_ms=float(np.percentile(latencies,95)))
        result.append(common|values|dict(threads=threads,batch=batch))
        print('MEASURED',directory.name,common['seed'],threads,batch,values,flush=True)
    return result
