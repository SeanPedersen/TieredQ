"""Isolate SIFT adapter rounding behind the observed one-bit batch recall difference."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'benchmark'))
import numpy as np
from threadpoolctl import threadpool_limits
import codec_native
from data import prepare
from l2_transform import L2Transform
from measure import recall


def main():
    with threadpool_limits(limits=1):
        base,queries,truth=prepare(Path('data'),'sift')
        transform=L2Transform.fit(base)
        serial=np.concatenate([transform.queries(q[None,:]) for q in queries])
        batched=np.concatenate([transform.queries(queries[i:i+32]) for i in range(0,len(queries),32)])
        codec_native.set_threads(8)
        index=codec_native.Index(transform.base_vectors(base),1,0,True,True)
        def search(probes,batch):
            ids=np.empty((len(probes),10),np.uint32)
            for offset in range(0,len(probes),batch):
                index.search(probes[offset:offset+batch],ids[offset:offset+batch],8)
            return ids
        ss=search(serial,1); sb=search(serial,32); bs=search(batched,1); bb=search(batched,32)
        result={'scope':'SIFT TieredQ 1-bit seed 0 only; targeted correctness diagnostic, no performance measurements', 'max_adapter_coordinate_difference':float(np.max(np.abs(serial-batched))), 'adapter_coordinates_different':int(np.count_nonzero(serial!=batched)), 'same_serial_probes_ids_equal_across_native_batches':bool(np.array_equal(ss,sb)), 'same_batched_probes_ids_equal_across_native_batches':bool(np.array_equal(bs,bb)), 'serial_adapter_recall':recall(truth,ss), 'batched_adapter_recall':recall(truth,bb), 'query_top10_sets_changed':sum(set(a)!=set(b) for a,b in zip(ss,bb)), 'interpretation':'See equality checks: adapter matrix multiplication rounding is isolated by holding transformed probes fixed while varying native batch size.'}
        Path('../batch-recall-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
