"""Isolate scheduling recall differences with fixed transformed SIFT probes."""
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_calibrated import ROOT,OUT,build,load_data
from measure import recall


def search(index,probes,batch,threads):
    ids=np.empty((len(probes),10),np.uint32)
    for offset in range(0,len(probes),batch):
        index.search(probes[offset:offset+batch],ids[offset:offset+batch],threads)
    return ids


def main():
    rows=list(csv.DictReader((OUT/'results.csv').open()))
    results=[]
    for s in json.loads((OUT/'selections.json').read_text()):
        group=[r for r in rows if r['dataset']==s['dataset'] and int(r['rabitq_anchor_bits'])==s['anchor_bits']]
        if len({r['recall10'] for r in group})==1: continue
        base,queries,truth,_=load_data(s['dataset'])
        index,transform,_=build(base,s['dataset'],s['target_code_bits'])
        assert transform is not None, 'Non-SIFT recall variation requires further investigation'
        serial=np.concatenate([transform.queries(q[None,:]) for q in queries])
        batched=np.concatenate([transform.queries(queries[i:i+32]) for i in range(0,len(queries),32)])
        ss=search(index,serial,1,8);sb=search(index,serial,32,8)
        bs=search(index,batched,1,8);bb=search(index,batched,32,8)
        single_thread=search(index,serial,1,1)
        result={'dataset':s['dataset'],'anchor_bits':s['anchor_bits'],'target_code_bits':s['target_code_bits'],
                'max_adapter_coordinate_difference':float(np.max(np.abs(serial-batched))),
                'adapter_coordinates_different':int(np.count_nonzero(serial!=batched)),
                'same_serial_probes_ids_equal_across_native_batches':bool(np.array_equal(ss,sb)),
                'same_batched_probes_ids_equal_across_native_batches':bool(np.array_equal(bs,bb)),
                'same_probes_ids_equal_across_threads':bool(np.array_equal(ss,single_thread)),
                'serial_recall':recall(truth,ss),'batch_recall':recall(truth,bb),
                'queries_top10_sets_changed':np.flatnonzero([set(a)!=set(b) for a,b in zip(ss,bb)]).tolist()}
        assert result['same_probes_ids_equal_across_threads']
        expected={int(r['batch']):float(r['recall10']) for r in group}
        assert expected[1]==result['serial_recall'] and expected[32]==result['batch_recall']
        np.savez(OUT/f"recall-diagnostic-{s['dataset']}-{s['anchor_bits']}.npz",serial_ids=ss,batched_ids=bb)
        results.append(result)
        print(json.dumps(result),flush=True)
    (OUT/'recall-diagnostics.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
