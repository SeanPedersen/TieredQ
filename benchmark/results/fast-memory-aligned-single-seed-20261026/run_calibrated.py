"""Storage-only discrete calibration followed by the requested serving matrix."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits
from data import prepare
from l2_transform import L2Transform
from measure import L2Search, measure
from native_build import build_native
from run import FIELDS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
TOLERANCE = 0.005
MAX_ATTEMPTS = 20
DATASETS = {'sift': (1000000,128), 'glove': (100000,200), 'openai': (100000,1536)}
STARTS = {'sift': [140,340,580,1090], 'glove': [210,470,980,2000], 'openai': [800,2048,5344,11488]}


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_data(dataset):
    base, queries, truth = prepare(ROOT/'benchmark/data', dataset)
    assert base.shape == DATASETS[dataset] and len(queries) == 1000
    expected = json.loads((ROOT/'historical-data-verification.json').read_text())[dataset]
    hashes = {key:digest(value) for key,value in [('base',base),('queries',queries),('truth',truth)]}
    assert hashes == expected['array_sha256'], (dataset, hashes)
    return base,queries,truth,hashes


def build(base, dataset, budget):
    import codec_native
    codec_native.set_threads(8)
    start = time.perf_counter()
    transform = L2Transform.fit(base) if dataset == 'sift' else None
    vectors = transform.base_vectors(base) if transform else base
    index = codec_native.Index(vectors, min(8,max(1,math.ceil(budget/base.shape[1]))), 0, True, dataset=='sift', True, budget)
    return index,transform,time.perf_counter()-start


def storage(index, transform):
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        path = Path(temp)/'index.bin'
        reported = index.storage(str(path))
        details = index.storage_details(str(path))
        assert reported == max(details['disk_bytes'], details['resident_bytes']-details['index_shell_bytes'])
        details['index_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        details['adapter_bytes'] = transform.storage_bytes if transform else 0
        details['serving_bytes'] = max(details['disk_bytes'],details['resident_bytes'])+details['adapter_bytes']
        return details


def calibrate():
    import codec_native
    native=build_native()
    assert codec_native.rabitq_high_accuracy and codec_native.compiled_avx512bw
    metadata={'status':'calibrating','affinity':sorted(os.sched_getaffinity(0)), 'compiler_commands':native.compiler_commands,
              'binary_sha256':hashlib.sha256(next(native.directory.glob('codec_native*.so')).read_bytes()).hexdigest(),
              'selection_rule':'First measured point <= exact archived anchor and within 0.5%; storage only. Secant proposals are measured, bounded by observed under/over budgets; if stuck bisect untried bracket. At adjacent boundary or maximum supported budget retain closest measured under-anchor point and report gap. No recall evaluated during calibration.',
              'tolerance':TOLERANCE,'max_attempts':MAX_ATTEMPTS,'data':{}}
    save('meta.json',metadata)
    anchors=list(csv.DictReader((ROOT/'core-results.csv').open()))
    history=json.loads((OUT/'calibration.json').read_text()) if (OUT/'calibration.json').exists() else []
    selections=json.loads((OUT/'selections.json').read_text()) if (OUT/'selections.json').exists() else []
    start=time.perf_counter()
    for dataset in DATASETS:
        base,queries,truth,hashes=load_data(dataset)
        metadata['data'][dataset]=hashes
        for anchor_bits,initial in zip((1,2,4,8),STARTS[dataset]):
            if any(s['dataset']==dataset and s['anchor_bits']==anchor_bits for s in selections): continue
            maximum_budget=8*(base.shape[1]+(dataset=='sift'))
            initial=min(initial,maximum_budget)
            control=next(r for r in anchors if r['dataset']==dataset and r['codec']=='rabitq' and int(r['target_bits'])==anchor_bits)
            limit=float(control['effective_bytes_per_vector'])
            measured={}; budget=initial
            for attempt in range(MAX_ATTEMPTS):
                tick=time.perf_counter()
                index,transform,build_s=build(base,dataset,budget)
                details=storage(index,transform)
                size=details['serving_bytes']/len(base)
                record={'dataset':dataset,'anchor_bits':anchor_bits,'anchor_bytes':limit,'attempt':attempt+1,'target_code_bits':budget,'bytes_per_vector':size,'build_s':build_s,'storage':details,'diagnostics':index.diagnostics(),'wall_s':time.perf_counter()-tick}
                history.append(record); measured[budget]=record
                save('calibration.json',history)
                print('CALIBRATION',json.dumps(record),flush=True)
                del index,transform
                if size<=limit and ((limit-size)/limit<=TOLERANCE or budget==maximum_budget): break
                under=[b for b,r in measured.items() if r['bytes_per_vector']<=limit]
                over=[b for b,r in measured.items() if r['bytes_per_vector']>limit]
                lower=max(under,default=0); upper=min((b for b in over if b>lower),default=maximum_budget+1)
                proposed=max(lower+1,min(upper-1,budget+math.floor((limit-size)*8)))
                if proposed in measured:
                    untried=[b for b in range(lower+1,upper) if b not in measured]
                    if not untried: break
                    proposed=untried[len(untried)//2]
                budget=proposed
            candidates=[r for r in measured.values() if r['bytes_per_vector']<=limit]
            if not candidates: raise RuntimeError(f'No fitting configuration {dataset}/{anchor_bits}')
            selected=max(candidates,key=lambda r:r['bytes_per_vector'])
            selections.append(selected)
            save('selections.json',selections)
    recovery=json.loads((ROOT/'recovery.json').read_text()) if (ROOT/'recovery.json').exists() else {'failed_phase_wall_s':0}
    metadata.update(status='calibrated',calibration_wall_s=time.perf_counter()-start+recovery['failed_phase_wall_s'],recovery=recovery)
    save('meta.json',metadata)


def run_matrix():
    import codec_native
    metadata=json.loads((OUT/'meta.json').read_text())
    wait_start=time.perf_counter()
    while True:
        processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
        competing=[line for line in processes.splitlines() if 'python' in line and 'benchmark/experiments/' in line]
        if not competing: break
        print('Waiting for existing experiment:',competing,flush=True)
        time.sleep(15)
    metadata['other_experiment_wait_s']=time.perf_counter()-wait_start
    (OUT/'host-before-matrix.txt').write_text(subprocess.check_output(['ps','-eo','pid,etime,pcpu,args','--sort=-pcpu'],text=True))
    selections=json.loads((OUT/'selections.json').read_text())
    selected_wall=0
    with (OUT/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS+['target_code_bits','rabitq_anchor_bits','rabitq_anchor_bytes','percent_below_anchor'])
        writer.writeheader()
        for selected in selections:
            dataset=selected['dataset'];budget=selected['target_code_bits']
            base,queries,truth,_=load_data(dataset)
            index,transform,build_s=build(base,dataset,budget)
            details=storage(index,transform)
            assert details==selected['storage']
            selected['selected_build_s']=build_s
            size=details['serving_bytes']/len(base)
            search=L2Search(index,transform.scale,transform) if transform else index
            common=dict(dataset=dataset,d=base.shape[1],base_n=len(base),queries=len(queries),metric='l2' if transform else 'cosine',codec='tieredq',config='fast-native-exact-budget',target_bits=budget/base.shape[1],effective_bits_per_dim=size*8/base.shape[1],effective_bytes_per_vector=size,build_s=build_s,encode_vps=len(base)/index.encode_seconds,rerank=False,seed=0,target_code_bits=budget,rabitq_anchor_bits=selected['anchor_bits'],rabitq_anchor_bytes=selected['anchor_bytes'],percent_below_anchor=100*(selected['anchor_bytes']-size)/selected['anchor_bytes'])
            tick=time.perf_counter()
            for threads in (1,8):
                for batch in (1,32):
                    metrics=measure(search,queries,truth,threads,batch)
                    writer.writerow(common|metrics|dict(threads=threads,batch=batch));stream.flush()
                    print('MEASURED',dataset,budget,threads,batch,metrics,flush=True)
            selected_wall+=build_s+time.perf_counter()-tick
            save('selections.json',selections)
            del search,index,transform
    metadata.update(status='complete',selected_build_and_search_wall_s=selected_wall)
    save('meta.json',metadata)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        calibrate() if sys.argv[1]=='calibrate' else run_matrix()
