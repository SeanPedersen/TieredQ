"""Extend the audited matrix with production Qdrant TQ+ and storage-only TieredQ selections."""
import csv
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
import importlib.util
from threadpoolctl import threadpool_limits
from runtime import ROOT,HERE,load,build,save,sha
from qdrant_adapter import Qdrant
from evaluation import evaluate,predictions
from selection import select_budget
BUDGETS=(1,2,4)
DATASETS=('gemini','glove','openai','sift')


def write(rows):
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def common(dataset,base,queries,bits,seed,codec,size,build_s,encode_vps,code_bits=None,anchor=None):
    return dict(dataset=dataset,d=base.shape[1],base_n=len(base),queries=len(queries),
        metric='l2' if dataset=='sift' else 'cosine',codec=codec,config='native-production-plus' if codec=='turboquant' else 'fast-native-exact-code-budget',
        target_bits=bits,code_bits=code_bits or bits*base.shape[1],anchor_bits=anchor or '',
        effective_bits_per_dim=size*8/base.shape[1],effective_bytes_per_vector=size,
        build_s=build_s,encode_vps=encode_vps,rerank=False,seed=seed)


def main():
    assert sorted(os.sched_getaffinity(0))==list(range(8))
    spec=importlib.util.spec_from_file_location('archived_runtime',ROOT/'storage-sweep/runtime.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    meta=old.verify_environment()
    meta.update(status='running',started=datetime.now(timezone.utc).isoformat(),qdrant_revision='222b2e44ab1c3f17e8246396ead6558e45041df7',
        qdrant_binary_sha256=sha(HERE/'libqdrant_bench.so'),seed_policy='Only calibration sampler is seeded; production rotation seeds unchanged')
    save(HERE/'meta.json',meta)
    reference=list(csv.DictReader((ROOT/'output/results.csv').open()))
    rows=list(csv.DictReader((HERE/"results.csv").open())) if (HERE/"results.csv").exists() else []
    for dataset in DATASETS:
        base,queries,truth=load(dataset)
        print('INPUT VERIFIED',dataset,base.shape,flush=True)
        for bits in BUDGETS:
            directory=HERE/f'{dataset}-turboquant{bits}';directory.mkdir(exist_ok=True)
            anchor=None
            for seed in (0,1,2):
                retained=[r for r in rows if r['dataset']==dataset and r['codec']=='turboquant' and int(r['code_bits'])==bits*base.shape[1] and int(r['seed'])==seed]
                if retained:
                    assert len(retained)==4
                    if seed==0:anchor=float(retained[0]['effective_bytes_per_vector'])
                    print('REUSED completed native capture',dataset,bits,seed,flush=True)
                    continue
                index=Qdrant(base,bits,directory/f'model-seed{seed}',seed,dataset=='sift')
                details=index.details();save(directory/f'storage-seed{seed}.json',details)
                size=details['serving_bytes']/len(base)
                if seed==0:anchor=size
                record=common(dataset,base,queries,bits,seed,'turboquant',size,index.build_s,len(base)/index.encode_seconds)
                rows+=evaluate(index,queries,truth,record,directory);write(rows)
                index.close()
            selected_dir=HERE/f'{dataset}-tieredq-tq{bits}';selected_dir.mkdir(exist_ok=True)
            main=[r for r in reference if r['dataset']==dataset and r['codec']=='tieredq' and r['seed']=='0' and r['threads']=='8' and r['batch']=='32']
            anchors=[dict(code_bits=int(float(r['target_bits'])*base.shape[1]),bytes=float(r['effective_bytes_per_vector'])) for r in main]
            anchors.sort(key=lambda a:a['code_bits'])
            selected=select_budget(anchor,anchors,lambda b:build(base,b,0,selected_dir,dataset=='sift'),lambda a:print('STORAGE',dataset,bits,a,flush=True),8*base.shape[1])
            save(selected_dir/'selection.json',dict(dataset=dataset,anchor_bits=bits,cap_bytes=anchor,selected_code_bits=selected.code_bits,
                selected_bytes_per_vector=selected.built.bytes_per_vector,in_target_band=selected.in_target_band,attempts=selected.attempts,
                policy='Seed-0 storage only before recall; 99.0–99.5% target band, up to 16 attempts; largest measured feasible fallback; integer code budgets 1..8*d'))
            for seed in (0,1,2):
                built=selected.built if seed==0 else build(base,selected.code_bits,seed,selected_dir,dataset=='sift')
                record=common(dataset,base,queries,selected.code_bits/base.shape[1],seed,'tieredq',built.bytes_per_vector,
                    built.build_s,built.encode_vps,selected.code_bits,bits)
                rows+=evaluate(built.index,queries,truth,record,selected_dir);write(rows)
                if seed==0:selected.built=None
                del built
        del base,queries,truth
    meta.update(status='complete',completed=datetime.now(timezone.utc).isoformat(),rows=len(rows))
    save(HERE/'meta.json',meta)
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
