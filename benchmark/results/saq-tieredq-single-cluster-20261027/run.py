"""Run full-scan SAQ on archived inputs; reuse separately audited TieredQ measurements."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits
from adapter import SAQ
HERE=Path(__file__).resolve().parent
REFRESH=Path('/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542')
sys.path.insert(0,str(REFRESH/'source/benchmark'))
from data import prepare
DATA=Path('/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data')
GEMINI=Path('/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R/TieredQ/benchmark/data/gemini-prepared.npz')
from evaluation import evaluate,predictions,recall


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(dataset):
    if dataset=='gemini':
        with np.load(GEMINI) as a:arrays=tuple(a[k] for k in ('base','queries','truth'))
    else:arrays=prepare(DATA,dataset,None,1000)
    hashes={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in zip(('base','queries','truth'),arrays)}
    assert hashes==json.loads((REFRESH/'output/inputs.json').read_text())[dataset]['sha256']
    save(HERE/f'inputs-{dataset}.json',hashes)
    np.save(HERE/f'truth-{dataset}.npy',arrays[2])
    return arrays


def pca(base):
    tick=time.perf_counter()
    mean=base.mean(axis=0,dtype=np.float64)
    covariance=base.astype(np.float64).T@base.astype(np.float64)/len(base)-np.outer(mean,mean)
    values,vectors=np.linalg.eigh(covariance)
    basis=np.ascontiguousarray(vectors[:,::-1],dtype=np.float32)
    mean=np.ascontiguousarray(mean,dtype=np.float32)
    return basis,mean,time.perf_counter()-tick


def write(rows):
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert sorted(os.sched_getaffinity(0))==list(range(8))
    meta=dict(status='running',host=platform.node(),started=datetime.now(timezone.utc).isoformat(),affinity=list(range(8)),
        saq_revision='2163ebcedd0ad9c9f4de326e6ca7a860f9eafe52',tieredq_revision='575e167a0ad66920119bd0d0836b6c0c92c0a4e5',
        binary_sha256=sha(HERE/'build/libsaq_bench.so'),seed_mapping='srand(seed+1) controls upstream Eigen Random rotations; full-corpus PCA is fixed',
        pca='Full-corpus centered covariance in FP64, full orthonormal eigenbasis in descending variance order, serving matrix and mean FP32; no dimensional truncation',
        scoring='One cluster; every compressed vector receives production compAccurateDist; native ResultPool top-10, no native pruning or FP32 backing vectors',
        compiler='GCC 16.2.1, C++20, -O3 -march=native -ffp-contract=off; explicit numerical policy instead of upstream -Ofast')
    save(HERE/'meta.json',meta)
    rows=[]
    for dataset in ('gemini','glove','openai','sift'):
        base,queries,truth=load(dataset)
        print('INPUT VERIFIED',dataset,base.shape,flush=True)
        basis,mean,pca_s=pca(base)
        np.savez(HERE/f'pca-{dataset}.npz',basis=basis,mean=mean)
        save(HERE/f'pca-{dataset}.json',dict(seconds=pca_s,orthogonality_max_error=float(np.max(np.abs(basis.T@basis-np.eye(base.shape[1]))))))
        print('PCA',dataset,pca_s,flush=True)
        for bits in (1,2,4):
            directory=HERE/f'{dataset}-saq{bits}';directory.mkdir(exist_ok=True)
            for seed in (0,1,2):
                model=directory/f'model-seed{seed}.bin'
                padded=(base.shape[1]+63)//64*64
                native_bits=bits*base.shape[1]/padded
                index=SAQ(base,basis,mean,native_bits,seed,model)
                save(directory/f'storage-seed{seed}.json',dict(resident_bytes=index.resident_bytes,disk_bytes=index.disk_bytes,
                    bytes_per_vector=index.bytes_per_vector,model_sha256=sha(model)))
                common=dict(dataset=dataset,d=base.shape[1],base_n=len(base),queries=len(queries),metric='l2' if dataset=='sift' else 'normalized-cosine-via-l2',
                    codec='saq',config='single-cluster-full-accurate-scan',target_bits=bits,native_padded_bits=native_bits,seed=seed,
                    effective_bytes_per_vector=index.bytes_per_vector,effective_bits_per_dim=index.bytes_per_vector*8/base.shape[1],
                    build_s=pca_s+index.build_s,pca_s=pca_s,encode_vps=len(base)/index.encode_seconds)
                rows+=evaluate(index,queries,truth,common,directory);write(rows)
                index.fullscan=False
                native=predictions(index,queries)
                np.save(directory/f'ids-native-pruned-seed{seed}.npy',native)
                full=np.load(directory/f'ids-seed{seed}.npy')
                save(directory/f'pruning-control-seed{seed}.json',dict(full_recall=recall(truth,full),native_recall=recall(truth,native),
                    identical_rows=int(np.all(full==native,axis=1).sum()),queries=len(queries)))
                index.close()
        print('DATASET COMPLETE',dataset,flush=True)
    meta.update(status='complete',completed=datetime.now(timezone.utc).isoformat(),rows=len(rows))
    save(HERE/'meta.json',meta)
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
