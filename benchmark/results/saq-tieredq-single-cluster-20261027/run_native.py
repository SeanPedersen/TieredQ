"""Time SAQ's native pruned flat scan separately; retain full-accurate results as primary."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from adapter import SAQ
from run import HERE,load,save
from evaluation import evaluate,predictions,recall,SCHEDULE


def main():
    original=list(csv.DictReader((HERE/'results.csv').open()))
    output=HERE/'native-pruned';output.mkdir(exist_ok=True)
    rows=[]
    for dataset in ('gemini','glove','openai','sift'):
        base,queries,truth=load(dataset)
        with np.load(HERE/f'pca-{dataset}.npz') as a:basis,mean=a['basis'],a['mean']
        for bits in (1,2,4):
            directory=output/f'{dataset}-saq{bits}';directory.mkdir(exist_ok=True)
            group=[r for r in original if r['dataset']==dataset and int(r['target_bits'])==bits]
            seed0=next(r for r in group if r['seed']=='0')
            index=SAQ(base,basis,mean,float(seed0['native_padded_bits']),0,directory/'model-seed0.bin')
            assert index.bytes_per_vector==float(seed0['effective_bytes_per_vector'])
            index.fullscan=False
            expected=np.load(HERE/f'{dataset}-saq{bits}/ids-native-pruned-seed0.npy')
            assert np.array_equal(predictions(index,queries),expected)
            common={k:v for k,v in seed0.items() if k not in ('recall10','qps','p50_ms','p95_ms','threads','batch')}
            common.update(codec='saq-native',config='single-cluster-native-pruned-scan',seed=0)
            rows+=evaluate(index,queries,truth,common,directory)
            index.close()
            for seed in (1,2):
                ids=np.load(HERE/f'{dataset}-saq{bits}/ids-native-pruned-seed{seed}.npy')
                source=next(r for r in group if r['seed']==str(seed))
                common={k:v for k,v in source.items() if k not in ('recall10','qps','p50_ms','p95_ms','threads','batch')}
                common.update(codec='saq-native',config='single-cluster-native-pruned-scan',seed=seed)
                np.save(directory/f'ids-seed{seed}.npy',ids)
                for t,b in SCHEDULE:rows.append(common|dict(recall10=recall(truth,ids),qps=math.nan,p50_ms=math.nan,p95_ms=math.nan,threads=t,batch=b))
            with (output/'results.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(output/'meta.json',dict(status='complete',rows=len(rows),seed12_recall='Reused captured native predictions; seed-0 rebuild reproduces all 1,000 native IDs and storage before timing',
        construction='Build/encode fields reference the corresponding original three-seed construction, not the timing-only rebuild'))
    print('NATIVE COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
