"""Supplement SIFT with GloVe/OpenAI allocator screening and narrow/wide lookup checks."""
import argparse
import itertools
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import search,recall,timing
from common import append_json,output_arrays,parse_model

TRAINING_ROWS=8192
class Identity:
    def queries(self,x):
        return np.ascontiguousarray(x,dtype=np.float32)

parser=argparse.ArgumentParser(description=__doc__)
for name in ('data','native','results'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--mode',choices=('quality','timing'),required=True)
parser.add_argument('--variant',choices=('register','byte'),default='register')
parser.add_argument('--controlled',action='store_true')
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(8)
os.environ['TIEREDQ_EXPERIMENT_GROUPING']=args.variant
with threadpool_limits(limits=1,user_api='blas'):
    for dataset in ('glove','openai'):
        with np.load(args.data/f'{dataset}-100000-1000-v2.npz') as data:
            base,queries,truth=data['base'],data['queries'],data['truth']
        sample=np.ascontiguousarray(base[np.random.default_rng(0).choice(len(base),TRAINING_ROWS,replace=False)])
        if args.mode=='quality':
            for bits in (4,8):
                for policy in ('marginal','log','cap4'):
                    if bits==8 and policy=='cap4':
                        continue
                    os.environ['TIEREDQ_EXPERIMENT_ALLOCATION']=policy
                    stem=f'{dataset}-{policy}-b{bits}'
                    start=time.perf_counter()
                    raw=native.train_total(sample,bits*base.shape[1],True,0)
                    train_s=time.perf_counter()-start
                    (args.results/f'{stem}.bin').write_bytes(bytes(raw))
                    if bits==8 and policy=='log':
                        assert bytes(raw)==(args.results/f'{dataset}-marginal-b8.bin').read_bytes()
                    codec=native.Codec.load(raw)
                    index=native.Index(codec,True)
                    start=time.perf_counter()
                    index.add(base)
                    add_s=time.perf_counter()-start
                    path=args.results/f'{stem}.tq'
                    index.save(str(path))
                    index=native.Index.load(str(path))
                    scores,ids=search(index,queries)
                    np.savez_compressed(args.results/f'{stem}-quality.npz',scores=scores,ids=ids,recall10=recall(ids,truth))
                    model=parse_model(raw)
                    heldout_ids=np.setdiff1d(np.arange(len(base)),np.random.default_rng(0).choice(len(base),TRAINING_ROWS,replace=False))[:2048]
                    heldout=np.ascontiguousarray(base[heldout_ids])
                    restored=np.empty_like(heldout)
                    codec.reconstruct(heldout,restored)
                    row=dict(dataset=dataset,policy=policy,bits_per_dimension=bits,seed=0,training_rows=TRAINING_ROWS,
                             train_s=train_s,add_s=add_s,recall10=float(recall(ids,truth).mean()),
                             base_reconstruction_mse=float(np.mean((heldout.astype(np.float64)-restored)**2)),
                             bytes_per_vector=(index.storage()+index.model_memory())/len(base),stride=index.stride(),
                             tiers=model['tiers'].tolist(),all8_model_identical=bits==8 and policy=='log')
                    append_json(args.results/'cross-quality.jsonl',row)
                    print(row,flush=True)
        else:
            # One all-document score check and all 1,000 top-10 scores/IDs before timing.
            for bits in (4,8):
                index=native.Index.load(str(args.results/f'{dataset}-marginal-b{bits}.tq'))
                if not args.controlled:
                    full_scores,full_ids=output_arrays(1,len(base))
                    index.search(queries[:1],full_scores,full_ids)
                    full_path=args.results/f'{dataset}-b{bits}-full-register.npz'
                    scores,ids=search(index,queries)
                    if args.variant=='register':
                        np.savez_compressed(full_path,scores=full_scores,ids=full_ids)
                    else:
                        with np.load(full_path) as reference:
                            np.testing.assert_array_equal(reference['scores'],full_scores)
                            np.testing.assert_array_equal(reference['ids'],full_ids)
                        with np.load(args.results/f'{dataset}-marginal-b{bits}-quality.npz') as reference:
                            np.testing.assert_array_equal(reference['scores'],scores)
                            np.testing.assert_array_equal(reference['ids'],ids)
                cases=list(itertools.product(range(3),(((1,8),(32,8)) if args.controlled else ((1,1),(1,8),(32,8)))))
                random.Random(14).shuffle(cases)
                for repeat,(batch,threads) in cases:
                    row=dict(dataset=dataset,bits=bits,grouping=args.variant,repeat=repeat,
                             **timing(native,index,queries,Identity(),truth,batch,threads))
                    append_json(args.results/('cross-isolated-timing.jsonl' if args.controlled else 'cross-timing.jsonl'),row)
                    print(row['dataset'],bits,args.variant,repeat,batch,threads,row['p50_ms'],flush=True)
