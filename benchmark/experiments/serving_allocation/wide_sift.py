"""Check and time 256-entry whole-coordinate FP32 tables across mixed/wide SIFT tiers."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import timing,search
from common import append_json,load_cache,output_arrays,read_transform
parser=argparse.ArgumentParser(description=__doc__)
for name in ('data','native','previous','results'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--variant',choices=('register','byte'),required=True)
parser.add_argument('--controlled',action='store_true')
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(8)
os.environ['TIEREDQ_EXPERIMENT_GROUPING']=args.variant
with threadpool_limits(limits=1,user_api='blas'):
    original,queries,truth=load_cache(args.data)
    transform=read_transform(args.previous/'adapter.npz')
    probes=transform.queries(queries)
    for budget in (256,512,1032):
        index=native.Index.load(str(args.results/f'marginal-b{budget}-s0.tq'))
        if not args.controlled:
            scores,ids=search(index,probes)
            with np.load(args.results/f'marginal-b{budget}-s0-quality.npz') as reference:
                np.testing.assert_array_equal(scores,reference['scores'])
                np.testing.assert_array_equal(ids,reference['ids'])
            full_scores,full_ids=output_arrays(1,len(original))
            index.search(probes[:1],full_scores,full_ids)
            path=args.results/f'wide-b{budget}-full-register.npz'
            if args.variant=='register':
                np.savez_compressed(path,scores=full_scores,ids=full_ids)
            else:
                with np.load(path) as reference:
                    np.testing.assert_array_equal(full_scores,reference['scores'])
                    np.testing.assert_array_equal(full_ids,reference['ids'])
        cases=[(r,b,t) for r in range(3) for b,t in (((1,8),(32,8)) if args.controlled else ((1,1),(1,8),(32,8)))]
        random.Random(14).shuffle(cases)
        for repeat,batch,threads in cases:
            row=dict(budget=budget,grouping=args.variant,repeat=repeat,
                     all_document_scores_equal=True,**timing(native,index,queries,transform,truth,batch,threads))
            append_json(args.results/('wide-isolated-timing.jsonl' if args.controlled else 'wide-timing.jsonl'),row)
            print(budget,args.variant,repeat,batch,threads,row['p50_ms'],flush=True)
