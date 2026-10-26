"""Measure separate instrumented phases for lookup and allocation layouts; not latency evidence."""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiny_budgets'))
from common import append_json,load_cache,output_arrays,read_transform

parser=argparse.ArgumentParser(description=__doc__)
for name in ('data','native','previous','results'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--output-name',default='phases.jsonl')
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(1)
with threadpool_limits(limits=1,user_api='blas'):
    _,queries,_=load_cache(args.data)
    transform=read_transform(args.previous/'adapter.npz')
    probes=transform.queries(queries[:32])
    cases=[(args.previous/f'b{b}-s0-fast.tq',f'lookup-b{b}',g) for b in (64,80,128) for g in ('register','byte')]
    cases.extend((args.results/f'{p}-b{b}-s0.tq',f'{p}-b{b}','register') for b in (128,192,256,512,1032) for p in ('marginal','log','cap4') if (args.results/f'{p}-b{b}-s0.tq').exists())
    for path,name,grouping in cases:
        os.environ['TIEREDQ_EXPERIMENT_GROUPING']=grouping
        index=native.Index.load(str(path))
        for batch in (1,32):
            scores,ids=output_arrays(batch)
            index.search(probes[:batch],scores,ids)
            expected=ids.copy()
            native.profile_reset()
            start=time.perf_counter()
            index.search(probes[:batch],scores,ids)
            wall=time.perf_counter()-start
            phases=native.profile_read()
            np.testing.assert_array_equal(ids,expected)
            row=dict(name=name,grouping=grouping,batch=batch,threads=1,wall_s=wall,
                     **dict(zip(('preparation_s','scan_s','selection_s','merge_s','tables_s'),phases,strict=True)))
            append_json(args.results/args.output_name,row)
            print(row,flush=True)
