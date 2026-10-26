"""Measure disjoint-base reconstruction distortion for every SIFT allocation recipe."""
import argparse
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiny_budgets'))
from common import append_json,load_cache,read_transform
parser=argparse.ArgumentParser(description=__doc__)
for name in ('data','native','previous','results'):
    parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(8)
with threadpool_limits(limits=1,user_api='blas'):
    original,_,_=load_cache(args.data)
    transform=read_transform(args.previous/'adapter.npz')
    for seed in (0,1,2):
        training=np.random.default_rng(seed).choice(len(original),32768,replace=False)
        ids=np.random.default_rng(1409).choice(np.setdiff1d(np.arange(len(original)),training),2048,replace=False)
        base=transform.base_vectors(original[ids])
        for path in sorted(args.results.glob(f'*-b*-s{seed}.bin')):
            codec=native.Codec.load(list(path.read_bytes()))
            restored=np.empty_like(base)
            codec.reconstruct(base,restored)
            row=dict(model=path.stem,seed=seed,base_rows=2048,disjoint_from_codec_training=True,
                     reconstruction_mse=float(np.mean((base.astype(np.float64)-restored)**2)))
            append_json(args.results/'reconstruction.jsonl',row)
        print('Reconstruction seed',seed,'complete',flush=True)
