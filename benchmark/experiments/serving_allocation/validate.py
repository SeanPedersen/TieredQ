"""Validate lookup boundaries, append/load, prior rankings, and fixed Gaussian tables."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiny_budgets'))
from common import output_arrays,parse_model,write_json

parser=argparse.ArgumentParser(description=__doc__)
for name in ('native','previous','results'):
    parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(1)
os.environ['TIEREDQ_EXPERIMENT_ALLOCATION']='marginal'
rng=np.random.default_rng(14)
sample=rng.normal(size=(512,16)).astype(np.float32)
raw=native.train_total(sample,16,True,14)
codec=native.Codec.load(raw)
checks=[]
with tempfile.TemporaryDirectory() as folder:
    for n in (1,15,16,17,255,256,257,4095,4096,4097):
        base=rng.normal(size=(n,16)).astype(np.float32)
        index=native.Index(codec,True)
        split=n//2
        if split:
            index.add(base[:split])
        index.add(base[split:])
        path=Path(folder)/'index.tq'
        index.save(str(path))
        index=native.Index.load(str(path))
        probes=np.vstack([rng.normal(size=(4,16)),np.zeros((1,16))]).astype(np.float32)
        ranked=[]
        for grouping in ('register','byte'):
            os.environ['TIEREDQ_EXPERIMENT_GROUPING']=grouping
            scores,ids=output_arrays(len(probes),n)
            index.search(probes,scores,ids)
            ranked.append((scores,ids))
        for a,b in zip(ranked[0],ranked[1],strict=True):
            np.testing.assert_array_equal(a,b)
        checks.append(n)
reference=parse_model(list((args.previous/'b128-s0.bin').read_bytes()))['levels']
recipes=list(args.results.glob('*.bin'))
for path in recipes:
    model=parse_model(list(path.read_bytes()))
    assert sum(int(t[1])*int(t[2]) for t in model['tiers'])==model['budget']
    for a,b in zip(reference,model['levels'],strict=True):
        np.testing.assert_array_equal(a,b)
for budget in (64,80,128):
    with np.load(args.previous/f'b{budget}-s0-fast-quality.npz') as prior, np.load(args.results/f'lookup-b{budget}-register.npz') as current:
        np.testing.assert_array_equal(prior['top10_ids'],current['ids'])
write_json(args.results/'validation.json',dict(boundary_append_save_load_equal_sizes=checks,
           prior_top10_equal_budgets=[64,80,128],gaussian_recipes_checked=len(recipes),
           exact_budget_checks=True,lookup_full_document_scores_and_rankings_equal=True))
print('Validation passed',flush=True)
