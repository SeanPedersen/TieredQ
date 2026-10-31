"""Prepare openai1m at full size and check it against the 100k openai set (queries, base rows, truth spot check)."""
import sys, time, hashlib; sys.path.insert(0, ".")
from pathlib import Path; import numpy as np
from data import prepare, SPLIT_SEED
t0=time.time()
b,q,t=prepare(Path("../../data"),"openai1m")
print(b.shape,q.shape,t.shape, f"{time.time()-t0:.0f}s", flush=True)
o=np.load("/home/sean/experiments/TieredQ-issue4-memory-20261028/data/openai-100000-1000-v2.npz")
print("queries equal 100k:", np.array_equal(o["queries"],q))
rng=np.random.default_rng(SPLIT_SEED); idx=np.sort(rng.choice(1_000_000,101_000,replace=False)); pos=np.sort(rng.choice(len(idx),1000,replace=False))
mask=np.ones(len(idx),bool); mask[pos]=False; sampled=idx[mask][:100_000]
base_rows=np.setdiff1d(np.arange(1_000_000),idx[pos]); slot=np.searchsorted(base_rows,sampled)
print("100k base rows equal 1M rows:", np.array_equal(b[slot],o["base"]))
s=q[:5]@b.T; top=np.argsort(-s,axis=1)[:,:10]
print("truth spot check:", np.array_equal(np.sort(top,1),np.sort(t[:5].astype(np.int64),1)))
print("norms", float(np.abs(np.linalg.norm(b[::997],axis=1)-1).max()))
