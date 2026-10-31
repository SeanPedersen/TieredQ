from pathlib import Path; import numpy as np, hashlib, time, resource
from data import prepare
t0=time.time()
b,q,t=prepare(Path("../../data"),"openai1m",max_base=20000)
print(b.shape,q.shape,t.shape,b.dtype, np.linalg.norm(b[:3],axis=1), f"{time.time()-t0:.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")
old="/home/sean/experiments/TieredQ-issue4-memory-20261028/data/"
o=np.load(old+"openai-100000-1000-v2.npz")
print("queries equal 100k:", np.array_equal(o["queries"],q))
# 100k base rows are a subset of the 1M base: check the first sampled row matches
idx=np.load(old+"openai-sampled-indices.npy"); pos=np.load(old+"openai-test-positions.npy")
mask=np.ones(len(idx),bool); mask[pos]=False; first=idx[mask][:5]
qrows=idx[pos]; base_rows=np.setdiff1d(np.arange(1_000_000),qrows)[:20000]
where={r:i for i,r in enumerate(base_rows)}
for k,r in enumerate(first):
    if r in where: print("base row",r,"equal:",np.array_equal(b[where[r]],o["base"][k]))
