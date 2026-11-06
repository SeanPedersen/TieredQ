"""Prepare a nested 1M GloVe split from cached HDF5, preserving queries and the original 100K rows."""
import hashlib
import json
from pathlib import Path
import sys
import time

B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B))
import h5py
import numpy as np
from threadpoolctl import threadpool_limits
from data import normalize,ground_truth,verify_checksum,SOURCES

w=Path(sys.argv[1]).resolve()
source=Path('/home/sean/experiments/diskovec-ntu-20261021/work/ntu-comparison/rabitq-turboquant-comparison/ann/data/raw/glove-200-angular.hdf5')
out=w/'data/glove1m-1000000-1000-v1.npz'
assert not out.exists(),'Refuse to overwrite a prepared split'
start=time.monotonic()
verify_checksum(source,SOURCES['glove']['sha256'])
with np.load(w/'data/glove-100000-1000-v2.npz') as arrays:
    old_base,old_queries=arrays['base'],arrays['queries']
with h5py.File(source) as handle:
    total,d=handle['train'].shape
    assert d==200 and total>=1_000_000
    rng=np.random.default_rng(42)
    old_ids=np.sort(rng.choice(total,100_000,replace=False))
    remaining=np.setdiff1d(np.arange(total),old_ids,assume_unique=True)
    extra=rng.choice(remaining,900_000,replace=False)
    selected=np.sort(np.concatenate([old_ids,extra]))
    base=normalize(np.asarray(handle['train'],dtype=np.float32)[selected])
    queries=normalize(np.asarray(handle['test'][:1000],dtype=np.float32))
positions=np.searchsorted(selected,old_ids)
np.testing.assert_array_equal(base[positions],old_base)
np.testing.assert_array_equal(queries,old_queries)
print('Verified original 100K base is byte-identical subset; all 1000 queries unchanged',flush=True)
with threadpool_limits(limits=8):
    truth=ground_truth(base,queries,False)
np.savez(out,base=base,queries=queries,truth=truth)
np.save(w/'data/glove1m-source-row-ids.npy',selected.astype(np.uint32))
np.save(w/'data/glove1m-original100k-positions.npy',positions.astype(np.uint32))
manifest=dict(dataset='glove1m',d=200,base_n=1_000_000,queries=1000,metric='cosine',
    source=str(source),source_sha256=SOURCES['glove']['sha256'],source_rows=total,split_seed=42,
    split='Original sorted seed-42 100K draw, extended by a 900K draw from remaining rows using the same RNG stream; sorted union',
    original100k_subset=True,queries_byte_identical=True,
    array_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in [('base',base),('queries',queries),('truth',truth)]},
    cache_sha256=hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),preparation_wall_s=time.monotonic()-start)
out.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
(w/'glove1m-data-provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
