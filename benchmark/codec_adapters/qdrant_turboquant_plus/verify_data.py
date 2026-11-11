"""Independent full-width FP32 top-k check and prepared-array identity audit."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
root=Path(sys.argv[1]);out=Path(sys.argv[2])
with np.load(root/'gemini-prepared.npz') as data:
    base,queries,truth=(data[k] for k in ('base','queries','truth'))
manifest=json.loads((root/'gemini-manifest.json').read_text())
for name,array in [('base',base),('queries',queries),('truth',truth)]:
    assert hashlib.sha256(array.tobytes()).hexdigest()==manifest['array_sha256'][name]
assert np.isfinite(base).all() and np.isfinite(queries).all()
assert np.max(np.abs(np.linalg.norm(base,axis=1)-1))<1e-6
assert np.max(np.abs(np.linalg.norm(queries,axis=1)-1))<1e-6
selected=np.array([0,1,31,32,65,127,255,511,777,999])
with threadpool_limits(limits=1):
    scores=queries[selected]@base.T
independent=np.argsort(-scores,axis=1,kind='stable')[:,:10]
assert all(set(a)==set(b) for a,b in zip(independent,truth[selected],strict=True))
report=dict(status='PASS',array_hashes_verified=True,finite_normalized_arrays=True,
    independent_full_corpus_argsort_queries=selected.tolist(),top10_sets_equal=True,
    base_norm_range=[float(np.min(np.linalg.norm(base,axis=1))),float(np.max(np.linalg.norm(base,axis=1)))])
out.write_text(json.dumps(report,indent=2)+'\n');print(report)
