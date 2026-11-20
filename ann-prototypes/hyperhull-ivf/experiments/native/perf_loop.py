"""Repeated searches at the R>=0.90 config for perf sampling (index built first, then a marker line)."""
import sys, time
import numpy as np
from hyperhull_ivf.native_router import make_router, native_arrays
from hyperhull_ivf.real_data import load_dbpedia
from hyperhull_ivf.two_stage import build_two_level
T, F, V = (int(x) for x in sys.argv[1:4])
d = load_dbpedia(n_queries=1000)
router = make_router(native_arrays(build_two_level(np.asarray(d.base, dtype=np.float32), 1024, 32)))
queries = np.ascontiguousarray(d.queries[500:], dtype=np.float32)
print("SEARCH START", flush=True)
t = time.perf_counter()
for _ in range(int(sys.argv[4]) if len(sys.argv) > 4 else 40):
    router.search(queries, 10, shortlist=T, fine_cells=F, rerank=V)
print(f"ms/query {(time.perf_counter() - t) / (int(sys.argv[4]) if len(sys.argv) > 4 else 40) / len(queries) * 1e3:.3f}", flush=True)
