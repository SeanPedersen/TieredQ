"""Per-stage single-thread time of the C++ hull router at the benchmark's chosen configs (full DBpedia)."""
import numpy as np
from hyperhull_native import SIMD_LEVEL
from hyperhull_ivf.native_router import make_router, native_arrays
from hyperhull_ivf.real_data import load_dbpedia
from hyperhull_ivf.two_stage import build_two_level

CONFIGS = {0.90: (64, 128, 80), 0.95: (64, 512, 160), 0.99: (256, 2048, 640)}
STAGES = ("project", "coarse", "fine", "sketch", "rerank")
d = load_dbpedia(n_queries=1000)
router = make_router(native_arrays(build_two_level(np.asarray(d.base, dtype=np.float32), 1024, 32)))
queries = np.ascontiguousarray(d.queries[500:], dtype=np.float32)
print(f"kernels {SIMD_LEVEL}; mean µs per query per stage")
print(f"{'target':<8}" + "".join(f"{s:>9}" for s in STAGES) + f"{'total':>9}{'sketched':>10}{'reranked':>10}")
for target, (t, f, v) in CONFIGS.items():
    router.search(queries, 10, shortlist=t, fine_cells=f, rerank=v)  # warm caches
    _, _, st = router.search(queries, 10, shortlist=t, fine_cells=f, rerank=v)
    us = st[:, 2:].mean(0) / 1e3
    print(f"R>={target:<5}" + "".join(f"{x:9.1f}" for x in us) + f"{us.sum():9.1f}{st[:, 0].mean():10.0f}{st[:, 1].mean():10.0f}")
