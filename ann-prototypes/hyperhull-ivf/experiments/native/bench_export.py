"""Python HullRouter on an exported index dir (same arrays as hullbench): recall + µs/query per config.

Usage: bench_export.py <export_dir> <prefix_dims> T,F,V,R ...
"""
import sys, time
from pathlib import Path
import numpy as np
from hyperhull_ivf.evaluate import recall
from hyperhull_ivf.native_router import make_router

DTYPES = {"float32": np.float32, "uint8": np.uint8, "int64": np.int64}
root = Path(sys.argv[1])
arrays, extra = {}, {}
for line in (root / "meta.txt").read_text().splitlines():
    kind, *rest = line.split()
    if kind == "array":
        name, dtype, *shape = rest
        arrays[name] = np.fromfile(root / f"{name}.bin", dtype=DTYPES[dtype]).reshape([int(v) for v in shape])
    elif kind == "fine_per_coarse":
        extra["fine_per_coarse"] = int(rest[0])
    elif kind == "gamma":
        extra["gamma"] = float(rest[0])
queries, gt = arrays.pop("queries"), arrays.pop("gt")
router = make_router({**arrays, **extra}, prefix_dims=int(sys.argv[2]))
for cfg in sys.argv[3:]:
    T, F, V, R = (int(v) for v in cfg.split(","))
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        _, ids, _ = router.search(queries, 10, shortlist=T, fine_cells=F, rerank=V, refine=R, n_threads=1)
        best = min(best, time.perf_counter() - t)
    print(f"{cfg:<20} recall {recall(ids, gt):.3f}  {best / len(queries) * 1e6:7.1f} us", flush=True)
