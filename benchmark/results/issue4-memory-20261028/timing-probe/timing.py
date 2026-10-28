import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import codec_native
from data import prepare
from measure import measure, L2Search
from l2_transform import L2Transform
dataset, path = sys.argv[1], sys.argv[2]
with threadpool_limits(limits=1):
    _, queries, truth = prepare(Path("data"), dataset)
    index = codec_native.Index.load(path, True)
    if dataset == "sift":
        with np.load("probe/sift-transform.npz") as s:
            t = L2Transform(float(s["scale"]), s["base_map"], s["query_map"])
        index = L2Search(index, t.scale, t)
    print(" ".join(f"{measure(index, queries, truth, 1, b)[chr(113)+chr(112)+chr(115)]:.1f}" for b in (1, 32)))
