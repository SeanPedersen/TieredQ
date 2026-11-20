"""Export full DBpedia two-level indexes for the standalone C++ bench.

Usage: export_index.py <out_prefix> m<fine>:p<pca>:b<bits> ...   e.g. m32:p256:b4 m16:p256:b2
Each variant goes to <out_prefix>_m<fine>_p<pca>_b<bits>/. Two-level k-means is built once per m.
"""
import sys
import numpy as np
from hyperhull_ivf.faiss_full import exact_gt
from hyperhull_ivf.native_router import export, native_arrays
from hyperhull_ivf.real_data import load_dbpedia
from hyperhull_ivf.two_stage import build_two_level

d = load_dbpedia(n_queries=1000)
base = np.asarray(d.base, dtype=np.float32)
queries = np.ascontiguousarray(d.queries[500:], dtype=np.float32)
gt = exact_gt(base, queries)
variants = [tuple(int(part[1:]) for part in v.split(":")) for v in sys.argv[2:]]
for m in sorted({v[0] for v in variants}):
    index = build_two_level(base, 1024, m)
    for _, pca, bits in [v for v in variants if v[0] == m]:
        out = f"{sys.argv[1]}_m{m}_p{pca}_b{bits}"
        export(native_arrays(index, pca_dims=pca, sketch_bits=bits), queries, gt, out)
        print("exported", out, flush=True)
