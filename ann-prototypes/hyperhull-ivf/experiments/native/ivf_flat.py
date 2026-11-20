"""FAISS IVF1024,Flat and IVF-PQ4fs+refine wall time (single thread) on the exported test queries."""
import sys, time
import faiss
import numpy as np
from hyperhull_ivf.evaluate import recall
from hyperhull_ivf.real_data import load_dbpedia

export_dir = sys.argv[1]
d = load_dbpedia(n_queries=1000)
base = np.asarray(d.base, dtype=np.float32)
queries = np.fromfile(f"{export_dir}/queries.bin", dtype=np.float32).reshape(-1, base.shape[1])
gt = np.fromfile(f"{export_dir}/gt.bin", dtype=np.int64).reshape(len(queries), 10)
faiss.omp_set_num_threads(36)
index = faiss.index_factory(base.shape[1], "IVF1024,Flat")
index.train(base); index.add(base)
faiss.omp_set_num_threads(1)
ivf = faiss.extract_index_ivf(index)
print("FAISS IVF1024,Flat single thread: nprobe recall us/query")
for nprobe in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
    ivf.nprobe = nprobe
    best = 1e9
    for _ in range(2):
        t = time.perf_counter(); _, ids = index.search(queries, 10); best = min(best, time.perf_counter() - t)
    print(f"  {nprobe:4d} {recall(ids, gt):.3f} {best / len(queries) * 1e6:8.1f}", flush=True)
