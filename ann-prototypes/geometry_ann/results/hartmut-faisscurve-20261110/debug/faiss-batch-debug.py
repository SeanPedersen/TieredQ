import os, sys, types
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/sean/experiments/TieredQ-issue44-faisscurve-20261110/source")
import faiss
from faiss_reference import build_reference
E = Path("/home/sean/experiments"); Q = E / "TieredQ-issue44-msmarco8m-20261107"
base = np.memmap(Q / "data/base.f32" if (Q / "data/base.f32").exists() else "/dev/null", dtype=np.float32, mode="r") if False else None
class Shape:  # build_reference only reads base.shape with an SQ8 base
    shape = (8840823, 1024)
    size = 8840823 * 1024
raw = np.fromfile(Q / "data/test_queries.fvecs", dtype=np.int32); w = raw[0]
queries = np.ascontiguousarray(raw.reshape(-1, w + 1)[:64, 1:].view(np.float32)[:, :1024])
args = types.SimpleNamespace(variant="ivf", nlist=65536, sample=0, seed=42, int8_base=Q / "base.i8", int8_scales=Q / "scales.f32",
                             int8_offsets=Q / "offsets.f32", centroids=E / "TieredQ-issue44-routerduel-20261110/results/64k/centroids.f32",
                             assignments=E / "TieredQ-issue44-finelists-20261110/results/lloyd-64k/labels.u32")
faiss.omp_set_num_threads(32)
index, _, _ = build_reference(np.broadcast_to(np.float32(0), (8840823, 1024)), args)
index.nprobe = 64
faiss.omp_set_num_threads(1)
single = np.vstack([index.search(q[None, :], 10)[1] for q in queries])
raw_truth = np.fromfile(Q / "data/test_truth.ivecs", dtype=np.int32); tw = raw_truth[0]
truth = raw_truth.reshape(-1, tw + 1)[:750, 1:11]
test = np.ascontiguousarray(raw.reshape(-1, w + 1)[:750, 1:].view(np.float32)[:, :1024])
import time
os.sched_setaffinity(0, range(os.cpu_count()))
faiss.omp_set_num_threads(32)
for budget in (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072):
    index.nprobe = budget
    for mode in (1, 2, 3):
        index.parallel_mode = mode
        index.search(test, 10)
        t = time.perf_counter(); found = index.search(test, 10)[1]; elapsed = time.perf_counter() - t
        recall = sum(len(set(r) & set(e)) for r, e in zip(found, truth)) / (10 * len(test))
        print(f"FLATBATCH budget={budget} mode={mode} recall={recall:.4f} qps={len(test) / elapsed:.0f}", flush=True)
