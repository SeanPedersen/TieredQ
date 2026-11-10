"""Real-data repro: faiss_reference.build_reference (SQ8 codes via add_sa_codes) on the first 500K Qwen rows, then the
same budget measured one query at a time and as one batch call (measure / measure_batch)."""
import sys, types
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/sean/experiments/TieredQ-issue44-faisscurve-20261110/source")
import faiss
from faiss_reference import build_reference, measure, measure_batch
Q = Path("/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107")
rows, dim = 500000, 1024
codes_all = np.memmap(Q / "base.i8", dtype=np.int8, mode="r", shape=(8840823, dim))
np.asarray(codes_all[:rows]).tofile("/tmp/repro-base.i8")
base = np.zeros((rows, dim), dtype=np.float32)  # only its shape is used with an SQ8 base
def vecs(path, dtype, n):
    raw = np.fromfile(path, dtype=np.int32); w = raw[0]; t = raw.reshape(-1, w + 1)[:n, 1:]
    return t.view(dtype) if dtype != np.int32 else t
queries = np.ascontiguousarray(vecs(Q / "data/test_queries.fvecs", np.float32, 750)[:, :dim])
codes = np.memmap("/tmp/repro-base.i8", dtype=np.int8, mode="r", shape=(rows, dim))
scales = np.fromfile(Q / "scales.f32", dtype=np.float32); offsets = np.fromfile(Q / "offsets.f32", dtype=np.float32)
decoded = codes.astype(np.float32) * scales + offsets
flat = faiss.IndexFlatL2(dim); flat.add(decoded); faiss.omp_set_num_threads(32)
truth = flat.search(queries, 10)[1]
for variant in ("ivf", "ivf-hnsw"):
    args = types.SimpleNamespace(variant=variant, nlist=4096, sample=4096 * 40, seed=42, int8_base=Path("/tmp/repro-base.i8"),
                                 int8_scales=Q / "scales.f32", int8_offsets=Q / "offsets.f32", centroids=None, assignments=None)
    faiss.omp_set_num_threads(32)
    index, _, _ = build_reference(base, args)
    faiss.omp_set_num_threads(1)
    single = measure(index, variant, queries, truth, 64)
    batch = measure_batch(index, variant, queries, truth, 64, 32)
    print("RESULT", variant, "single", single["recall10"], "batch", batch["recall10"], flush=True)
