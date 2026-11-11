"""Faiss 1.15.1 vs 1.16.0 on one fixed partition (a Faiss-vs-Faiss diagnostic, not a scorecard row): do the 1.15.1
baselines of #52 still hold? 1.16.0 ships the fix for the nested-OpenMP database-parallel kNN bug (faiss#5702, #5824)
that broke parallel_mode 0 batches with a flat quantizer at 32K+ lists. Builds IVF-SQ8 on the given 64K Lloyd partition
(given centroids and list ids, shared SQ8 codes) with a flat and an HNSW quantizer, then measures single queries alone
on pinned core 2 and 32-thread batches in parallel modes 0-3, on the 750 held-out queries.
Usage: faissver.py OUT_JSON"""
import json, os, sys, types
from pathlib import Path
import numpy as np
SOURCE = Path("/home/sean/experiments/TieredQ-issue44-faisscurve-20261110/source")
sys.path.insert(0, str(SOURCE))
import faiss
from faiss_reference import build_reference, measure, measure_batch

E = Path("/home/sean/experiments"); Q = E / "TieredQ-issue44-msmarco8m-20261107"
ROWS, DIM, NLIST, SEARCH_CPU, THREADS = 8840823, 1024, 65536, 2, 32
SINGLE = {"ivf": (128, 512, 2048), "ivf-hnsw": (128, 256, 1024)}
BATCH = (128, 512)


def vecs(path, rows, width):
    raw = np.fromfile(path, dtype=np.int32); w = raw[0]
    return raw.reshape(-1, w + 1)[:rows, 1:width + 1]


queries = np.ascontiguousarray(vecs(Q / "data/test_queries.fvecs", 750, DIM).view(np.float32))
truth = vecs(Q / "data/test_truth.ivecs", 750, 10)
results = dict(faiss=faiss.__version__, rows=[])
for variant in ("ivf", "ivf-hnsw"):
    args = types.SimpleNamespace(variant=variant, nlist=NLIST, sample=0, seed=42, int8_base=Q / "base.i8",
                                 int8_scales=Q / "scales.f32", int8_offsets=Q / "offsets.f32",
                                 centroids=E / "TieredQ-issue44-routerduel-20261110/results/64k/centroids.f32",
                                 assignments=E / "TieredQ-issue44-finelists-20261110/results/lloyd-64k/labels.u32")
    os.sched_setaffinity(0, range(os.cpu_count())); faiss.omp_set_num_threads(THREADS)
    index, _, build_s = build_reference(np.broadcast_to(np.float32(0), (ROWS, DIM)), args)
    os.sched_setaffinity(0, {SEARCH_CPU}); faiss.omp_set_num_threads(1)
    for budget in SINGLE[variant]:
        row = measure(index, variant, queries, truth, budget)
        results["rows"].append(dict(variant=variant, kind="single", build_s=build_s, **row))
    for budget in BATCH:
        for mode in (0, 1, 2, 3):
            index.parallel_mode = mode
            row = measure_batch(index, variant, queries, truth, budget, THREADS)
            results["rows"].append(dict(variant=variant, kind="batch", parallel_mode=mode, **row))
        index.parallel_mode = 0
    del index
Path(sys.argv[1]).write_text(json.dumps(results, indent=1) + "\n")
