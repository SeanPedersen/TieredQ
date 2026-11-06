"""Faiss reference construction and measurement, shared by calibration and held-out replay."""
import sys
import time

import faiss
import numpy as np

K = 10
TARGETS = (0.95, 0.99)
ADD_BATCH = 8192
HNSW_NEIGHBORS = 32
HNSW_CONSTRUCTION_EF = 100
QUANTIZER_SEARCH_EF = 128
LLOYD_ITERATIONS = 4
HNSW_SEARCH_BUDGETS = (10, 20, 40, 80, 160, 320, 640, 1280)


def build_reference(base: np.ndarray, args) -> tuple:
    """Include all training, assignment and insertion work in wall time."""
    start = time.perf_counter()
    dimension = base.shape[1]
    if args.variant == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, HNSW_NEIGHBORS)
        index.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
        budgets = HNSW_SEARCH_BUDGETS
    else:
        quantizer = faiss.IndexFlatL2(dimension) if args.variant == "ivf" else faiss.IndexHNSWFlat(dimension, HNSW_NEIGHBORS)
        if args.variant == "ivf-hnsw":
            quantizer.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
            quantizer.hnsw.efSearch = QUANTIZER_SEARCH_EF
        index = faiss.IndexIVFFlat(quantizer, dimension, args.nlist, faiss.METRIC_L2)
        index.cp.niter = LLOYD_ITERATIONS
        index.cp.seed = args.seed
        sample = np.random.default_rng(args.seed).choice(len(base), min(args.sample, len(base)), replace=False)
        print(f"Training {args.variant}: {len(sample)} vectors, {args.nlist} lists", file=sys.stderr, flush=True)
        index.train(np.ascontiguousarray(base[sample]))
        budgets = sorted({min(args.nlist, 2**power) for power in range(args.nlist.bit_length() + 1)})
    for offset in range(0, len(base), ADD_BATCH):
        index.add(base[offset : offset + ADD_BATCH])
        print(f"Added {min(offset + ADD_BATCH, len(base))}/{len(base)} ({time.perf_counter() - start:.1f}s)",
              file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - start
    print(f"Built {args.variant} in {elapsed:.3f}s", file=sys.stderr, flush=True)
    return index, budgets, elapsed


def measure(index, variant: str, queries: np.ndarray, truth: np.ndarray, budget: int) -> dict:
    """Warm each setting and count distances independently of Python timing overhead."""
    if variant == "hnsw":
        index.hnsw.efSearch = budget
    else:
        index.nprobe = budget
    for query in queries[:20]:
        index.search(query.reshape(1, -1), K)
    latency, hits, evaluations, maximum = [], 0, 0, 0
    for query, expected in zip(queries, truth, strict=True):
        stats = faiss.cvar.hnsw_stats if variant == "hnsw" else faiss.cvar.indexIVF_stats
        stats.reset()
        start = time.perf_counter_ns()
        _, found = index.search(query.reshape(1, -1), K)
        latency.append((time.perf_counter_ns() - start) / 1000)
        hits += len(set(found[0]) & set(expected))
        evaluations += stats.ndis
        maximum = max(maximum, stats.ndis)
    ordered = sorted(latency)
    p50, p99 = (ordered[max(0, int(np.ceil(p * len(ordered))) - 1)] for p in (0.5, 0.99))
    result = dict(budget=budget, recall10=hits / (K * len(queries)), p50_us=p50, p99_us=p99,
                  mean_distance_evaluations=evaluations / len(queries), max_distance_evaluations=maximum)
    print(f"{variant} budget={budget} recall={result['recall10']:.4f} p50_us={p50:.1f}", file=sys.stderr, flush=True)
    return result


def select_targets(observations: list[dict]) -> dict:
    """Select using calibration measurements only; a missing target remains explicit."""
    return {str(target): min((row for row in observations if row["recall10"] >= target),
                             key=lambda row: row["p50_us"], default=None) for target in TARGETS}
