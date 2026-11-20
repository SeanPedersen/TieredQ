"""Build-time comparison: FAISS Lloyd vs TieredQ e-means two-level partition, on full DBpedia-1536 (305k),
1024 coarse x 32 fine; both with the router's self-search + reverse-edge 16-NN graph.

One build per process (clean peak RSS). hartmut is shared with other sessions, so every stage reports process CPU
seconds (all threads; robust to a co-tenant taking cores) next to wall seconds, and runs go in ABBA order
(faiss, tieredq, tieredq, faiss) so both sides share a window. Prints the build stages, graph recall@k on a sampled set of rows
(brute force), and recall / µs per query (single thread, pinned) at the frontier configs of CLAUDE.md, with and
without graph expansion, so partitions are compared at equal search parameters.

Usage (from the hyperhull-ivf folder, on hartmut):
  uv run python experiments/native/build_compare.py faiss|tieredq [seed]
"""

import os
import resource
import sys
import time

import numpy as np

from hyperhull_ivf.evaluate import K, recall
from hyperhull_ivf.faiss_full import exact_gt
from hyperhull_ivf.ivf import sq_dists
from hyperhull_ivf.native_router import make_router, native_arrays
from hyperhull_ivf.real_data import load_dbpedia
from hyperhull_ivf.two_stage import build_two_level, build_two_level_tieredq

N_COARSE, FINE_PER_COARSE = 1024, 32
N_QUERIES = 1000  # test queries = second half, as in hyperhull-native-bench
PCA_DIMS, PREFIX_DIMS = 256, 128
THREADS = 32
BENCH_CORE = 20
REPEATS = 3
GRAPH_K = 16
SELF_SEARCH_GRAPH = dict(shortlist=64, fine_cells=128, rerank=60, refine=400)  # current router graph build
GRAPH_SAMPLE_ROWS = 2000
# frontier configs @R>=0.90 / 0.95 / 0.99 (CLAUDE.md, router + 16-NN graph)
CONFIGS = (
    dict(shortlist=16, fine_cells=50, rerank=20, refine=150, hops=1, expand_from=10, expand_exact=5),
    dict(shortlist=32, fine_cells=100, rerank=30, refine=200, hops=2, expand_from=20, expand_exact=10),
    dict(shortlist=256, fine_cells=1000, rerank=60, refine=400, hops=1, expand_from=30, expand_exact=20),
)


def graph_recall(graph: np.ndarray, vectors: np.ndarray, rng: np.random.Generator) -> float:
    """Share of each sampled row's true k nearest other rows (brute force, stored-row space) found in its graph row."""
    rows = rng.choice(len(vectors), size=GRAPH_SAMPLE_ROWS, replace=False)
    hits = 0
    for start in range(0, len(rows), 256):
        block = rows[start : start + 256]
        d = sq_dists(vectors[block], vectors)
        d[np.arange(len(block)), block] = np.inf
        truth = np.argpartition(d, GRAPH_K, axis=1)[:, :GRAPH_K]
        hits += (graph[block][:, :, None] == truth[:, None, :]).any(-1).sum()
    return hits / (len(rows) * GRAPH_K)


class Stage:
    """Wall and process-CPU seconds of a with-block, appended to a stage dict as name=(wall, cpu)."""

    def __init__(self, stages: dict, name: str):
        self.stages, self.name = stages, name

    def __enter__(self):
        self.wall, self.cpu = time.perf_counter(), time.process_time()

    def __exit__(self, *_):
        self.stages[self.name] = (time.perf_counter() - self.wall, time.process_time() - self.cpu)


def search_point(router, queries, gt, params: dict) -> tuple[float, float]:
    """Recall and CPU µs per query (single thread: CPU time excludes time the core was taken by others)."""
    best = np.inf
    for _ in range(REPEATS):
        t = time.process_time()
        _, ids, _ = router.search(queries, K, n_threads=1, expand_k=GRAPH_K, **params)
        best = min(best, time.process_time() - t)
    return recall(ids, gt), best / len(queries) * 1e6


def main() -> None:
    trainer = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    d = load_dbpedia(n_queries=N_QUERIES)
    base = np.asarray(d.base, dtype=np.float32)
    queries = np.ascontiguousarray(d.queries[N_QUERIES // 2 :], dtype=np.float32)
    gt = exact_gt(base, queries)
    print(f"{trainer} seed={seed}: DBpedia-1536 N={len(base)}, {len(queries)} test queries", flush=True)

    stages, details = {}, {}
    with Stage(stages, "partition"):
        if trainer == "faiss":
            import faiss

            faiss.omp_set_num_threads(THREADS)
            index = build_two_level(base, N_COARSE, FINE_PER_COARSE)
        else:
            index, partition_stats = build_two_level_tieredq(base, N_COARSE, FINE_PER_COARSE, seed=seed,
                                                             n_threads=THREADS)
            details.update(partition_stats)
    with Stage(stages, "arrays"):
        arrays = native_arrays(index, pca_dims=PCA_DIMS, timings=details)
    with Stage(stages, "router"):
        router = make_router(arrays, prefix_dims=PREFIX_DIMS)
    stages["total_no_graph"] = tuple(sum(v[i] for v in stages.values()) for i in range(2))
    print("  build (wall s / cpu s) " + "  ".join(f"{k}={w:.2f}/{c:.1f}" for k, (w, c) in stages.items()), flush=True)
    print("  stage detail (wall s) " + "  ".join(f"{k}={v:.2f}" for k, v in details.items()), flush=True)

    rng = np.random.default_rng(0)
    graph_time, graphs = {}, {}
    with Stage(graph_time, "self-search+reverse"):
        router.build_graph(GRAPH_K, **SELF_SEARCH_GRAPH, reverse_edges=True, n_threads=THREADS)
    graphs["self-search+reverse"] = router.graph()

    all_cores = os.sched_getaffinity(0)
    for name, graph in graphs.items():
        router.set_graph(np.ascontiguousarray(graph, dtype=np.uint32))
        wall, cpu = graph_time[name]
        print(f"  graph {name}: {wall:.2f} s wall / {cpu:.1f} s cpu, "
              f"recall@{GRAPH_K} {graph_recall(graph, index.vectors, rng):.3f}", flush=True)
        os.sched_setaffinity(0, {BENCH_CORE})
        for target, params in zip((0.90, 0.95, 0.99), CONFIGS):
            plain = search_point(router, queries, gt, {**params, "hops": 0})
            expanded = search_point(router, queries, gt, params)
            print(f"    config@{target}: plain R={plain[0]:.4f} {plain[1]:6.1f} cpu-us | "
                  f"+graph R={expanded[0]:.4f} {expanded[1]:6.1f} cpu-us", flush=True)
        os.sched_setaffinity(0, all_cores)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  peak RSS {peak_gb:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
