"""Wall-time benchmark: C++ combined hull router (± k-NN graph expansion) vs FAISS IVF-PQ4 fast-scan + refine,
full DBpedia-1536.

Both indexes are swept over their search parameters on the same test queries (second half of the
first n_queries) and timed single-threaded on the same pinned core (best of REPEATS runs over the whole
batch). The router uses the two-pass sketch (PREFIX_DIMS streamed, full sketch for `refine`). Reported per
recall target: the fastest configuration reaching it, plus build time and index memory beyond the raw
float vectors. A multi-threaded throughput line uses the chosen configs with THREADS threads. "router+graph"
adds a 16-NN graph (router self-search + reverse edges, built with THREADS threads; its build time and
memory are reported separately) and sweeps the expansion parameters.

Usage: uv run hyperhull-native-bench [n_coarse] [m] [n_queries]   (defaults 1024 32 1000)
"""

import itertools
import os
import sys
import time

import numpy as np

from .evaluate import K, recall
from .faiss_full import PQ_SUBQUANTIZERS, exact_gt
from .native_router import extra_bytes, make_router, native_arrays
from .real_data import load_dbpedia
from .two_stage import build_two_level

ROUTER_GRID = dict(shortlist=(32, 48, 64, 80, 96, 192, 256), fine_cells=(96, 128, 192, 256, 320, 1280, 1536, 1792, 2048),
                   rerank=(30, 40, 60, 100, 120, 160), refine=(200, 300, 600, 800))
GRAPH_K = 16
GRAPH_BUILD = dict(shortlist=64, fine_cells=128, rerank=60, refine=400)  # self-search params of the graph build
GRAPH_SCAN = ((16, 50), (32, 50), (32, 100), (64, 200), (128, 400), (256, 800), (256, 1000), (256, 1200))
GRAPH_GRID = dict(rerank_refine=((20, 150), (30, 200), (60, 400), (80, 600), (100, 800)), hops=(1, 2),
                  expand_from=(10, 20, 30), expand_exact=(5, 10, 20))
PCA_DIMS = 256     # sketch dims (4-bit); the first PREFIX_DIMS are streamed in pass 1
PREFIX_DIMS = 128
BENCH_CORE = 20  # single-thread sweeps of both methods run pinned to this core
FAISS_GRID = dict(nprobe=(8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256), k_factor=(2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))
TARGETS = (0.90, 0.95, 0.99)
REPEATS = 2
THREADS = 32


def _timed(fn) -> tuple[float, object]:
    best, out = np.inf, None
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def sweep_router(router, queries, gt) -> list[dict]:
    points = []
    for shortlist, fine_cells, rerank, refine in itertools.product(*ROUTER_GRID.values()):
        params = dict(shortlist=shortlist, fine_cells=fine_cells, rerank=rerank, refine=refine)
        secs, (_, ids, stats) = _timed(lambda: router.search(queries, K, n_threads=1, **params))
        points.append(dict(params=params, recall=recall(ids, gt), ms=secs / len(queries) * 1e3,
                           sketched=float(stats[:, 0].mean())))
    return points


def sweep_graph(router, queries, gt) -> list[dict]:
    points = []
    for (shortlist, fine_cells), (rerank, refine), hops, expand_from, expand_exact in itertools.product(
            GRAPH_SCAN, *GRAPH_GRID.values()):
        params = dict(shortlist=shortlist, fine_cells=fine_cells, rerank=rerank, refine=refine, hops=hops,
                      expand_from=expand_from, expand_k=GRAPH_K, expand_exact=expand_exact)
        secs, (_, ids, stats) = _timed(lambda: router.search(queries, K, n_threads=1, **params))
        points.append(dict(params=params, recall=recall(ids, gt), ms=secs / len(queries) * 1e3,
                           sketched=float(stats[:, 0].mean())))
    return points


def build_faiss(base: np.ndarray, nlist: int):
    import faiss

    index = faiss.index_factory(base.shape[1], f"IVF{nlist},PQ{PQ_SUBQUANTIZERS}x4fs,Refine(Flat)")
    index.train(base)
    index.add(base)
    return index


def sweep_faiss(index, queries, gt) -> list[dict]:
    import faiss

    faiss.omp_set_num_threads(1)
    refine = faiss.downcast_index(index)
    space = faiss.ParameterSpace()
    points = []
    for nprobe, k_factor in itertools.product(*FAISS_GRID.values()):
        space.set_index_parameter(index, "nprobe", nprobe)
        refine.k_factor = k_factor
        secs, (_, ids) = _timed(lambda: index.search(queries, K))
        points.append(dict(params=dict(nprobe=nprobe, k_factor=k_factor), recall=recall(ids, gt),
                           ms=secs / len(queries) * 1e3))
    return points


def fastest(points: list[dict], target: float) -> dict | None:
    ok = [p for p in points if p["recall"] >= target]
    return min(ok, key=lambda p: p["ms"]) if ok else None


def main() -> None:
    import faiss

    n_coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    d = load_dbpedia(n_queries=n_queries)
    base = np.asarray(d.base, dtype=np.float32)
    queries = np.ascontiguousarray(d.queries[n_queries // 2 :], dtype=np.float32)
    gt = exact_gt(base, queries)
    from hyperhull_native import SIMD_LEVEL

    print(f"DBpedia-1536 N={len(base)}, Q={len(queries)} test queries, single-thread timings (best of {REPEATS}), "
          f"hull router kernels: {SIMD_LEVEL}", flush=True)

    faiss.omp_set_num_threads(THREADS)
    t = time.perf_counter()
    index = build_two_level(base, n_coarse, m)
    arrays = native_arrays(index, pca_dims=PCA_DIMS)
    router = make_router(arrays, prefix_dims=PREFIX_DIMS)
    router_build = time.perf_counter() - t
    t = time.perf_counter()
    router.build_graph(GRAPH_K, **GRAPH_BUILD, reverse_edges=True, n_threads=THREADS)
    graph_build = time.perf_counter() - t
    graph_mb = len(base) * GRAPH_K * 4 / 2**20
    t = time.perf_counter()
    faiss_index = build_faiss(base, n_coarse)
    faiss_build = time.perf_counter() - t
    faiss_extra = len(base) * PQ_SUBQUANTIZERS // 2 + len(base) * 8 + n_coarse * base.shape[1] * 4
    print(f"  hull router {n_coarse}x{m}: build {router_build:.0f}s, extra memory {extra_bytes(arrays) / 2**20:.0f} MB; "
          f"+ {GRAPH_K}-NN graph: +{graph_build:.1f}s, +{graph_mb:.0f} MB")
    print(f"  FAISS IVF{n_coarse},PQ{PQ_SUBQUANTIZERS}x4fs,Refine(Flat): build {faiss_build:.0f}s, "
          f"extra memory ~{faiss_extra / 2**20:.0f} MB (codes + ids + centroids)", flush=True)

    all_cores = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {BENCH_CORE})
    router_points = sweep_router(router, queries, gt)
    graph_points = sweep_graph(router, queries, gt)  # plain sweep: hops=0 ignores the graph
    faiss_points = sweep_faiss(faiss_index, queries, gt)
    os.sched_setaffinity(0, all_cores)

    print(f"\n  {'target':<8}{'router ms':>10}{'+graph ms':>10}{'FAISS ms':>10}{'speedup':>9}   (vs FAISS, best router variant)")
    for target in TARGETS:
        r, g, f = fastest(router_points, target), fastest(graph_points, target), fastest(faiss_points, target)
        ms = lambda p: f"{p['ms']:.3f}" if p else "-"
        best = min((p for p in (r, g) if p), key=lambda p: p["ms"], default=None)
        speedup = f"{f['ms'] / best['ms']:.2f}x" if best and f else "-"
        print(f"  R>={target:<5}{ms(r):>10}{ms(g):>10}{ms(f):>10}{speedup:>9}", flush=True)
        for name, p in (("router", r), ("+graph", g), ("FAISS", f)):
            if p:
                print(f"           {name:7s} {p['params']} R={p['recall']:.3f}", flush=True)

    print(f"\n  throughput with {THREADS} threads (queries/s) at the single-thread-optimal configs")
    faiss.omp_set_num_threads(THREADS)
    space, refine = faiss.ParameterSpace(), faiss.downcast_index(faiss_index)
    for target in TARGETS:
        r, g, f = fastest(router_points, target), fastest(graph_points, target), fastest(faiss_points, target)
        if not (r and g and f):
            continue
        r_secs, _ = _timed(lambda: router.search(queries, K, n_threads=THREADS, **r["params"]))
        g_secs, _ = _timed(lambda: router.search(queries, K, n_threads=THREADS, **g["params"]))
        space.set_index_parameter(faiss_index, "nprobe", f["params"]["nprobe"])
        refine.k_factor = f["params"]["k_factor"]
        f_secs, _ = _timed(lambda: faiss_index.search(queries, K))
        print(f"  R>={target:<5} router {len(queries) / r_secs:9.0f}   +graph {len(queries) / g_secs:9.0f}   "
              f"FAISS {len(queries) / f_secs:9.0f}", flush=True)


if __name__ == "__main__":
    main()
