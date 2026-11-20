"""FAISS IVF-PQ4 fast-scan + exact refine on full DBpedia-1536, under the combined router's cost model.

Same test queries (second half of the first n_queries) and recall definition (mean Recall@10).
Cost per query (distance-equivalents): nlist (routing) + codes scanned · code_bytes/(4·D) + k_factor·K exact.

Usage: uv run hyperhull-faiss-full [nlist] [n_queries]   (defaults 1024 1000)
"""

import sys
import time

import numpy as np

from .evaluate import K, recall
from .ivf import sq_dists
from .real_data import load_dbpedia

NPROBES = (4, 8, 16, 32, 64, 128, 256, 512)
K_FACTORS = (1, 2, 4, 8, 16, 32, 64)
PQ_SUBQUANTIZERS = 192  # 4-bit each → 96-byte codes
TARGETS = (0.90, 0.95, 0.99)


def exact_gt(base: np.ndarray, queries: np.ndarray, chunk: int = 64) -> np.ndarray:
    return np.concatenate([np.argpartition(sq_dists(queries[s : s + chunk], base), K, axis=1)[:, :K]
                           for s in range(0, len(queries), chunk)])


def main() -> None:
    import faiss

    nlist = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    d = load_dbpedia(n_queries=n_queries)
    base = np.asarray(d.base, dtype=np.float32)
    queries = d.queries[n_queries // 2 :]
    dim = base.shape[1]
    gt = exact_gt(base, queries)

    t = time.perf_counter()
    index = faiss.index_factory(dim, f"IVF{nlist},PQ{PQ_SUBQUANTIZERS}x4fs,Refine(Flat)")
    index.train(base)
    index.add(base)
    build_s = time.perf_counter() - t
    code_cost = (PQ_SUBQUANTIZERS * 4 / 8) / (4 * dim)
    print(f"FAISS IVF{nlist},PQ{PQ_SUBQUANTIZERS}x4fs,Refine(Flat): N={len(base)}, Q={len(queries)}, built in {build_s:.0f}s")

    space = faiss.ParameterSpace()
    refine = faiss.downcast_index(index)
    points = []
    for nprobe in NPROBES:
        space.set_index_parameter(index, "nprobe", nprobe)
        for k_factor in K_FACTORS:
            refine.k_factor = k_factor
            faiss.cvar.indexIVF_stats.reset()
            _, found = index.search(queries, K)
            codes = faiss.cvar.indexIVF_stats.ndis / len(queries)
            exact = k_factor * K
            points.append((recall(found, gt), nlist + codes * code_cost, exact, nprobe, k_factor, codes))

    print(f"  {'target':<8}{'exact':>8}{'routing+codes':>15}{'total':>8}   (nprobe, k_factor, codes scanned)")
    for target in TARGETS:
        ok = [p for p in points if p[0] >= target]
        if not ok:
            print(f"  R>={target:<5}  not reached")
            continue
        best = min(ok, key=lambda p: p[1] + p[2])
        print(f"  R>={target:<5}{best[2]:8.0f}{best[1]:15.0f}{best[1] + best[2]:8.0f}   ({best[3]}, {best[4]}, {best[5]:.0f})")


if __name__ == "__main__":
    main()
