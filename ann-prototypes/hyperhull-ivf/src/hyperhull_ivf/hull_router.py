"""Two-stage hull router: does a cluster hull cut the number of clusters a query must scan?

Hull of IVF cluster l = its own finer k-means partition: m sub-balls (centroid s_j, rms radius r_j).
    est_l(q) = min_j (||q - s_j|| - alpha · r_j)_+^2      ≈ distance to the cluster's nearest member
m = 1 is plain centroid routing; m → |cluster| approaches the exact nearest member (oracle).
Clusters are scanned in est order; we report how many clusters (and vectors) are needed to reach
Recall@10 targets. alpha is tuned on the first half of the queries, numbers are on the second half.

Usage: uv run hyperhull-route [n_lists] [n_queries]   (DBpedia-1536 100k; default 256 lists, 1000 queries)
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

from .bench import load_data
from .evaluate import QueryCache, build_cache, recall
from .ivf import IVFIndex, sq_dists
from .real_data import load_dbpedia_queries

SUB_BALLS = (2, 4, 8, 16, 32)
ALPHAS = np.round(np.arange(0.0, 1.51, 0.1), 2)
TARGETS = (0.90, 0.95, 0.99)
TUNE_TARGET = 0.95
QUERY_CHUNK = 64


@dataclass(frozen=True)
class SubBallHulls:
    centers: np.ndarray  # (L * m, D), padded by repetition for small clusters
    radii: np.ndarray  # (L * m,)
    m: int


def build_ivf_faiss(base: np.ndarray, n_lists: int) -> IVFIndex:
    import faiss

    km = faiss.Kmeans(base.shape[1], n_lists, niter=20, seed=1)
    km.train(base)
    assign = km.index.search(base, 1)[1][:, 0]
    order = np.argsort(assign, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=n_lists))])
    return IVFIndex(km.centroids, base[order], order, offsets)


def build_sub_hulls(index: IVFIndex, m: int) -> SubBallHulls:
    import faiss

    centers, radii = [], []
    for l in range(index.n_lists):
        members = index.vectors[index.list_slice(l)]
        k = min(m, len(members))
        if k > 1:
            km = faiss.Kmeans(members.shape[1], k, niter=15, seed=1, min_points_per_centroid=1)
            km.train(members)
            sub = km.centroids
        else:
            sub = members.mean(0, keepdims=True)
        assign = sq_dists(members, sub).argmin(1)
        r = np.array([np.sqrt(((members[assign == j] - sub[j]) ** 2).sum(1).mean()) if (assign == j).any() else 0.0
                      for j in range(len(sub))])
        pad = np.resize(np.arange(len(sub)), m)  # repeat sub-balls so every cluster has exactly m
        centers.append(sub[pad])
        radii.append(r[pad])
    return SubBallHulls(np.concatenate(centers).astype(np.float32), np.concatenate(radii).astype(np.float32), m)


def hull_distances(queries: np.ndarray, hulls: SubBallHulls, n_lists: int) -> tuple[np.ndarray, np.ndarray]:
    """Per (query, cluster, sub-ball): distance to sub-ball center, and the sub-ball radii."""
    d = np.concatenate([np.sqrt(sq_dists(queries[s : s + QUERY_CHUNK], hulls.centers))
                        for s in range(0, len(queries), QUERY_CHUNK)])
    return d.reshape(len(queries), n_lists, hulls.m), hulls.radii.reshape(n_lists, hulls.m)


def hull_scores(center_d: np.ndarray, radii: np.ndarray, alpha: float) -> np.ndarray:
    return (np.maximum(center_d - alpha * radii[None], 0.0) ** 2).min(2)


def clusters_to_recall(scores: np.ndarray, cache: QueryCache, sizes: np.ndarray) -> dict[float, tuple[float, float] | None]:
    """Scan clusters in score order; (clusters, vectors) needed to reach each recall target (interpolated)."""
    order = np.argsort(scores, axis=1)
    rows = np.arange(len(scores))[:, None]
    out: dict[float, tuple[float, float] | None] = {t: None for t in TARGETS}
    prev = (0.0, 0.0, 0.0)  # recall, clusters, vectors
    for n in range(1, scores.shape[1] + 1):
        lists = order[:, :n]
        cand_d = cache.list_topk_d[rows, lists].reshape(len(scores), -1)
        cand_i = cache.list_topk_i[rows, lists].reshape(len(scores), -1)
        found = np.take_along_axis(cand_i, np.argsort(cand_d, 1)[:, :10], 1)
        rec, vecs = recall(found, cache.gt), float(sizes[lists].sum(1).mean())
        for t in TARGETS:
            if out[t] is None and rec >= t:
                frac = (t - prev[0]) / max(rec - prev[0], 1e-12)
                out[t] = (prev[1] + frac * (n - prev[1]), prev[2] + frac * (vecs - prev[2]))
        if all(v is not None for v in out.values()):
            break
        prev = (rec, float(n), vecs)
    return out


def voronoi_scores(queries: np.ndarray, centroids: np.ndarray, n_bisectors: int = 1) -> np.ndarray:
    """Sound polytope bound on k-means (Voronoi) cells: every x in cell l is on l's side of the
    bisector with any other centroid c_j, so ||q - x|| >= (||q-c_l||^2 - ||q-c_j||^2) / (2 ||c_l - c_j||).
    Uses the query's `n_bisectors` nearest centroids as c_j and keeps the largest bound."""
    d2 = sq_dists(queries, centroids)
    nearest = np.argsort(d2, axis=1)[:, :n_bisectors]
    best = np.zeros_like(d2)
    for j in range(n_bisectors):
        cj = nearest[:, j]
        gap = np.linalg.norm(centroids[None] - centroids[cj][:, None], axis=2)
        bound = (d2 - d2[np.arange(len(d2)), cj][:, None]) / np.maximum(2 * gap, 1e-12)
        best = np.maximum(best, np.maximum(bound, 0.0))
    return best**2


def _subset(cache: QueryCache, rows) -> QueryCache:
    return QueryCache(cache.gt[rows], cache.list_topk_d[rows], cache.list_topk_i[rows])


def _row(name: str, routing: float, result) -> str:
    cells = "".join(f"  {c:6.1f} {v:7.0f}" if (r := result[t]) and (c := r[0], v := r[1]) else f"  {'-':>6} {'-':>7}"
                    for t in TARGETS)
    return f"  {name:<30}{routing:8.0f}{cells}"


def main() -> None:
    n_lists = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    t0 = time.perf_counter()
    base, _, _ = load_data("dbpedia100k")
    queries = load_dbpedia_queries(n_queries)
    index = build_ivf_faiss(base, n_lists)
    del base
    cache = build_cache(index, queries)
    sizes = index.sizes
    half = n_queries // 2
    train, test = np.arange(half), np.arange(half, n_queries)
    c_train, c_test = _subset(cache, train), _subset(cache, test)
    print(f"DBpedia-1536 100k: {n_lists} clusters (size {sizes.min()}-{sizes.max()}), "
          f"{len(test)} test queries, built in {time.perf_counter() - t0:.0f}s", flush=True)

    head = "".join(f"  {'R>=' + str(t):>14}" for t in TARGETS)
    print(f"\n  {'router':<30}{'routing':>8}{head}")
    print(f"  {'':<30}{'dists':>8}" + "  clusters vectors" * len(TARGETS))
    oracle = cache.list_topk_d[:, :, 0]
    print(_row("oracle (true nearest member)", float("nan"), clusters_to_recall(oracle[test], c_test, sizes)), flush=True)
    centroid = sq_dists(queries[test], index.centroids)
    print(_row("centroid (m=1)", n_lists, clusters_to_recall(centroid, c_test, sizes)), flush=True)
    for nb in (1, 4):
        voronoi = voronoi_scores(queries[test], index.centroids, nb)
        print(_row(f"voronoi polytope bound ({nb} bisect.)", n_lists * nb, clusters_to_recall(voronoi, c_test, sizes)), flush=True)

    for m in SUB_BALLS:
        t = time.perf_counter()
        hulls = build_sub_hulls(index, m)
        build_s = time.perf_counter() - t
        center_d, radii = hull_distances(queries, hulls, n_lists)
        tuned = {a: clusters_to_recall(hull_scores(center_d[train], radii, a), c_train, sizes)[TUNE_TARGET] for a in ALPHAS}
        alpha = min((a for a in ALPHAS if tuned[a]), key=lambda a: tuned[a][1])  # vectors scanned
        result = clusters_to_recall(hull_scores(center_d[test], radii, alpha), c_test, sizes)
        print(_row(f"sub-ball hull m={m} a={alpha:g}", n_lists * m, result) + f"   build {build_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
