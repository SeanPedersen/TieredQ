"""Hull-native partitionings vs k-means: which cells let a hull cut the clusters scanned?

Partitions (same number of cells):
  * kmeans    – FAISS k-means (baseline)
  * pca-tree  – balanced hyperplane tree: split each node at the median of its top principal
                direction. A leaf is an exact convex polytope (intersection of its path half-spaces),
                so its hull is free: for each path split (unit u, threshold t, side s) a query on
                the wrong side is at least v = max(0, -s·(u·q - t)) away from every leaf point.
  * kcenter   – greedy farthest-point centers (minimax radius), nearest-center assignment.

Routers per partition (score ≈ distance to the cell's nearest member, lower = scan first):
  centroid · sub-ball hull (m finer k-means balls per cell, alpha = 0) · oracle
  approximate hulls from approx_hulls.py (parameter tuned on train queries: vectors scanned for R >= 0.95)
  pca-tree only: polytope sound (max_i v_i) and polytope approx (sqrt(sum_i v_i^2))
Metric: vectors scanned (fair across unequal cell sizes; clusters also shown) to reach Recall@10, on test queries.
Total work = vectors scanned + routing distances.

Usage: uv run hyperhull-cluster [n_cells] [n_queries]   (DBpedia-1536 100k; default 256, 1000)
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from .bench import load_data
from .approx_hulls import all_families
from .evaluate import QueryCache, build_cache
from .hull_router import (
    TARGETS,
    _row,
    _subset,
    build_ivf_faiss,
    build_sub_hulls,
    clusters_to_recall,
    hull_distances,
    hull_scores,
)
from .ivf import IVFIndex, sq_dists
from .real_data import load_dbpedia_queries

SUB_BALLS = (8, 32)
REDUCED_DIMS = (32, 64, 128, 256)  # sub-ball distances in a PCA space of the sub-centroids
SHORTLISTS = (64, 128)  # two-stage: sub-ball hull only on the T nearest centroids
SHORTLIST_M = 32
TUNE_TARGET = 0.95
PCA_SAMPLE = 4096  # points used to estimate each node's split direction


@dataclass(frozen=True)
class TreeHull:
    directions: np.ndarray  # (n_internal, D) unit split directions, heap order
    thresholds: np.ndarray  # (n_internal,)
    leaf_paths: np.ndarray  # (L, depth) internal-node ids on each leaf's path
    leaf_sides: np.ndarray  # (L, depth) +1 = leaf lies on the u·x >= t side, -1 otherwise


def _group(base: np.ndarray, assign: np.ndarray, centers: np.ndarray | None, n_cells: int) -> IVFIndex:
    order = np.argsort(assign, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=n_cells))])
    vectors = base[order]
    if centers is None:
        centers = np.stack([vectors[offsets[l] : offsets[l + 1]].mean(0) for l in range(n_cells)])
    return IVFIndex(centers.astype(np.float32), vectors, order, offsets)


def pca_tree_partition(base: np.ndarray, depth: int, seed: int = 0) -> tuple[IVFIndex, TreeHull]:
    rng = np.random.default_rng(seed)
    n_internal = 2**depth - 1
    directions = np.zeros((n_internal, base.shape[1]), dtype=np.float32)
    thresholds = np.zeros(n_internal, dtype=np.float32)
    node_rows = {0: np.arange(len(base))}
    assign = np.empty(len(base), dtype=np.int64)
    for node in range(n_internal):
        rows = node_rows.pop(node)
        sample = base[rows if len(rows) <= PCA_SAMPLE else rng.choice(rows, PCA_SAMPLE, replace=False)]
        u = np.linalg.svd(sample - sample.mean(0), full_matrices=False)[2][0]
        proj = base[rows] @ u
        t = np.median(proj)
        directions[node], thresholds[node] = u, t
        left, right = rows[proj < t], rows[proj >= t]
        for child, child_rows in ((2 * node + 1, left), (2 * node + 2, right)):
            if child < n_internal:
                node_rows[child] = child_rows
            else:
                assign[child_rows] = child - n_internal
    n_leaves = 2**depth
    paths = np.zeros((n_leaves, depth), dtype=np.int64)
    sides = np.zeros((n_leaves, depth), dtype=np.float32)
    for leaf in range(n_leaves):
        node = leaf + n_internal
        for level in range(depth - 1, -1, -1):
            parent = (node - 1) // 2
            paths[leaf, level], sides[leaf, level] = parent, (1.0 if node == 2 * parent + 2 else -1.0)
            node = parent
    return _group(base, assign, None, n_leaves), TreeHull(directions, thresholds, paths, sides)


def tree_violations(queries: np.ndarray, tree: TreeHull) -> np.ndarray:
    """(Q, L, depth) distance of q to each wrong-side half-space on each leaf's path."""
    margins = queries @ tree.directions.T - tree.thresholds[None]  # (Q, n_internal)
    return np.maximum(-tree.leaf_sides[None] * margins[:, tree.leaf_paths], 0.0)


def kcenter_partition(base: np.ndarray, k: int, seed: int = 0) -> IVFIndex:
    rng = np.random.default_rng(seed)
    centers = [rng.integers(len(base))]
    closest = sq_dists(base, base[centers[0]][None])[:, 0]
    for _ in range(1, k):
        centers.append(int(closest.argmax()))
        closest = np.minimum(closest, sq_dists(base, base[centers[-1]][None])[:, 0])
    c = base[centers]
    assign = np.concatenate([sq_dists(base[s : s + 16384], c).argmin(1) for s in range(0, len(base), 16384)])
    return _group(base, assign, c, k)


def reduced_center_distances(queries: np.ndarray, hulls, n_cells: int, n_components: int) -> np.ndarray:
    """Sub-ball center distances in an r-dim PCA space fitted on the sub-centroids (query projected once)."""
    mean = hulls.centers.mean(0)
    basis = np.linalg.svd(hulls.centers - mean, full_matrices=False)[2][:n_components]
    zc, zq = (hulls.centers - mean) @ basis.T, (queries - mean) @ basis.T
    return np.sqrt(sq_dists(zq, zc)).reshape(len(queries), n_cells, hulls.m)


def shortlist_scores(centroid_d: np.ndarray, hull: np.ndarray, shortlist: int) -> np.ndarray:
    """Hull scores for each query's `shortlist` nearest centroids; all others ranked after, by centroid."""
    top = np.argpartition(centroid_d, shortlist - 1, axis=1)[:, :shortlist]
    out = centroid_d + hull.max() + 1.0
    np.put_along_axis(out, top, np.take_along_axis(hull, top, 1), 1)
    return out


def evaluate_partition(name: str, index: IVFIndex, queries, cache: QueryCache, train, test, tree: TreeHull | None) -> None:
    sizes, n_cells = index.sizes, index.n_lists
    c_train, c_test = _subset(cache, train), _subset(cache, test)
    print(f"\n### {name}: {n_cells} cells (size {sizes.min()}-{sizes.max()}, std {sizes.std():.0f})", flush=True)
    print(f"  {'router':<30}{'routing':>8}" + "".join(f"  {'R>=' + str(t):>14}" for t in TARGETS))
    print(_row("oracle (true nearest member)", float("nan"), clusters_to_recall(cache.list_topk_d[test, :, 0], c_test, sizes)), flush=True)
    print(_row("centroid", n_cells, clusters_to_recall(sq_dists(queries[test], index.centroids), c_test, sizes)), flush=True)
    if tree is not None:
        v = tree_violations(queries[test], tree)
        n_internal = len(tree.thresholds)
        print(_row("polytope sound (max margin)", n_internal, clusters_to_recall(v.max(2) ** 2, c_test, sizes)), flush=True)
        print(_row("polytope approx (sum margin^2)", n_internal, clusters_to_recall((v**2).sum(2), c_test, sizes)), flush=True)
    for m in SUB_BALLS:
        t = time.perf_counter()
        hulls = build_sub_hulls(index, m)
        build_s = time.perf_counter() - t
        center_d, radii = hull_distances(queries[test], hulls, n_cells)
        result = clusters_to_recall(hull_scores(center_d, radii, 0.0), c_test, sizes)
        print(_row(f"sub-ball hull m={m}", n_cells * m, result) + f"   build {build_s:.1f}s", flush=True)
        if m == SHORTLIST_M:
            centroid_d = sq_dists(queries[test], index.centroids)
            hull = hull_scores(center_d, radii, 0.0)
            for t_short in SHORTLISTS:
                print(_row(f"  shortlist T={t_short} + m={m}", n_cells + t_short * m,
                           clusters_to_recall(shortlist_scores(centroid_d, hull, t_short), c_test, sizes)), flush=True)
            dim = index.vectors.shape[1]
            for r in REDUCED_DIMS:
                reduced = hull_scores(reduced_center_distances(queries[test], hulls, n_cells, r), radii, 0.0)
                for t_short in SHORTLISTS:
                    cost = n_cells + r + t_short * m * r / dim
                    print(_row(f"  pca{r} T={t_short} + m={m}", cost,
                               clusters_to_recall(shortlist_scores(centroid_d, reduced, t_short), c_test, sizes)), flush=True)
    if os.environ.get("HULL_SKIP_APPROX"):
        return
    t = time.perf_counter()
    families = all_families(index)
    print(f"  (approx hull stats built in {time.perf_counter() - t:.1f}s)", flush=True)
    for fam in families:
        train_scores = {p: fam.scores(queries[train], p) for p in fam.grid}
        tuned = {p: clusters_to_recall(sc, c_train, sizes)[TUNE_TARGET] for p, sc in train_scores.items()}
        best = min((p for p in fam.grid if tuned[p]), key=lambda p: tuned[p][1])  # vectors scanned, not clusters
        result = clusters_to_recall(fam.scores(queries[test], best), c_test, sizes)
        print(_row(f"{fam.name} [{best:g}]", fam.cost, result), flush=True)


def main() -> None:
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    depth = int(np.log2(n_cells))
    assert 2**depth == n_cells, "n_cells must be a power of two (pca-tree depth)"
    base, _, _ = load_data("dbpedia100k")
    queries = load_dbpedia_queries(n_queries)
    train, test = np.arange(n_queries // 2), np.arange(n_queries // 2, n_queries)
    print(f"DBpedia-1536 100k, {len(test)} test queries; metric: clusters / vectors scanned to reach recall", flush=True)

    builders = {
        "kmeans": lambda: (build_ivf_faiss(base, n_cells), None),
        "pca-tree": lambda: pca_tree_partition(base, depth),
        "kcenter": lambda: (kcenter_partition(base, n_cells), None),
    }
    only = sys.argv[3:] or list(builders)
    for name in only:
        t = time.perf_counter()
        index, tree = builders[name]()
        build_s = time.perf_counter() - t
        cache = build_cache(index, queries)
        evaluate_partition(f"{name} (built in {build_s:.0f}s)", index, queries, cache, train, test, tree)


if __name__ == "__main__":
    main()
