"""Full-scale hull routing + two-stage k-means on DBpedia-1536 (305k).

Index: coarse k-means (L cells); each coarse cell split by its own k-means into m fine cells.
Vectors are laid out coarse-major, fine-minor, so coarse and fine cells are both contiguous slices.

Strategies (score per scan unit; units scanned in score order):
  A  flat IVF, coarse cells, centroid order                         routing L
  B  coarse cells ranked by sub-ball hull (min distance to their fine centroids) on the T nearest
     coarse centroids, fine-centroid distances in an r-dim PCA space; whole coarse cells scanned
                                                                    routing L + r + T·m·r/D
  C  two-stage: same routing as B, but *fine cells* are the scan units (fine-centroid order inside
     the shortlist; the rest follow in coarse-centroid order)       routing L + r + T·m·r/D
  D  flat IVF over all L·m fine cells, centroid order (no hierarchy) routing L·m
  oracle-coarse / oracle-fine: units ordered by their true nearest member
Recall is exact without caches: a true neighbour is found iff its unit has been scanned, so
recall(n) = share of ground-truth rows whose unit ranks < n. Reported: vectors scanned and
total = vectors + routing (distance-equivalents) to reach Recall@10 targets.

Usage: uv run hyperhull-twostage [n_coarse] [n_queries]   (default 1024 coarse cells, 500 queries)
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

from .evaluate import K
from .ivf import sq_dists
from .real_data import load_dbpedia

FINE_PER_COARSE = (16, 32)
SHORTLISTS = (64, 128, 256)
REDUCED_DIMS = (128, 256, None)  # None = full dimension
TARGETS = (0.90, 0.95, 0.99)
CHUNK = 64


@dataclass(frozen=True)
class TwoLevel:
    vectors: np.ndarray  # (N, D) coarse-major, fine-minor
    ids: np.ndarray  # (N,) original row id of each stored vector
    coarse_centroids: np.ndarray  # (L, D)
    coarse_offsets: np.ndarray  # (L + 1,)
    fine_centroids: np.ndarray  # (L·m, D) padded: empty fine cells repeat a sibling centroid
    fine_offsets: np.ndarray  # (L·m + 1,)
    m: int


def _kmeans(x: np.ndarray, k: int, niter: int = 20):
    import faiss

    km = faiss.Kmeans(x.shape[1], k, niter=niter, seed=1, min_points_per_centroid=1, max_points_per_centroid=100_000)
    km.train(x)
    return km.centroids, km.index.search(x, 1)[1][:, 0]


def build_two_level(base: np.ndarray, n_coarse: int, m: int, trainer: str = "faiss") -> TwoLevel:
    """trainer 'faiss': FAISS Lloyd k-means at both levels; 'tieredq': see build_two_level_tieredq."""
    if trainer == "tieredq":
        return build_two_level_tieredq(base, n_coarse, m)[0]
    coarse_c, coarse_a = _kmeans(base, n_coarse)
    order_parts, fine_c, fine_sizes, coarse_sizes = [], [], [], []
    for l in range(n_coarse):
        rows = np.flatnonzero(coarse_a == l)
        coarse_sizes.append(len(rows))
        k = min(m, len(rows))
        if k >= 2:
            sub_c, sub_a = _kmeans(base[rows], k, niter=15)
        else:
            sub_c = base[rows].mean(0, keepdims=True) if len(rows) else coarse_c[l : l + 1]
            sub_a = np.zeros(len(rows), dtype=np.int64)
        within = np.argsort(sub_a, kind="stable")
        order_parts.append(rows[within])
        counts = np.bincount(sub_a, minlength=len(sub_c))
        pad = np.resize(np.arange(len(sub_c)), m)
        fine_c.append(sub_c[pad])
        fine_sizes.append(np.concatenate([counts, np.zeros(m - len(sub_c), dtype=np.int64)]))
    order = np.concatenate(order_parts)
    fine_sizes = np.concatenate(fine_sizes)
    return TwoLevel(
        vectors=base[order],
        ids=order.astype(np.int64),
        coarse_centroids=coarse_c,
        coarse_offsets=np.concatenate([[0], np.cumsum(coarse_sizes)]),
        fine_centroids=np.concatenate(fine_c).astype(np.float32),
        fine_offsets=np.concatenate([[0], np.cumsum(fine_sizes)]),
        m=m,
    )


def build_two_level_tieredq(base: np.ndarray, n_coarse: int, m: int, seed: int = 42, n_threads: int = 0):
    """TieredQ e-means at both levels (native Partition). Returns (TwoLevel, build_stats)."""
    from hyperhull_native import Partition

    rows = np.ascontiguousarray(base, dtype=np.float32)
    partition = Partition(rows, coarse_lists=n_coarse, fine_per_coarse=m, seed=seed, n_threads=n_threads)
    order = partition.order
    index = TwoLevel(
        vectors=rows[order],
        ids=order,
        coarse_centroids=partition.coarse_centroids,
        coarse_offsets=partition.coarse_offsets,
        fine_centroids=partition.fine_centroids,
        fine_offsets=partition.fine_offsets,
        m=m,
    )
    return index, partition.build_stats


def ground_truth(index: TwoLevel, queries: np.ndarray):
    """Exact top-K rows, plus true nearest-member distance per coarse and per fine cell (oracles)."""
    gt, coarse_min, fine_min = [], [], []
    starts_c, starts_f = index.coarse_offsets[:-1], index.fine_offsets[:-1]
    nonempty_f = np.diff(index.fine_offsets) > 0
    for s in range(0, len(queries), CHUNK):
        d = sq_dists(queries[s : s + CHUNK], index.vectors)
        gt.append(np.argpartition(d, K, axis=1)[:, :K])
        coarse_min.append(np.minimum.reduceat(d, starts_c, axis=1))
        f = np.full((len(d), len(starts_f)), np.inf, dtype=np.float32)
        f[:, nonempty_f] = np.minimum.reduceat(d, starts_f[nonempty_f], axis=1)
        fine_min.append(f)
    return np.concatenate(gt), np.concatenate(coarse_min), np.concatenate(fine_min)


def cost_to_recall(scores: np.ndarray, gt_unit: np.ndarray, sizes: np.ndarray, routing: float) -> dict:
    """Vectors scanned and total work to reach each recall target when scanning units in score order."""
    n_q, n_units = scores.shape
    order = np.argsort(scores, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n_units)[None].repeat(n_q, 0), 1)
    gt_rank = np.take_along_axis(rank, gt_unit, 1)
    recall_curve = np.cumsum(np.bincount(gt_rank.ravel(), minlength=n_units)) / gt_unit.size
    vectors_curve = np.cumsum(sizes[order], axis=1).mean(0)
    out = {}
    for t in TARGETS:
        n = int(np.searchsorted(recall_curve, t - 1e-12))
        if n >= n_units:
            out[t] = None
            continue
        prev_r, prev_v = (recall_curve[n - 1], vectors_curve[n - 1]) if n else (0.0, 0.0)
        frac = (t - prev_r) / max(recall_curve[n] - prev_r, 1e-12)
        vectors = prev_v + frac * (vectors_curve[n] - prev_v)
        out[t] = (vectors, vectors + routing)
    return out


def _projector(centers: np.ndarray, r: int | None):
    if r is None:
        return lambda x: x
    mean = centers.mean(0)
    basis = np.linalg.svd(centers - mean, full_matrices=False)[2][:r]
    return lambda x: (x - mean) @ basis.T


def _finite_max(x: np.ndarray) -> float:
    return float(x[np.isfinite(x)].max())


def _print_row(name: str, routing: float, result: dict) -> None:
    cells = "".join(f"  {v[0]:8.0f} {v[1]:8.0f}" if (v := result[t]) else f"  {'-':>8} {'-':>8}" for t in TARGETS)
    print(f"  {name:<40}{routing:8.0f}{cells}", flush=True)


def evaluate(index: TwoLevel, queries: np.ndarray, gt, coarse_min, fine_min) -> None:
    n_coarse, m = len(index.coarse_centroids), index.m
    dim = index.vectors.shape[1]
    row_coarse = np.repeat(np.arange(n_coarse), np.diff(index.coarse_offsets))
    row_fine = np.repeat(np.arange(n_coarse * m), np.diff(index.fine_offsets))
    coarse_sizes, fine_sizes = np.diff(index.coarse_offsets), np.diff(index.fine_offsets)
    gt_coarse, gt_fine = row_coarse[gt], row_fine[gt]
    fine_owner = np.repeat(np.arange(n_coarse), m)

    print(f"\n### {n_coarse} coarse x {m} fine cells (fine size mean {fine_sizes.mean():.1f}, "
          f"empty {np.mean(fine_sizes == 0):.1%})")
    print(f"  {'strategy':<40}{'routing':>8}" + "".join(f"  {'R>=' + str(t) + ' vec / total':>17}" for t in TARGETS))
    centroid_d = sq_dists(queries, index.coarse_centroids)
    _print_row("oracle-coarse (perfect coarse hull)", 0, cost_to_recall(coarse_min, gt_coarse, coarse_sizes, 0))
    _print_row("oracle-fine (perfect fine hull)", 0, cost_to_recall(fine_min, gt_fine, fine_sizes, 0))
    _print_row("A flat IVF coarse, centroid", n_coarse, cost_to_recall(centroid_d, gt_coarse, coarse_sizes, n_coarse))
    fine_d_full = sq_dists(queries, index.fine_centroids)
    fine_d_full[:, fine_sizes == 0] = np.inf
    _print_row(f"D flat IVF {n_coarse * m} fine cells", n_coarse * m,
               cost_to_recall(fine_d_full, gt_fine, fine_sizes, n_coarse * m))

    for r in REDUCED_DIMS:
        project = _projector(index.fine_centroids, r)
        fine_d = sq_dists(project(queries), project(index.fine_centroids))
        fine_d[:, fine_sizes == 0] = np.inf
        sub_ball = fine_d.reshape(len(queries), n_coarse, m).min(2)
        r_eff = r or dim
        for t_short in SHORTLISTS:
            routing = n_coarse + (r or 0) + t_short * m * r_eff / dim
            shortlist = np.argpartition(centroid_d, t_short - 1, axis=1)[:, :t_short]
            in_short = np.zeros_like(centroid_d, dtype=bool)
            np.put_along_axis(in_short, shortlist, True, 1)
            # B: coarse cells, hull score inside the shortlist, centroid order after it
            b = np.where(in_short, sub_ball, centroid_d + _finite_max(sub_ball) + 1)
            label = f"pca{r}" if r else "full"
            _print_row(f"B coarse sub-ball {label} T={t_short}", routing, cost_to_recall(b, gt_coarse, coarse_sizes, routing))
            # C: fine cells, fine-centroid order inside the shortlist, then coarse order
            fine_in = in_short[:, fine_owner]
            big = _finite_max(fine_d) + 1
            after = (centroid_d + big)[:, fine_owner] + np.minimum(fine_d, big) / big
            c = np.where(fine_in, fine_d, after)
            _print_row(f"C two-stage fine {label} T={t_short}", routing, cost_to_recall(c, gt_fine, fine_sizes, routing))


def main() -> None:
    n_coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    t0 = time.perf_counter()
    d = load_dbpedia(n_queries=1000)
    base = np.asarray(d.base, dtype=np.float32)
    queries = d.queries[-n_queries:]  # held out from the 100k-subset tuning queries
    print(f"DBpedia-1536 full: N={len(base)}, Q={len(queries)}, loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    for m in FINE_PER_COARSE:
        t = time.perf_counter()
        index = build_two_level(base, n_coarse, m)
        build_s = time.perf_counter() - t
        gt, coarse_min, fine_min = ground_truth(index, queries)
        print(f"  built {n_coarse}x{m} in {build_s:.0f}s, ground truth in {time.perf_counter() - t - build_s:.0f}s", flush=True)
        evaluate(index, queries, gt, coarse_min, fine_min)


if __name__ == "__main__":
    main()
