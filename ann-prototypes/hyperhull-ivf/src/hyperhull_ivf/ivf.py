"""Plain IVF index: k-means coarse quantizer + contiguous inverted lists.

Vectors are reordered so every list is one contiguous slice of `vectors`
(cache-friendly scans); `offsets[l]:offsets[l+1]` addresses list l.
Input may be a (float16) memory map: k-means trains on a sample and the
full set is assigned in chunks, so only one float32 copy is ever materialized.
"""

from dataclasses import dataclass

import numpy as np

ASSIGN_CHUNK = 32_768


@dataclass(frozen=True)
class IVFIndex:
    centroids: np.ndarray  # (L, D)
    vectors: np.ndarray  # (N, D) float32, grouped by list
    ids: np.ndarray  # (N,), original id of each row in `vectors`
    offsets: np.ndarray  # (L + 1,)

    @property
    def n_lists(self) -> int:
        return len(self.centroids)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def list_slice(self, list_id: int) -> slice:
        return slice(self.offsets[list_id], self.offsets[list_id + 1])


def sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances, (len(a), len(b))."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def _kmeans_pp_init(x, k, rng):
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(len(x))]
    closest = sq_dists(x, centers[:1])[:, 0]
    for c in range(1, k):
        centers[c] = x[rng.choice(len(x), p=closest / closest.sum())]
        closest = np.minimum(closest, sq_dists(x, centers[c : c + 1])[:, 0])
    return centers


def _cluster_means(x, assign, k, fallback):
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nonempty = counts > 0
    means = fallback.copy()
    sums = np.add.reduceat(x[order], starts[nonempty], axis=0)
    means[nonempty] = sums / counts[nonempty, None]
    return means


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(x, k, rng)
    for _ in range(iters):
        dist = sq_dists(x, centers)
        # Empty clusters keep their old position except one gets re-seeded at the worst-fit point.
        fallback = centers.copy()
        fallback[np.bincount(dist.argmin(1), minlength=k).argmin()] = x[dist.min(1).argmax()]
        centers = _cluster_means(x, dist.argmin(1), k, fallback).astype(np.float32)
    return centers


def assign_chunked(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), ASSIGN_CHUNK):
        block = np.asarray(x[start : start + ASSIGN_CHUNK], dtype=np.float32)
        out[start : start + len(block)] = sq_dists(block, centroids).argmin(1)
    return out


def build_ivf(x: np.ndarray, n_lists: int, train_size: int, seed: int = 0) -> IVFIndex:
    rng = np.random.default_rng(seed)
    train_rows = np.sort(rng.choice(len(x), size=min(train_size, len(x)), replace=False))
    centroids = kmeans(np.asarray(x[train_rows], dtype=np.float32), n_lists, seed=seed)
    assign = assign_chunked(x, centroids)
    order = np.argsort(assign, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=n_lists))])
    vectors = np.empty((len(x), x.shape[1]), dtype=np.float32)
    for start in range(0, len(x), ASSIGN_CHUNK):
        rows = order[start : start + ASSIGN_CHUNK]
        vectors[start : start + len(rows)] = x[rows]
    return IVFIndex(centroids=centroids, vectors=vectors, ids=order, offsets=offsets)
