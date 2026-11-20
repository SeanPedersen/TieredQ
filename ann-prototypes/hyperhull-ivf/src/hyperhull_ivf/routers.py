"""List-ranking strategies. Each router scores (query, list) pairs; lower = probe first.

Every router also reports its routing cost in *distance-equivalents per query*
(one D-dim dot product / distance = 1), so it can be added to the scan cost.

Scores are heuristic estimates of "squared distance from q to the nearest
member of the list" — not safe lower bounds (except `sphere`).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ivf import IVFIndex, sq_dists
from .summaries import ListSummaries


@dataclass(frozen=True)
class Router:
    name: str
    score: Callable[[np.ndarray], np.ndarray]  # (Q, D) -> (Q, L)
    cost_per_query: float


def centroid_router(index: IVFIndex) -> Router:
    return Router("ivf-centroid", lambda q: sq_dists(q, index.centroids), index.n_lists)


def sphere_router(index: IVFIndex, s: ListSummaries) -> Router:
    """Conservative bound: distance to the list's bounding sphere."""

    def score(q):
        gap = np.sqrt(sq_dists(q, index.centroids)) - s.radius[None]
        return np.maximum(gap, 0.0) ** 2

    return Router("sphere-bound", score, index.n_lists)


def hull_points_router(index: IVFIndex, s: ListSummaries, n_points: int) -> Router:
    """Min distance to centroid or any of the first `n_points` sampled extreme points."""
    hull = np.ascontiguousarray(s.hull_points[:, :n_points])
    n_lists, m, dim = hull.shape
    flat = hull.reshape(-1, dim)

    def score(q):
        to_hull = sq_dists(q, flat).reshape(len(q), n_lists, m).min(2)
        return np.minimum(to_hull, sq_dists(q, index.centroids))

    return Router(f"hull-points-{n_points}", score, n_lists * (n_points + 1))


def _box_estimate(z, centroid_sq, pmin, pmax, orth_floor):
    """Estimated NN distance from the list's principal-subspace box.

    ||q - x||^2 = ||z_q - z_x||^2 + ||o_q - o_x||^2 (in-subspace + orthogonal parts).
    In-subspace: distance from z_q to the [min, max] box. Orthogonal: assume
    o_q ⟂ o_x (high-dim noise), giving ||o_q||^2 + a low quantile of ||o_x||^2.
    """
    box_gap = np.maximum(z - pmax, 0.0) + np.maximum(pmin - z, 0.0)
    orth_sq = np.maximum(centroid_sq - (z**2).sum(-1), 0.0)
    return (box_gap**2).sum(-1) + orth_sq + orth_floor


def pca_box_router(index: IVFIndex, s: ListSummaries, n_components: int) -> Router:
    n_lists, _, dim = s.pca_basis.shape
    basis = np.ascontiguousarray(s.pca_basis[:, :n_components]).reshape(-1, dim)
    c_proj = s.centroid_proj[:, :n_components]
    pmin, pmax = s.proj_min[:, :n_components], s.proj_max[:, :n_components]

    def score(q):
        z = (q @ basis.T).reshape(len(q), n_lists, n_components) - c_proj[None]
        return _box_estimate(z, sq_dists(q, index.centroids), pmin[None], pmax[None], s.orth_sq_floor[None])

    return Router(f"pca-box-{n_components}", score, n_lists * (n_components + 1))


def pca_rerank_router(index: IVFIndex, s: ListSummaries, n_components: int, shortlist: int) -> Router:
    """Two-stage: centroid distances pick a shortlist, PCA box re-scores only those.
    Lists outside the shortlist are ranked after it (by centroid distance)."""

    def score(q):
        centroid_d = sq_dists(q, index.centroids)
        top = np.argpartition(centroid_d, shortlist - 1, axis=1)[:, :shortlist]
        z = np.einsum("qd,qtrd->qtr", q, s.pca_basis[top, :n_components]) - s.centroid_proj[top, :n_components]
        refined = _box_estimate(
            z,
            np.take_along_axis(centroid_d, top, 1),
            s.proj_min[top, :n_components],
            s.proj_max[top, :n_components],
            s.orth_sq_floor[top],
        )
        out = centroid_d + refined.max() + 1.0  # strictly behind the shortlist
        np.put_along_axis(out, top, refined, 1)
        return out

    cost = index.n_lists + shortlist * n_components
    return Router(f"pca{n_components}-rerank{shortlist}", score, cost)


def all_routers(index: IVFIndex, s: ListSummaries, shortlists: tuple[int, ...]) -> list[Router]:
    return [
        centroid_router(index),
        sphere_router(index, s),
        hull_points_router(index, s, 8),
        hull_points_router(index, s, 16),
        pca_box_router(index, s, 4),
        pca_box_router(index, s, 8),
        *(pca_rerank_router(index, s, 8, shortlist=t) for t in shortlists),
    ]
