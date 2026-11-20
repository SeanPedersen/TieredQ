"""Two-level outer-hull search: exact (eps = 0) or guaranteed (1 + eps)-approximate.

Per query:
  1. centroid distances for all lists (routing, cost L); lists visited nearest-first
  2. skip a list if its free sphere bound already exceeds the threshold
  3. otherwise evaluate the list's hull bound (cost R); skip if above threshold
  4. evaluate every block hull in the list (cost R_b + 1 each), scan surviving blocks
Threshold = (current K-th best distance)^2 / (1 + eps)^2, so any pruned cell
provably holds no point closer than kth / (1 + eps).
"""

from dataclasses import dataclass

import numpy as np

from .evaluate import K, CurvePoint, recall
from .hulls import BlockLayout, CellHulls, lower_bound_sq
from .ivf import IVFIndex, sq_dists


@dataclass(frozen=True)
class HullIndex:
    ivf: IVFIndex
    layout: BlockLayout
    list_hulls: CellHulls
    block_hulls: CellHulls


@dataclass
class SearchCost:
    routing: float = 0.0
    bounds: float = 0.0
    scanned: float = 0.0


def search_one(q: np.ndarray, hx: HullIndex, eps: float) -> tuple[np.ndarray, SearchCost]:
    ivf, layout = hx.ivf, hx.layout
    cost = SearchCost(routing=ivf.n_lists)
    centroid_d = np.sqrt(sq_dists(q[None], ivf.centroids)[0])
    sphere_lb_sq = np.maximum(centroid_d - hx.list_hulls.radius, 0.0) ** 2
    shrink = 1.0 / (1.0 + eps) ** 2
    best_d = np.full(K, np.inf, dtype=np.float32)
    best_i = np.full(K, -1, dtype=np.int64)
    list_cost = hx.list_hulls.n_components
    block_cost = hx.block_hulls.n_components + 1

    for l in np.argsort(centroid_d):
        threshold = best_d[-1] * shrink
        if sphere_lb_sq[l] > threshold:
            continue
        cost.bounds += list_cost
        if lower_bound_sq(q, hx.list_hulls, l) > threshold:
            continue
        blocks = slice(layout.list_blocks[l], layout.list_blocks[l + 1])
        block_lb = lower_bound_sq(q, hx.block_hulls, blocks)
        cost.bounds += block_cost * len(block_lb)
        for b in np.flatnonzero(block_lb <= threshold) + blocks.start:
            rows = slice(layout.block_offsets[b], layout.block_offsets[b + 1])
            diff = ivf.vectors[rows] - q
            d = np.einsum("nd,nd->n", diff, diff)
            cost.scanned += len(d)
            merged_d = np.concatenate([best_d, d])
            merged_i = np.concatenate([best_i, np.arange(rows.start, rows.stop)])
            keep = np.argsort(merged_d)[:K]
            best_d, best_i = merged_d[keep], merged_i[keep]
    return best_i, cost


def bounded_curve(hx: HullIndex, queries: np.ndarray, gt: np.ndarray, eps_values) -> list[CurvePoint]:
    curve = []
    for eps in eps_values:
        results = [search_one(q, hx, eps) for q in queries]
        found = np.stack([r[0] for r in results])
        costs = [r[1] for r in results]
        scanned = float(np.mean([c.scanned for c in costs]))
        total = float(np.mean([c.routing + c.bounds + c.scanned for c in costs]))
        curve.append(CurvePoint(eps, recall(found, gt), scanned, total))
    return curve
