"""Per-vector outer hulls: scalar-quantization boxes (VA-file style) inside IVF lists.

Each vector's residual r = x - c_l is quantized per dimension with b bits over the
list's [min, max] range; the code identifies an axis-aligned box that provably
contains r. For a query residual q_r:
    ||q - x||^2 >= sum_d dist(q_r[d], [lo_d, hi_d])^2        (exact lower bound)
Search (exact at eps = 0, guaranteed (1+eps)-approximate otherwise):
  lists nearest-first; skip a list if its free sphere bound exceeds the threshold;
  otherwise bound every vector from its codes (cost b/32 per vector), then compute
  exact distances in ascending-bound order, stopping when the bound exceeds the threshold.
"""

from dataclasses import dataclass

import numpy as np

from .evaluate import K, CurvePoint, recall
from .hulls import BOUND_SAFETY
from .ivf import IVFIndex, sq_dists

RERANK_CHUNK = 16


@dataclass(frozen=True)
class QuantizedLists:
    bits: int
    codes: np.ndarray  # (N, D) uint8, rows aligned with ivf.vectors
    lo: np.ndarray  # (L, D) per-list residual minimum
    step: np.ndarray  # (L, D) per-list quantization step
    radius: np.ndarray  # (L,)


def quantize_lists(ivf: IVFIndex, bits: int) -> QuantizedLists:
    levels = 2**bits
    codes = np.empty(ivf.vectors.shape, dtype=np.uint8)
    lo = np.empty_like(ivf.centroids)
    step = np.empty_like(ivf.centroids)
    radius = np.zeros(ivf.n_lists, dtype=np.float32)
    for l in range(ivf.n_lists):
        sl = ivf.list_slice(l)
        resid = ivf.vectors[sl] - ivf.centroids[l]
        if len(resid) == 0:
            lo[l], step[l] = 0.0, 1.0
            continue
        lo[l] = resid.min(0)
        step[l] = np.maximum(resid.max(0) - lo[l], 1e-12) / levels
        codes[sl] = np.clip(((resid - lo[l]) / step[l]).astype(np.int64), 0, levels - 1)
        radius[l] = np.sqrt((resid**2).sum(1).max())
    return QuantizedLists(bits, codes, lo, step, radius)


def _box_lower_bound_sq(q_resid, codes, lo, step):
    cell_lo = lo + codes * step
    gap = np.maximum(cell_lo - q_resid, 0.0) + np.maximum(q_resid - (cell_lo + step), 0.0)
    bound = np.einsum("nd,nd->n", gap, gap)
    return np.maximum(bound * (1 - BOUND_SAFETY) - BOUND_SAFETY, 0.0)


@dataclass
class VACost:
    routing: float
    code_evals: int = 0
    exact: int = 0


def search_one(q, ivf: IVFIndex, ql: QuantizedLists, eps: float, max_lists: int):
    cost = VACost(routing=ivf.n_lists)
    centroid_d = np.sqrt(sq_dists(q[None], ivf.centroids)[0])
    sphere_lb_sq = np.maximum(centroid_d - ql.radius, 0.0) ** 2
    shrink = 1.0 / (1.0 + eps) ** 2
    best_d = np.full(K, np.inf, dtype=np.float32)
    best_i = np.full(K, -1, dtype=np.int64)

    for l in np.argsort(centroid_d)[:max_lists]:
        if sphere_lb_sq[l] > best_d[-1] * shrink:
            continue
        sl = ivf.list_slice(l)
        bounds = _box_lower_bound_sq(q - ivf.centroids[l], ql.codes[sl], ql.lo[l], ql.step[l])
        cost.code_evals += len(bounds)
        order = np.argsort(bounds)
        for start in range(0, len(order), RERANK_CHUNK):
            chunk = order[start : start + RERANK_CHUNK]
            chunk = chunk[bounds[chunk] <= best_d[-1] * shrink]
            if len(chunk) == 0:
                break
            diff = ivf.vectors[sl.start + chunk] - q
            d = np.einsum("nd,nd->n", diff, diff)
            cost.exact += len(d)
            merged_d = np.concatenate([best_d, d])
            merged_i = np.concatenate([best_i, sl.start + chunk])
            keep = np.argsort(merged_d)[:K]
            best_d, best_i = merged_d[keep], merged_i[keep]
    return best_i, cost


def va_curve(ivf, ql: QuantizedLists, queries, gt, eps_values, max_lists: int) -> list[CurvePoint]:
    code_weight = ql.bits / 32  # bytes read relative to a float32 vector
    curve = []
    for eps in eps_values:
        results = [search_one(q, ivf, ql, eps, max_lists) for q in queries]
        found = np.stack([r[0] for r in results])
        costs = [r[1] for r in results]
        exact = float(np.mean([c.exact for c in costs]))
        total = float(np.mean([c.routing + c.code_evals * code_weight + c.exact for c in costs]))
        curve.append(CurvePoint(eps, recall(found, gt), exact, total))
    return curve
