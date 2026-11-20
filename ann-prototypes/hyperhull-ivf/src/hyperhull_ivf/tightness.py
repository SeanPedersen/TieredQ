"""Bound-tightness diagnostics for outer hulls.

For each query and cell (list or block), compares the hull lower bound with the
true min distance, against the final K-th NN distance:
  oracle  = share of vectors in cells whose true min > kth (what a perfect bound prunes)
  pruned  = share of vectors in cells whose lower bound > kth (what our bound prunes)
  per-component pruning (sphere / box / ring alone) shows which part does the work.
"""

import numpy as np

from .evaluate import K
from .hulls import CellHulls


def _components(q: np.ndarray, h: CellHulls):
    offset = q - h.centroid
    centroid_sq = (offset * offset).sum(-1)
    z = h.basis @ q - h.centroid_proj
    box = ((np.maximum(z - h.proj_max, 0) + np.maximum(h.proj_min - z, 0)) ** 2).sum(-1)
    orth = np.sqrt(np.maximum(centroid_sq - (z * z).sum(-1), 0))
    ring = (np.maximum(orth - h.orth_max, 0) + np.maximum(h.orth_min - orth, 0)) ** 2
    sphere = np.maximum(np.sqrt(centroid_sq) - h.radius, 0) ** 2
    return {"sphere": sphere, "box": box, "ring": ring, "box+ring": box + ring, "all": np.maximum(box + ring, sphere)}


def diagnose(vectors: np.ndarray, offsets: np.ndarray, hulls: CellHulls, queries: np.ndarray) -> dict[str, float]:
    sizes = np.diff(offsets)
    totals: dict[str, float] = {}
    for q in queries:
        d = ((vectors - q) ** 2).sum(1)
        kth = np.partition(d, K - 1)[K - 1]
        true_min = np.minimum.reduceat(d, offsets[:-1])
        shares = {"oracle": sizes[true_min > kth].sum()}
        for name, lb in _components(q, hulls).items():
            shares[name] = sizes[lb > kth].sum()
        far = true_min > kth
        shares["median LB/true (prunable cells)"] = np.median(_components(q, hulls)["all"][far] / true_min[far]) * len(vectors) if far.any() else 0
        for key, value in shares.items():
            totals[key] = totals.get(key, 0.0) + value / len(vectors) / len(queries)
    return totals


def print_diagnosis(label: str, result: dict[str, float]) -> None:
    cells = "  ".join(f"{k}={v:.3f}" for k, v in result.items())
    print(f"  {label:<10} {cells}", flush=True)
