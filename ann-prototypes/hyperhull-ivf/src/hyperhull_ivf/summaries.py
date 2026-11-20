"""Per-list geometric summaries ("hyper-hull" sketches) built once at index time.

For each inverted list we store:
  * bounding sphere radius around the centroid
  * a local PCA basis (R components) with min/max projection extents (a box
    in the cluster's principal subspace), the centroid's own projection (so
    query projections need only q·B), and a low quantile of the members'
    off-subspace residual energy
  * M extreme points: the members with min/max projection along the top
    M/2 principal axes (a sampled hull boundary)
"""

from dataclasses import dataclass

import numpy as np

from .ivf import IVFIndex

ORTH_FLOOR_QUANTILE = 0.1


@dataclass(frozen=True)
class ListSummaries:
    radius: np.ndarray  # (L,)
    pca_basis: np.ndarray  # (L, R, D)
    centroid_proj: np.ndarray  # (L, R)
    proj_min: np.ndarray  # (L, R)
    proj_max: np.ndarray  # (L, R)
    orth_sq_floor: np.ndarray  # (L,)
    hull_points: np.ndarray  # (L, M, D)


def _summarize_list(members: np.ndarray, centroid: np.ndarray, n_components: int, n_hull: int):
    if len(members) == 0:  # never probed usefully; centroid stands in for every point
        members = centroid[None]
    centered = members - centroid
    radius = np.sqrt((centered**2).sum(1).max())

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = np.zeros((n_components, members.shape[1]), dtype=np.float32)
    usable = min(n_components, len(vt))
    basis[:usable] = vt[:usable]

    proj = centered @ basis.T
    orth_sq = (centered**2).sum(1) - (proj**2).sum(1)

    axes = proj[:, : n_hull // 2]
    extreme_rows = np.concatenate([axes.argmin(0), axes.argmax(0)])
    hull = members[extreme_rows]

    return (
        radius,
        basis,
        basis @ centroid,
        proj.min(0),
        proj.max(0),
        np.quantile(orth_sq, ORTH_FLOOR_QUANTILE),
        hull,
    )


def build_summaries(index: IVFIndex, n_components: int = 8, n_hull: int = 16) -> ListSummaries:
    parts = [
        _summarize_list(index.vectors[index.list_slice(l)], index.centroids[l], n_components, n_hull)
        for l in range(index.n_lists)
    ]
    return ListSummaries(*(np.stack(p).astype(np.float32) for p in zip(*parts)))
