"""Sound outer hulls ("hyper-hulls") for cells of vectors, and their lower bounds.

A cell's hull is the product of
  * a box in the cell's local PCA subspace: per-axis [min, max] of B(x - c)
  * a ring in the orthogonal complement: [min, max] of ||o_x||, o_x = residual
  * a bounding sphere: ||x - c|| <= radius
Every member lies inside all three, so for a query q (z_q = B(q - c), o_q its residual):
    ||q - x||^2 = ||z_q - z_x||^2 + ||o_q - o_x||^2
               >= dist(z_q, box)^2 + ring_gap(||o_q||)^2           (product bound)
    ||q - x||   >= ||q - c|| - radius                              (sphere bound)
Both are exact lower bounds; we use the larger one.

Cells are either IVF lists or "blocks": sub-k-means partitions of each list,
laid out contiguously inside their list so a block is one slice of vectors.
"""

from dataclasses import dataclass

import numpy as np

from .ivf import IVFIndex, kmeans, sq_dists

# Guards float32 rounding in the bound so pruning never drops a true neighbour.
BOUND_SAFETY = 1e-4


@dataclass(frozen=True)
class CellHulls:
    centroid: np.ndarray  # (C, D)
    basis: np.ndarray  # (C, R, D)
    centroid_proj: np.ndarray  # (C, R)  B c
    proj_min: np.ndarray  # (C, R)
    proj_max: np.ndarray  # (C, R)
    orth_min: np.ndarray  # (C,)
    orth_max: np.ndarray  # (C,)
    radius: np.ndarray  # (C,)

    @property
    def n_components(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True)
class BlockLayout:
    block_offsets: np.ndarray  # (B + 1,) row ranges in index.vectors
    list_blocks: np.ndarray  # (L + 1,) block-id ranges per list


def _cell_hull(members: np.ndarray, n_components: int):
    centroid = members.mean(0)
    centered = members - centroid
    basis = np.zeros((n_components, members.shape[1]), dtype=np.float32)
    if len(members) > 1:
        vt = np.linalg.svd(centered, full_matrices=False)[2]
        usable = min(n_components, len(vt))
        basis[:usable] = vt[:usable]
    proj = centered @ basis.T
    orth = np.sqrt(np.maximum((centered**2).sum(1) - (proj**2).sum(1), 0.0))
    radius = np.sqrt((centered**2).sum(1).max())
    return centroid, basis, basis @ centroid, proj.min(0), proj.max(0), orth.min(), orth.max(), radius


def build_hulls(vectors: np.ndarray, offsets: np.ndarray, n_components: int) -> CellHulls:
    parts = [_cell_hull(vectors[a:b], n_components) for a, b in zip(offsets[:-1], offsets[1:])]
    return CellHulls(*(np.stack(p).astype(np.float32) for p in zip(*parts)))


def subdivide_lists(index: IVFIndex, block_size: int, seed: int = 0) -> tuple[IVFIndex, BlockLayout]:
    """Split every list into ~block_size sub-clusters; permutes rows *within* lists in place."""
    block_offsets, list_blocks = [0], [0]
    ids = index.ids.copy()
    for l in range(index.n_lists):
        sl = index.list_slice(l)
        members = index.vectors[sl]
        n_blocks = max(1, -(-len(members) // block_size))
        if n_blocks > 1:
            centers = kmeans(members, n_blocks, iters=8, seed=seed)
            assign = sq_dists(members, centers).argmin(1)
            order = np.argsort(assign, kind="stable")
            index.vectors[sl] = members[order]
            ids[sl] = ids[sl][order]
            counts = np.bincount(assign, minlength=n_blocks)
            counts = counts[counts > 0]
        else:
            counts = np.array([len(members)])
        block_offsets.extend(sl.start + np.cumsum(counts))
        list_blocks.append(list_blocks[-1] + len(counts))
    reordered = IVFIndex(index.centroids, index.vectors, ids, index.offsets)
    return reordered, BlockLayout(np.array(block_offsets), np.array(list_blocks))


@dataclass(frozen=True)
class SharedBlockHulls:
    """Block hulls expressed in their *list's* PCA coordinates (basis from list hulls).

    Projecting q once per list (cost R) makes every block bound in that list cost
    only O(R) scalar ops instead of R full-dimensional dot products.
    """

    box_min: np.ndarray  # (B, R)
    box_max: np.ndarray  # (B, R)
    sub_center: np.ndarray  # (B, R)
    sub_radius: np.ndarray  # (B,)  max ||z_x - sub_center||
    orth_min: np.ndarray  # (B,)  residual norms outside the list subspace
    orth_max: np.ndarray  # (B,)


def build_shared_block_hulls(index: IVFIndex, layout: BlockLayout, list_hulls: CellHulls) -> SharedBlockHulls:
    parts = []
    for l in range(index.n_lists):
        basis, c = list_hulls.basis[l], list_hulls.centroid[l]
        for b in range(layout.list_blocks[l], layout.list_blocks[l + 1]):
            centered = index.vectors[layout.block_offsets[b] : layout.block_offsets[b + 1]] - c
            z = centered @ basis.T
            orth = np.sqrt(np.maximum((centered**2).sum(1) - (z**2).sum(1), 0.0))
            center = z.mean(0)
            radius = np.sqrt(((z - center) ** 2).sum(1).max())
            parts.append((z.min(0), z.max(0), center, radius, orth.min(), orth.max()))
    return SharedBlockHulls(*(np.stack(p).astype(np.float32) for p in zip(*parts)))


def project_to_list(q: np.ndarray, list_hulls: CellHulls, l: int) -> tuple[np.ndarray, float]:
    """q in list l's PCA coordinates, plus the norm of its residual outside that subspace."""
    offset = q - list_hulls.centroid[l]
    z = list_hulls.basis[l] @ offset
    return z, float(np.sqrt(max(offset @ offset - z @ z, 0.0)))


def shared_block_lower_bound_sq(z_q: np.ndarray, orth_q: float, h: SharedBlockHulls, blocks: slice) -> np.ndarray:
    box = ((np.maximum(z_q - h.box_max[blocks], 0) + np.maximum(h.box_min[blocks] - z_q, 0)) ** 2).sum(-1)
    sub = np.sqrt(((z_q - h.sub_center[blocks]) ** 2).sum(-1))
    sphere = np.maximum(sub - h.sub_radius[blocks], 0) ** 2
    ring = (np.maximum(orth_q - h.orth_max[blocks], 0) + np.maximum(h.orth_min[blocks] - orth_q, 0)) ** 2
    bound = np.maximum(box, sphere) + ring
    return np.maximum(bound * (1 - BOUND_SAFETY) - BOUND_SAFETY, 0.0)


def lower_bound_sq(q: np.ndarray, hulls: CellHulls, cells) -> np.ndarray:
    """Squared lower bound on min_x ||q - x||^2 over each cell in `cells` (int or slice)."""
    offset = q - hulls.centroid[cells]
    centroid_sq = (offset * offset).sum(-1)
    z = hulls.basis[cells] @ q - hulls.centroid_proj[cells]
    box_gap = np.maximum(z - hulls.proj_max[cells], 0.0) + np.maximum(hulls.proj_min[cells] - z, 0.0)
    orth = np.sqrt(np.maximum(centroid_sq - (z * z).sum(-1), 0.0))
    ring_gap = np.maximum(orth - hulls.orth_max[cells], 0.0) + np.maximum(hulls.orth_min[cells] - orth, 0.0)
    product = (box_gap * box_gap).sum(-1) + ring_gap * ring_gap
    sphere = np.maximum(np.sqrt(centroid_sq) - hulls.radius[cells], 0.0) ** 2
    return np.maximum(np.maximum(product, sphere) * (1 - BOUND_SAFETY) - BOUND_SAFETY, 0.0)
