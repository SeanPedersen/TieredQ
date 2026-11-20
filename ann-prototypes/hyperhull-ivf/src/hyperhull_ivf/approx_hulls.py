"""Approximate cell hulls as list routers (score ≈ distance to a cell's nearest member).

Each family maps a parameter to a (Q, L) score matrix; parameters are tuned on train queries
by clusters needed for a recall target (see hull_clustering).
  shrunk sphere   (||q-c|| - alpha·rms)_+^2                                   cost L
  gaussian hull   ||u||^2 + E||v||^2 - 2·kappa·sqrt(u'Σu), Σ = top-R eigs + iso rest   cost L(R+1)
  local pca box   box gap^2 in the cell's R-dim PCA box shrunk by beta + ||o_q||^2 + q10(||o_x||^2)  cost L(R+1)
  pca sketch      min_x ||z_q-z_x||^2 + o_q^2 + o_x^2 - 2·gamma·o_q·o_x, global R-dim PCA per vector  cost N(R+1)/D
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ivf import IVFIndex, sq_dists

LOCAL_COMPONENTS = 8
SKETCH_COMPONENTS = 16
QUERY_CHUNK = 64


@dataclass(frozen=True)
class Family:
    name: str
    grid: tuple[float, ...]
    scores: Callable[[np.ndarray, float], np.ndarray]  # (queries, param) -> (Q, L)
    cost: float  # routing distance-equivalents per query


def _chunked(fn, queries):
    return np.concatenate([fn(queries[s : s + QUERY_CHUNK]) for s in range(0, len(queries), QUERY_CHUNK)])


@dataclass(frozen=True)
class LocalPCA:
    basis: np.ndarray  # (L, R, D)
    centroid_proj: np.ndarray  # (L, R)
    eigval: np.ndarray  # (L, R)
    rest_var: np.ndarray  # (L,) per-dimension variance outside the subspace
    proj_min: np.ndarray  # (L, R)
    proj_max: np.ndarray  # (L, R)
    orth_q10: np.ndarray  # (L,) 10% quantile of members' squared residual
    mean_sq: np.ndarray  # (L,)
    rms: np.ndarray  # (L,)


def build_local_pca(index: IVFIndex, n_components: int = LOCAL_COMPONENTS) -> LocalPCA:
    dim = index.vectors.shape[1]
    parts = []
    for l in range(index.n_lists):
        x = index.vectors[index.list_slice(l)]
        c = index.centroids[l]
        v = x - c
        n = len(x)
        gram_vals, gram_vecs = np.linalg.eigh(v @ v.T / n)  # cells have far fewer points than dims
        gram_vals, gram_vecs = np.maximum(gram_vals[::-1], 0), gram_vecs[:, ::-1]
        k = min(n_components, n)
        top = gram_vals[:k]
        basis = ((v.T @ gram_vecs[:, :k]) / np.sqrt(np.maximum(top * n, 1e-12))).T
        basis = np.pad(basis, ((0, n_components - k), (0, 0)))
        top = np.pad(top, (0, n_components - k))
        z = v @ basis.T
        sq = (v**2).sum(1)
        orth = np.maximum(sq - (z**2).sum(1), 0)
        rest = max(sq.mean() - top.sum(), 0) / max(dim - n_components, 1)
        parts.append((basis, basis @ c, top, rest, z.min(0), z.max(0), np.quantile(orth, 0.1), sq.mean(), np.sqrt(sq.mean())))
    return LocalPCA(*(np.stack(p).astype(np.float32) for p in zip(*parts)))


def _local_projection(q, index: IVFIndex, p: LocalPCA):
    n_lists, n_comp, dim = p.basis.shape
    z = (q @ p.basis.reshape(-1, dim).T).reshape(len(q), n_lists, n_comp) - p.centroid_proj[None]
    u_sq = sq_dists(q, index.centroids)
    return z, u_sq, np.maximum(u_sq - (z**2).sum(-1), 0)


def shrunk_sphere(index: IVFIndex, p: LocalPCA) -> Family:
    def scores(queries, alpha):
        return np.maximum(np.sqrt(sq_dists(queries, index.centroids)) - alpha * p.rms[None], 0) ** 2

    return Family("shrunk sphere", tuple(np.round(np.arange(0, 1.51, 0.1), 2)), scores, index.n_lists)


def gaussian_hull(index: IVFIndex, p: LocalPCA) -> Family:
    def scores(queries, kappa):
        def fn(q):
            z, u_sq, orth_sq = _local_projection(q, index, p)
            spread = np.sqrt((z**2 * p.eigval[None]).sum(-1) + orth_sq * p.rest_var[None])
            return u_sq + p.mean_sq[None] - 2 * kappa * spread

        return _chunked(fn, queries)

    cost = index.n_lists * (p.basis.shape[1] + 1)
    return Family("gaussian hull", tuple(np.round(np.arange(0, 6.01, 0.25), 2)), scores, cost)


def local_pca_box(index: IVFIndex, p: LocalPCA) -> Family:
    center = (p.proj_min + p.proj_max) / 2
    half = (p.proj_max - p.proj_min) / 2

    def scores(queries, beta):
        lo, hi = center - beta * half, center + beta * half

        def fn(q):
            z, _, orth_sq = _local_projection(q, index, p)
            gap = np.maximum(z - hi[None], 0) + np.maximum(lo[None] - z, 0)
            return (gap**2).sum(-1) + orth_sq + p.orth_q10[None]

        return _chunked(fn, queries)

    cost = index.n_lists * (p.basis.shape[1] + 1)
    return Family("local pca box", tuple(np.round(np.arange(0.1, 1.21, 0.1), 2)), scores, cost)


def pca_sketch(index: IVFIndex, n_components: int = SKETCH_COMPONENTS, sample: int = 20_000) -> Family:
    rng = np.random.default_rng(0)
    rows = rng.choice(len(index.vectors), size=min(sample, len(index.vectors)), replace=False)
    mean = index.vectors[rows].mean(0)
    basis = np.linalg.svd(index.vectors[rows] - mean, full_matrices=False)[2][:n_components].astype(np.float32)
    v = index.vectors - mean
    z_data = (v @ basis.T).astype(np.float32)
    o_data = np.sqrt(np.maximum((v**2).sum(1) - (z_data**2).sum(1), 0)).astype(np.float32)

    def scores(queries, gamma):
        def fn(q):
            vq = q - mean
            z_q = vq @ basis.T
            o_q = np.sqrt(np.maximum((vq**2).sum(1) - (z_q**2).sum(1), 0))
            per_point = sq_dists(z_q, z_data) + o_q[:, None] ** 2 + o_data[None] ** 2 - 2 * gamma * o_q[:, None] * o_data[None]
            return np.minimum.reduceat(per_point, index.offsets[:-1], axis=1)

        return _chunked(fn, queries)

    cost = n_components + len(index.vectors) * (n_components + 1) / index.vectors.shape[1]
    return Family(f"pca{n_components} sketch", tuple(np.round(np.arange(0, 1.01, 0.1), 2)), scores, cost)


def all_families(index: IVFIndex) -> list[Family]:
    p = build_local_pca(index)
    return [shrunk_sphere(index, p), gaussian_hull(index, p), local_pca_box(index, p), pca_sketch(index)]
