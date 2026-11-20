"""Hyper-hull lab: fast iteration on cell-level hull estimators.

Setup: GloVe-100 (100k random subset, L2-normalized), k-means 256; take a seed
cluster plus its nearest neighbours (close clusters = hard routing). 15% of each
cluster is held out as queries. Cells are either the clusters themselves or
sub-blocks of them.

For each estimator est(q, cell) ≈ min_x∈cell ||q - x||^2 we report:
  prune@R   : share of vectors pruned (cells with est > tau·kth^2) at recall ≥ R,
              using the true kth (isolates hull quality from search order)
  scan@R    : vectors scanned to reach recall R when visiting cells in est order
  cost/cell : distance-equivalents to evaluate the estimator for one cell
Learned estimators are fit on the first half of queries; all metrics use the second half.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyperhull_ivf.evaluate import CurvePoint, cost_at_recall
from hyperhull_ivf.ivf import kmeans, sq_dists
from hyperhull_ivf.real_data import load_glove

# HULL_LAB_DATA=glove (default) | dbpedia  (dbpedia reuses the benchmark's 100k subset cache)
DATASET = os.environ.get("HULL_LAB_DATA", "glove")
CACHE = Path(__file__).parent / ("cache.npz" if DATASET == "glove" else f"cache_{DATASET}.npz")
BENCH_CACHE = Path(__file__).resolve().parents[2] / ".cache" / f"{DATASET}100k.npz"
K = 10
N_LISTS = 256
HOLDOUT = 0.15
TAUS = np.geomspace(0.3, 4.0, 200)


# ---------------------------------------------------------------- data


def load_clusters():
    if CACHE.exists():
        c = np.load(CACHE)
        return c["base"], c["centroids"], c["assign"]
    base = load_glove(n_queries=1, n_base=100_000).base if DATASET == "glove" else np.load(BENCH_CACHE)["base"]
    centroids = kmeans(base, N_LISTS, iters=20)
    assign = sq_dists(base, centroids).argmin(1)
    np.savez(CACHE, base=base, centroids=centroids, assign=assign)
    return base, centroids, assign


@dataclass(frozen=True)
class Lab:
    data: np.ndarray  # (N, D) subset vectors, grouped by cell
    cell_offsets: np.ndarray  # (C + 1,)
    queries: np.ndarray  # (Q, D)
    true_min: np.ndarray  # (Q, C)
    kth: np.ndarray  # (Q,)
    gt_cell: np.ndarray  # (Q, K) cell of each true neighbour

    @property
    def sizes(self):
        return np.diff(self.cell_offsets)

    def cell(self, c):
        return self.data[self.cell_offsets[c] : self.cell_offsets[c + 1]]


def make_lab(n_clusters: int, block_size: int | None, seed: int = 0) -> Lab:
    base, centroids, assign = load_clusters()
    rng = np.random.default_rng(seed)
    sizes = np.bincount(assign, minlength=N_LISTS)
    seed_cluster = np.argsort(sizes)[N_LISTS // 2]
    group = np.argsort(sq_dists(centroids[seed_cluster][None], centroids)[0])[:n_clusters]

    cells, queries = [], []
    for g in group:
        members = base[assign == g][rng.permutation(sizes[g])]
        n_q = int(len(members) * HOLDOUT)
        queries.append(members[:n_q])
        kept = members[n_q:]
        if block_size is None:
            cells.append(kept)
            continue
        n_blocks = max(1, len(kept) // block_size)
        sub = sq_dists(kept, kmeans(kept, n_blocks, iters=10, seed=seed)).argmin(1)
        cells.extend(kept[sub == b] for b in range(n_blocks) if (sub == b).any())

    data = np.concatenate(cells)
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in cells])])
    queries = rng.permutation(np.concatenate(queries))
    d = sq_dists(queries, data)
    kth = np.partition(d, K - 1, axis=1)[:, K - 1]
    gt = np.argpartition(d, K - 1, axis=1)[:, :K]
    row_cell = np.repeat(np.arange(len(cells)), np.diff(offsets))
    return Lab(data, offsets, queries, np.minimum.reduceat(d, offsets[:-1], axis=1), kth, row_cell[gt])


# ---------------------------------------------------------------- metrics


def prune_at_recall(lab: Lab, est: np.ndarray, rows) -> dict[float, float]:
    kept_frac, recalls = [], []
    for tau in TAUS:
        keep = est[rows] <= tau * lab.kth[rows, None]
        recalls.append(np.take_along_axis(keep, lab.gt_cell[rows], 1).mean())
        kept_frac.append((keep * lab.sizes).sum(1).mean() / lab.sizes.sum())
    out = {}
    for target in (0.95, 0.99):
        ok = [1 - k for k, r in zip(kept_frac, recalls) if r >= target]
        out[target] = max(ok) if ok else 0.0
    return out


def scan_at_recall(lab: Lab, est: np.ndarray, rows) -> dict[float, float | None]:
    order = np.argsort(est[rows], 1)
    gt_cell = lab.gt_cell[rows]
    curve = []
    for n in range(1, est.shape[1] + 1):
        visited = order[:, :n]
        hit = (gt_cell[:, :, None] == visited[:, None, :]).any(-1).mean()
        scanned = lab.sizes[visited].sum(1).mean()
        curve.append(CurvePoint(n, float(hit), scanned, scanned))
    return {t: cost_at_recall(curve, t, "scanned") for t in (0.95, 0.99)}


# ---------------------------------------------------------------- estimators
# Each takes (lab, train_rows) and returns (est (Q, C), cost per cell).


def est_oracle(lab, _):
    return lab.true_min, float("nan")


def est_centroid(lab, _):
    cents = np.stack([lab.cell(c).mean(0) for c in range(len(lab.sizes))])
    return sq_dists(lab.queries, cents), 1.0


@dataclass(frozen=True)
class CellStats:
    centroid: np.ndarray  # (C, D)
    basis: np.ndarray  # (C, R, D) top principal directions
    eigval: np.ndarray  # (C, R) variance along them
    orth_var: np.ndarray  # (C,) mean variance per remaining dimension
    mean_sq_radius: np.ndarray  # (C,)
    max_radius: np.ndarray  # (C,)
    sizes: np.ndarray  # (C,)


_STATS_CACHE: dict = {}


def _cell_eig(v: np.ndarray, n_components: int):
    """Top eigenpairs of the cell covariance via the small Gram matrix (cells have << D points)."""
    n, dim = v.shape
    if n_components == 0:
        return np.zeros((0, dim)), np.zeros(0), (v**2).sum() / n / dim
    gram_vals, gram_vecs = np.linalg.eigh(v @ v.T / n)
    gram_vals, gram_vecs = np.maximum(gram_vals[::-1], 0), gram_vecs[:, ::-1]
    top = gram_vals[:n_components]
    basis = (v.T @ gram_vecs[:, :n_components]) / np.sqrt(np.maximum(top * n, 1e-12))
    basis = np.pad(basis, ((0, 0), (0, max(0, n_components - basis.shape[1]))))
    top = np.pad(top, (0, max(0, n_components - len(top))))
    rest_var = max((v**2).sum() / n - top.sum(), 0) / max(dim - n_components, 1)
    return basis.T, top, rest_var


def cell_stats(lab: Lab, n_components: int) -> CellStats:
    key = (id(lab), n_components)
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    parts = []
    for c in range(len(lab.sizes)):
        x = lab.cell(c)
        mu = x.mean(0)
        v = x - mu
        basis, evals, rest_var = _cell_eig(v, n_components)
        sq = (v**2).sum(1)
        parts.append((mu, basis, evals, rest_var, sq.mean(), np.sqrt(sq.max()), len(x)))
    _STATS_CACHE[key] = CellStats(*(np.stack(p) for p in zip(*parts)))
    return _STATS_CACHE[key]


def _project(lab: Lab, s: CellStats):
    """u = q - c split into principal coords z (Q, C, R) and residual energy (Q, C)."""
    u_sq = sq_dists(lab.queries, s.centroid)
    z = np.einsum("qd,crd->qcr", lab.queries, s.basis) - np.einsum("cd,crd->cr", s.centroid, s.basis)[None]
    return u_sq, z, np.maximum(u_sq - (z**2).sum(-1), 0.0)


def tuned(family, grid):
    """Pick the grid parameter maximizing prune@.99 on the training queries."""

    def fn(lab, train):
        best = max(grid, key=lambda p: prune_at_recall(lab, family(lab, p)[0], train)[0.99])
        est, cost = family(lab, best)
        fn.chosen = best
        return est, cost

    return fn


def fam_sphere_shrunk(lab, alpha):
    """(||u|| - alpha * rms_radius)_+^2 ; alpha·max_radius analogue is the sound sphere."""
    s = cell_stats(lab, 0)
    return np.maximum(np.sqrt(sq_dists(lab.queries, s.centroid)) - alpha * np.sqrt(s.mean_sq_radius), 0) ** 2, 1.0


def est_sphere_sound(lab, _):
    s = cell_stats(lab, 0)
    return np.maximum(np.sqrt(sq_dists(lab.queries, s.centroid)) - s.max_radius, 0) ** 2, 1.0


def gaussian_hull(n_components):
    """Extreme-value Gaussian hull: min_x ||q-x||^2 ≈ ||u||^2 + E||v||^2 - 2·kappa·sqrt(u'Σu),
    u'Σu from top-R eigenpairs + isotropic remainder. kappa ~ sqrt(2 ln n) for Gaussian cells."""

    def family(lab, kappa):
        s = cell_stats(lab, n_components)
        u_sq, z, orth_sq = _project(lab, s)
        spread = np.sqrt((z**2 * s.eigval[None]).sum(-1) + orth_sq * s.orth_var[None])
        return u_sq + s.mean_sq_radius[None] - 2 * kappa * spread, 1.0 + n_components

    return family


def ellipsoid_hull(n_components):
    """Distance to the ellipsoid with semi-axes alpha·sqrt(λ_i) (principal dims) times a ball of
    radius alpha·sqrt((D-R)·orth_var) in the residual space (approximated per-part)."""

    def family(lab, alpha):
        s = cell_stats(lab, n_components)
        u_sq, z, orth_sq = _project(lab, s)
        axes = alpha * np.sqrt(np.maximum(s.eigval, 1e-12))[None]
        scale = np.sqrt(((z / axes) ** 2).sum(-1))  # >1 outside the ellipsoid
        in_sub = (np.maximum(1 - 1 / np.maximum(scale, 1e-12), 0) ** 2) * (z**2).sum(-1)  # radial approx
        dim_rest = lab.data.shape[1] - n_components
        orth_r = alpha * np.sqrt(dim_rest * s.orth_var)[None]
        return in_sub + np.maximum(np.sqrt(orth_sq) - orth_r, 0) ** 2, 1.0 + n_components

    return family


def union_of_balls(m):
    """Cell covered by m sub-k-means balls; est = min_j (||q-s_j|| - alpha·rms_j)_+^2."""

    def family(lab, alpha):
        cents, radii, owner = [], [], []
        for c in range(len(lab.sizes)):
            x = lab.cell(c)
            k = min(m, len(x))
            sub = kmeans(x, k, iters=8)
            a = sq_dists(x, sub).argmin(1)
            for j in range(k):
                if (a == j).any():
                    cents.append(sub[j])
                    radii.append(np.sqrt(((x[a == j] - sub[j]) ** 2).sum(1).mean()))
                    owner.append(c)
        d = np.maximum(np.sqrt(sq_dists(lab.queries, np.stack(cents))) - alpha * np.array(radii), 0) ** 2
        est = np.full((len(lab.queries), len(lab.sizes)), np.inf)
        np.minimum.at(est.T, np.array(owner), d.T)
        return est, float(m)

    return family


ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
KAPPAS = np.round(np.arange(0.0, 4.01, 0.2), 2)

# ---------------------------------------------------------------- global-PCA hulls
# One global basis: the query is projected once (cost R, amortized over all cells),
# after which every cell hull is evaluated in R dims with scalar ops (cost ∝ R/D).


@dataclass(frozen=True)
class GlobalPCA:
    z_data: np.ndarray  # (N, R)
    o_data: np.ndarray  # (N,) residual norms outside the subspace
    z_q: np.ndarray  # (Q, R)
    o_q: np.ndarray  # (Q,)


_PCA_CACHE: dict = {}


def global_pca(lab: Lab, n_components: int) -> GlobalPCA:
    key = (id(lab), n_components)
    if key not in _PCA_CACHE:
        mu = lab.data.mean(0)
        basis = np.linalg.eigh(np.cov((lab.data - mu).T))[1][:, ::-1][:, :n_components].T
        def split(x):
            v = x - mu
            z = v @ basis.T
            return z, np.sqrt(np.maximum((v**2).sum(1) - (z**2).sum(1), 0))
        _PCA_CACHE[key] = GlobalPCA(*split(lab.data), *split(lab.queries))
    return _PCA_CACHE[key]


def _residual_mix(p: GlobalPCA, rows, gamma):
    """Per-point estimate ||z_q - z_x||^2 + o_q^2 + o_x^2 - 2·gamma·o_q·o_x.
    gamma = 1: sound ring bound (residuals aligned); gamma = 0: residuals orthogonal."""
    z = p.z_data[rows]
    o = p.o_data[rows]
    proj = sq_dists(p.z_q, z)
    return proj + p.o_q[:, None] ** 2 + o[None] ** 2 - 2 * gamma * p.o_q[:, None] * o[None]


def pca_point_min(n_components):
    def family(lab, gamma):
        p = global_pca(lab, n_components)
        est = np.minimum.reduceat(_residual_mix(p, slice(None), gamma), lab.cell_offsets[:-1], axis=1)
        cost = lab.sizes.mean() * (n_components + 1) / lab.data.shape[1]
        return est, cost
    return family


_HULL_CACHE: dict = {}


def _hull_vertex_rows(lab: Lab, n_components: int) -> list[np.ndarray]:
    from scipy.spatial import ConvexHull, QhullError
    key = (id(lab), n_components)
    if key not in _HULL_CACHE:
        p = global_pca(lab, n_components)
        rows = []
        for c in range(len(lab.sizes)):
            a, b = lab.cell_offsets[c], lab.cell_offsets[c + 1]
            try:
                rows.append(a + ConvexHull(p.z_data[a:b], qhull_options="QJ").vertices)
            except (QhullError, ValueError):
                rows.append(np.arange(a, b))
        _HULL_CACHE[key] = rows
    return _HULL_CACHE[key]


def pca_hull_vertices(n_components):
    """Only the convex-hull vertices of each cell in PCA space (+ their residual norms)."""
    def family(lab, gamma):
        vertex_rows = _hull_vertex_rows(lab, n_components)
        rows = np.concatenate(vertex_rows)
        offsets = np.concatenate([[0], np.cumsum([len(r) for r in vertex_rows])])
        est = np.minimum.reduceat(_residual_mix(global_pca(lab, n_components), rows, gamma), offsets[:-1], axis=1)
        cost = np.mean([len(r) for r in vertex_rows]) * (n_components + 1) / lab.data.shape[1]
        family.vertices = np.mean([len(r) for r in vertex_rows])
        return est, cost
    return family


def pca_box_ring(n_components):
    """Sound: per-cell box in global PCA coords x ring of residual norms."""
    def family(lab, _):
        p = global_pca(lab, n_components)
        offs = lab.cell_offsets[:-1]
        lo, hi = np.minimum.reduceat(p.z_data, offs), np.maximum.reduceat(p.z_data, offs)
        omin, omax = np.minimum.reduceat(p.o_data, offs), np.maximum.reduceat(p.o_data, offs)
        zq = p.z_q[:, None]
        box = ((np.maximum(zq - hi, 0) + np.maximum(lo - zq, 0)) ** 2).sum(-1)
        oq = p.o_q[:, None]
        ring = (np.maximum(oq - omax, 0) + np.maximum(omin - oq, 0)) ** 2
        return box + ring, 2 * (n_components + 1) / lab.data.shape[1]
    return family


GAMMAS = np.round(np.arange(0.0, 1.01, 0.1), 2)

ESTIMATORS = {
    "oracle (true min)": est_oracle,
    "centroid": est_centroid,
    "sphere sound (max radius)": est_sphere_sound,
    "sphere shrunk (alpha·rms)": tuned(fam_sphere_shrunk, ALPHAS),
    "gaussian hull R=0": tuned(gaussian_hull(0), KAPPAS),
    "gaussian hull R=2": tuned(gaussian_hull(2), KAPPAS),
    "gaussian hull R=4": tuned(gaussian_hull(4), KAPPAS),
    "gaussian hull R=8": tuned(gaussian_hull(8), KAPPAS),
    "ellipsoid hull R=4": tuned(ellipsoid_hull(4), ALPHAS),
    "ellipsoid hull R=8": tuned(ellipsoid_hull(8), ALPHAS),
    "union of 2 balls": tuned(union_of_balls(2), ALPHAS),
    "union of 4 balls": tuned(union_of_balls(4), ALPHAS),
    **{f"pca{r} box x ring (sound)": tuned(pca_box_ring(r), [0]) for r in (4, 8, 16)},
    **{f"pca{r} point-min sound (g=1)": tuned(pca_point_min(r), [1.0]) for r in (2, 4, 8, 16, 32)},
    **{f"pca{r} point-min approx": tuned(pca_point_min(r), GAMMAS) for r in (2, 4, 8, 16, 32)},
    **{f"pca{r} hull-vertices approx": tuned(pca_hull_vertices(r), GAMMAS) for r in (2, 3, 4, 5)},
}


def run(n_clusters: int, block_size: int | None, names=None) -> None:
    lab = make_lab(n_clusters, block_size)
    half = len(lab.queries) // 2
    train, test = np.arange(half), np.arange(half, len(lab.queries))
    gran = f"blocks~{block_size}" if block_size else "clusters"
    print(f"\n### {n_clusters} close clusters as {gran}: N={len(lab.data)} cells={len(lab.sizes)} "
          f"Q_test={len(test)}  mean cell={lab.sizes.mean():.0f}")
    print(f"  {'estimator':<34}{'cost/cell':>9}  prune@.95  prune@.99   scan@.95  scan@.99")
    for name, fn in ESTIMATORS.items():
        if names and not any(name.startswith(n) for n in names) and not name.startswith("oracle") and name != "centroid":
            continue
        est, cost = fn(lab, train)
        p, s = prune_at_recall(lab, est, test), scan_at_recall(lab, est, test)
        fmt = lambda v: f"{v:8.0f}" if v is not None else "       -"
        label = f"{name} [{fn.chosen:g}]" if hasattr(fn, "chosen") else name
        routing = cost * len(lab.sizes)
        print(f"  {label:<34}{cost:9.1f}  {p[0.95]:8.3f}  {p[0.99]:8.3f}  {fmt(s[0.95])}  {fmt(s[0.99])}"
              f"   (+routing {routing:.0f})", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    # HULL_LAB_LEVEL=clusters|blocks|both (default both)
    level = os.environ.get("HULL_LAB_LEVEL", "both")
    if level in ("clusters", "both"):
        run(8, None, only)
    if level in ("blocks", "both"):
        run(8, 32, only)
