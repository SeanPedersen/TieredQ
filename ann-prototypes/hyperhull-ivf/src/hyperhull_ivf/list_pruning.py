"""Progressive list pruning with a calibrated cell hull (real search, running K-th distance).

Each hull gives est_l(q) ≈ distance from q to list l's nearest member. Search visits lists
in est order and stops once the next list's est exceeds slack × the running K-th best distance.
  * shrunk sphere: (||q - c_l|| - alpha * rms_l)_+^2 — one float per list, routing cost L.
    alpha = 0 is plain adaptive IVF with centroid scores — the control.
  * PCA sketch: every vector keeps R global-PCA coords + residual norm; est_l = min over the
    list of ||z_q - z_x||^2 + o_q^2 + o_x^2 - 2·gamma·o_q·o_x. Routing cost N·(R+1)/D.

Hull parameters are tuned on the first half of the queries; reported numbers use the second half.
scanned = vectors (or codes) in visited lists; total = scanned + routing cost.

Usage: uv run hyperhull-prune [glove100k|dbpedia100k] [n_lists]   (default: dbpedia100k 256)
"""

import sys
import time

import numpy as np

from .bench import load_data
from .evaluate import QueryCache, adaptive_curve, build_cache, cost_at_recall, fixed_nprobe_curve
from .ivf import IVFIndex, build_ivf, sq_dists
from .routers import Router, centroid_router

DEFAULT_LISTS = 256
SKETCH_COMPONENTS = 16
GAMMAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
ALPHAS = np.round(np.arange(0.0, 1.21, 0.1), 2)
SLACKS = np.round(np.geomspace(0.3, 8.0, 60), 4)
NPROBES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
TARGETS = (0.90, 0.95, 0.99)
TUNE_TARGET = 0.99


def rms_radius(index: IVFIndex) -> np.ndarray:
    return np.array(
        [np.sqrt(((index.vectors[index.list_slice(l)] - index.centroids[l]) ** 2).sum(1).mean()) for l in range(index.n_lists)],
        dtype=np.float32,
    )


def shrunk_sphere_router(index: IVFIndex, rms: np.ndarray, alpha: float) -> Router:
    def score(q):
        return np.maximum(np.sqrt(sq_dists(q, index.centroids)) - alpha * rms[None], 0.0) ** 2

    return Router(f"shrunk-sphere a={alpha:g}", score, index.n_lists)


def pca_sketch_router(index: IVFIndex, n_components: int, gamma: float, sketch=None) -> Router:
    z_data, o_data, mean, basis = sketch if sketch is not None else build_sketch(index, n_components)

    def score(q):
        v = q - mean
        z_q = v @ basis.T
        o_q = np.sqrt(np.maximum((v**2).sum(1) - (z_q**2).sum(1), 0.0))
        per_point = sq_dists(z_q, z_data) + o_q[:, None] ** 2 + o_data[None] ** 2 - 2 * gamma * o_q[:, None] * o_data[None]
        return np.minimum.reduceat(per_point, index.offsets[:-1], axis=1)

    cost = n_components + len(index.vectors) * (n_components + 1) / index.vectors.shape[1]
    return Router(f"pca{n_components}-sketch g={gamma:g}", score, cost)


def build_sketch(index: IVFIndex, n_components: int, sample: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(index.vectors), size=min(sample, len(index.vectors)), replace=False)
    mean = index.vectors[rows].mean(0)
    _, _, vt = np.linalg.svd(index.vectors[rows] - mean, full_matrices=False)
    basis = vt[:n_components].astype(np.float32)
    v = index.vectors - mean
    z = v @ basis.T
    o = np.sqrt(np.maximum((v**2).sum(1) - (z**2).sum(1), 0.0))
    return z.astype(np.float32), o.astype(np.float32), mean, basis


def _subset(cache: QueryCache, rows: np.ndarray) -> QueryCache:
    return QueryCache(cache.gt[rows], cache.list_topk_d[rows], cache.list_topk_i[rows])


def _cell(curve, target):
    scanned, total = cost_at_recall(curve, target, "scanned"), cost_at_recall(curve, target, "total")
    return f"{scanned:7.0f} /{total:7.0f}" if scanned is not None else f"{'-':>7} /{'-':>7}"


def _tune(make_router, grid, adaptive, q_train, c_train, label):
    scores = {p: cost_at_recall(adaptive(make_router(p), q_train, c_train), TUNE_TARGET, "total") for p in grid}
    print(f"  tuning {label} (train, total cost at R>=0.99): " + "  ".join(f"{p:g}:{v:.0f}" for p, v in scores.items() if v), flush=True)
    return min((p for p in grid if scores[p] is not None), key=lambda p: scores[p])


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dbpedia100k"
    n_lists = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LISTS
    t0 = time.perf_counter()
    base, queries, _ = load_data(dataset)
    index = build_ivf(base, n_lists, train_size=len(base))
    del base
    cache = build_cache(index, queries)
    rms = rms_radius(index)
    print(f"{dataset}: N={len(index.vectors)} D={index.vectors.shape[1]} lists={n_lists} "
          f"(size {index.sizes.min()}-{index.sizes.max()}), built in {time.perf_counter() - t0:.0f}s", flush=True)

    half = len(queries) // 2
    train, test = np.arange(half), np.arange(half, len(queries))
    q_train, q_test = queries[train], queries[test]
    c_train, c_test = _subset(cache, train), _subset(cache, test)

    def adaptive(router, q, c):
        return adaptive_curve(index, c, router, q, SLACKS, n_lists)

    best_alpha = _tune(lambda a: shrunk_sphere_router(index, rms, a), ALPHAS, adaptive, q_train, c_train, "alpha")
    sketch = build_sketch(index, SKETCH_COMPONENTS)
    best_gamma = _tune(lambda g: pca_sketch_router(index, SKETCH_COMPONENTS, g, sketch), GAMMAS, adaptive, q_train, c_train, "gamma")

    curves = {
        "IVF fixed nprobe": fixed_nprobe_curve(index, c_test, centroid_router(index), q_test, [n for n in NPROBES if n <= n_lists]),
        "IVF adaptive (a=0 control)": adaptive(shrunk_sphere_router(index, rms, 0.0), q_test, c_test),
        f"shrunk-sphere hull a={best_alpha:g}": adaptive(shrunk_sphere_router(index, rms, best_alpha), q_test, c_test),
        f"pca{SKETCH_COMPONENTS}-sketch hull g={best_gamma:g}": adaptive(
            pca_sketch_router(index, SKETCH_COMPONENTS, best_gamma, sketch), q_test, c_test
        ),
    }
    print(f"\n  test queries (Q={len(test)}): scanned / total (scanned + routing) to reach recall")
    print(f"  {'method':<32}" + "".join(f"  R>={t:<13}" for t in TARGETS))
    for name, curve in curves.items():
        print(f"  {name:<32}" + "".join(f"  {_cell(curve, t)}" for t in TARGETS), flush=True)


if __name__ == "__main__":
    main()
