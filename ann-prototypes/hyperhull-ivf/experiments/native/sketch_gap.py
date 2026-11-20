"""Idea 2: centroid-relative (residual) member sketches vs. absolute ones at equal bytes.

Usage: sketch_gap.py <export_dir>   (needs rank_gap.npz)
Pool per test query = members of the first F_POOL fine cells in router order (≈ the R≥0.99 scan).
Each variant estimates ||q-x||² ≈ ||z_q - ẑ_x||² + o_q² + o_x² - 2γ·o_q·o_x over its sketch dims
(o = exact norm outside those dims). Reported: candidates V (best by estimate) needed to keep 0.95 /
0.99 / 1.0 of the pool's true top-10, best γ per variant.
"""
import sys
from pathlib import Path

import numpy as np

from rank_gap import load

K = 10
F_POOL = 1265
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGETS = (0.95, 0.99, 1.0)
GAUSS_2BIT_SPAN = 1.494


def uniform(x: np.ndarray, levels: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    step = np.maximum(hi - lo, 1e-12) / (levels - 1)
    return lo + step * np.clip(np.rint((x - lo) / step), 0, levels - 1)


def minmax(x: np.ndarray, levels: int) -> np.ndarray:
    return uniform(x, levels, x.min(0), x.max(0))


def clipped(x: np.ndarray, levels: int, span: float) -> np.ndarray:
    mu, sd = x.mean(0), x.std(0)
    return uniform(x, levels, mu - span * sd, mu + span * sd)


def main() -> None:
    export = Path(sys.argv[1])
    a = load(export)
    vectors = np.asarray(a["vectors"])
    n = len(vectors)
    mean, proj = np.asarray(a["mean"]), np.asarray(a["proj"])
    fine_offsets = np.asarray(a["fine_offsets"])
    sizes = np.diff(fine_offsets)
    n_fine = len(sizes)
    cell_of_row = np.repeat(np.arange(n_fine), sizes)
    z = np.empty((n, 256), dtype=np.float32)
    norm2 = np.empty(n, dtype=np.float32)
    for s in range(0, n, 65536):
        c = vectors[s : s + 65536] - mean
        z[s : s + 65536] = c @ proj.T
        norm2[s : s + 65536] = (c * c).sum(1)
    nonempty = sizes > 0
    cent = np.zeros((n_fine, 256), dtype=np.float32)
    cent[nonempty] = np.add.reduceat(z, fine_offsets[:-1][nonempty], axis=0) / sizes[nonempty, None]
    cent_q = minmax(cent[nonempty], 16)
    cent_hat = np.zeros_like(cent)
    cent_hat[nonempty] = cent_q
    base = cent_hat[cell_of_row]

    def residual(dims, levels, span):
        e = z[:, :dims] - base[:, :dims]
        return base[:, :dims] + clipped(e, levels, span)

    variants = {  # name: (sketch ẑ over its dims, bytes per member)
        "abs4 p128 (current)": (minmax(z[:, :128], 16), 64),
        "abs2 p128": (clipped(z[:, :128], 4, GAUSS_2BIT_SPAN), 32),
        "res4 p128 ±3σ": (residual(128, 16, 3.0), 64),
        "res2 p128": (residual(128, 4, GAUSS_2BIT_SPAN), 32),
        "res2 p256": (residual(256, 4, GAUSS_2BIT_SPAN), 64),
        "abs4 p256": (minmax(z, 16), 128),
        "res4 p256 ±3σ": (residual(256, 16, 3.0), 128),
    }

    row_of_id = np.empty(n, dtype=np.int64)
    row_of_id[np.asarray(a["ids"])] = np.arange(n)
    queries = np.asarray(a["queries"])
    gt_rows = row_of_id[np.asarray(a["gt"])]
    router = np.load(export / "rank_gap.npz")["test_router"]
    pools = []
    for qi in range(len(queries)):
        cells = np.argsort(router[qi], kind="stable")[:F_POOL]
        pools.append(np.concatenate([np.arange(fine_offsets[c], fine_offsets[c + 1]) for c in cells]))

    print(f"pool: {np.mean([len(p) for p in pools]):.0f} vectors; GT in pool "
          f"{np.mean([np.isin(gt_rows[i], pools[i]).mean() for i in range(len(pools))]):.4f}")
    zq_all = (queries - mean) @ proj.T
    qn2_all = ((queries - mean) ** 2).sum(1)
    for name, (zhat, nbytes) in variants.items():
        dims = zhat.shape[1]
        o_x = np.sqrt(np.maximum(norm2 - (z[:, :dims] ** 2).sum(1), 0))
        best = None
        for gamma in GAMMAS:
            ranks = []
            for qi, pool in enumerate(pools):
                zq = zq_all[qi, :dims]
                o_q = np.sqrt(max(qn2_all[qi] - float(zq @ zq), 0.0))
                est = ((zhat[pool] - zq) ** 2).sum(1) + o_x[pool] ** 2 - 2 * gamma * o_q * o_x[pool]
                rank = np.empty(len(pool), dtype=np.int64)
                rank[np.argsort(est, kind="stable")] = np.arange(len(pool))
                in_pool = np.isin(pool, gt_rows[qi])
                ranks.extend(rank[in_pool].tolist())
            ranks = np.sort(ranks)
            need = [int(ranks[int(np.ceil(t * len(ranks))) - 1]) + 1 for t in TARGETS]
            if best is None or need[1] < best[1][1]:
                best = (gamma, need)
        gamma, need = best
        print(f"{name:22s} {nbytes:4d} B  γ={gamma:.2f}  V@" + "  ".join(f"{t}: {v:6d}" for t, v in zip(TARGETS, need)), flush=True)


if __name__ == "__main__":
    main()
