"""Tune spread-corrected fine-cell scores on the features saved by rank_gap.py.

Usage: rank_tune.py <export_dir>
score = d² + β·r² - α·spread·κ_n  (spread: iso d·rms or aniso σ_c(u)·d, κ_n = 1 or √(2 ln n)),
(β, α) chosen per target on train (min vectors scanned), reported on test. Also prints how the cells
holding the true top-10 differ from cells of the same centroid rank (size, rms, spread ratios).
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

from rank_gap import TARGETS, cells_needed

BETAS = (0.0, 0.5, 1.0, 1.5)
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def main() -> None:
    f = np.load(Path(sys.argv[1]) / "rank_gap.npz")
    sizes, rms = f["sizes"], f["rms"]
    empty = sizes == 0
    r2 = np.where(empty, 0, rms * rms)[None, :]
    log_n = np.sqrt(2 * np.log(np.maximum(sizes, 2)))[None, :]
    split = {sp: {k: f[f"{sp}_{k}"] for k in ("full", "aniso", "gt_cells")} for sp in ("train", "test")}

    def spread(d, kind, kappa):
        s = np.sqrt(np.maximum(d["full"], 0)) * rms[None, :] if kind == "iso" else d["aniso"]
        return s * (log_n if kappa else 1.0)

    def score(d, beta, alpha, kind, kappa):
        out = d["full"] + beta * r2 - alpha * spread(d, kind, kappa)
        out[:, empty] = np.inf
        return out

    for kind, kappa in product(("iso", "aniso"), (False, True)):
        grid = list(product(BETAS, ALPHAS))
        train = {g: cells_needed(score(split["train"], *g, kind, kappa), split["train"]["gt_cells"], sizes) for g in grid}
        line = []
        for t, target in enumerate(TARGETS):
            best = min(grid, key=lambda g: train[g][t][1])
            F, v = cells_needed(score(split["test"], *best, kind, kappa), split["test"]["gt_cells"], sizes)[t]
            line.append(f"R{target}: β={best[0]:.1f} α={best[1]:.1f} F={F:5d} vec={v:6.0f}")
        print(f"{kind:5s} κ={'√2ln n' if kappa else '1     '}  " + "   ".join(line), flush=True)

    # GT cells vs. cells of similar centroid rank (test split)
    d = split["test"]
    order = np.argsort(d["full"], axis=1)
    rank = np.empty_like(order)
    rank[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
    gt_cells = d["gt_cells"]
    gt_rank = np.take_along_axis(rank, gt_cells, axis=1).ravel()
    print("\nGT-cell centroid rank percentiles (50/75/90/95/99):", np.percentile(gt_rank, [50, 75, 90, 95, 99]).astype(int))
    peers = order[:, 20:2000]  # non-trivial rank band
    is_gt = np.zeros(order.shape, dtype=bool)
    np.put_along_axis(is_gt, gt_cells, True, axis=1)
    in_band = np.take_along_axis(is_gt, peers, axis=1)
    def stat(name, v):
        band = v[peers] if v.ndim == 1 else np.take_along_axis(v, peers, axis=1)
        print(f"  {name:22s} GT cells {np.median(band[in_band]):8.3f}   other cells in band {np.median(band[~in_band]):8.3f}")
    print("cells at centroid rank 20..2000:")
    stat("size", sizes.astype(float))
    stat("rms", rms)
    stat("aniso/(d·rms)", d["aniso"] / np.maximum(np.sqrt(np.maximum(d["full"], 0)) * rms[None, :], 1e-9))
    stat("d (centroid dist)", np.sqrt(np.maximum(d["full"], 0)))


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


def realistic(export: str) -> None:
    """β·r² folded into the fine bias of quantized fine scores, with the coarse shortlist T."""
    f = np.load(Path(export) / "rank_gap.npz")
    sizes, rms = f["sizes"], f["rms"]
    r2 = (rms * rms)[None, :]
    for base in ("router_all", "q4_256", "full"):
        for T in (256, 512, 1024):
            def score(sp, beta):
                s = f[f"{sp}_{base}"] + beta * r2
                s[f[f"{sp}_coarse_rank"] >= T] = np.inf
                return s
            betas = (0.0, 0.25, 0.5, 0.75, 1.0)
            train = {b: cells_needed(score("train", b), f["train_gt_cells"], sizes) for b in betas}
            line = []
            for t, target in enumerate(TARGETS):
                best = min(betas, key=lambda b: train[b][t][1])
                F, v = cells_needed(score("test", best), f["test_gt_cells"], sizes)[t]
                line.append(f"R{target}: β={best:.2f} F={F:5d} vec={v:6.0f}")
            print(f"{base:10s} T={T:4d}  " + "   ".join(line), flush=True)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "realistic":
    realistic(sys.argv[1])
