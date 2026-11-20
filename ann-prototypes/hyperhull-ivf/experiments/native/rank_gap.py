"""Fine-cell ranking gap to the hull oracle, on an exported index (no sketch / rerank effects).

Usage: rank_gap.py <export_dir> [T]
Scans fine cells in score order and reports, per ranking, the cells and vectors needed until the
true top-10 is covered at mean recall 0.90 / 0.95 / 0.99 (ranking quality alone: every vector of a
scanned cell counts as found). Rankings:
  router   quantized PCA-128 fine centroid score inside the T nearest coarse cells (what the C++ router does)
  pca256   exact member mean in PCA-256 (global)          full   exact member mean, all dims (global)
  oracle   min distance of the query to any member (the perfect hull)
Train = queries 0..499 (brute-force GT here), test = the export's queries (500..999).
Saves per-query features to <export_dir>/rank_gap.npz for learned rankings.
"""
import sys
from pathlib import Path

import numpy as np

from hyperhull_ivf.real_data import load_dbpedia_queries

K = 10
TARGETS = (0.90, 0.95, 0.99)
CHUNK = 50


def load(export: Path) -> dict:
    meta = {}
    for line in (export / "meta.txt").read_text().splitlines():
        parts = line.split()
        if parts[0] != "array":
            continue
        name, dtype, shape = parts[1], parts[2], tuple(int(s) for s in parts[3:])
        meta[name] = np.memmap(export / f"{name}.bin", dtype=dtype, mode="r", shape=shape)
    return meta


def unpack4(packed: np.ndarray) -> np.ndarray:
    n, half = packed.shape
    blocks = packed.reshape(n, half // 32, 1, 32)
    return np.concatenate([blocks & 0x0F, blocks >> 4], axis=2).reshape(n, half * 2).astype(np.float32)


def sq_dists(q: np.ndarray, x: np.ndarray, x_norm2: np.ndarray) -> np.ndarray:
    return x_norm2[None, :] - 2.0 * (q @ x.T) + (q * q).sum(1)[:, None]


def cells_needed(scores: np.ndarray, gt_cells: np.ndarray, sizes: np.ndarray) -> list[tuple[int, float]]:
    """Smallest F (cells) whose mean coverage reaches each target, and mean vectors scanned at that F."""
    order = np.argsort(scores, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(len(order))[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :]
    gt_rank = np.take_along_axis(rank, gt_cells, axis=1)  # (Q, K)
    gt_rank[~np.isfinite(np.take_along_axis(scores, gt_cells, axis=1))] = order.shape[1] + 1
    out = []
    for target in TARGETS:
        need = np.sort(gt_rank, axis=None)  # coverage(F) = #{gt_rank < F} / (Q·K)
        F = int(need[int(np.ceil(target * need.size)) - 1]) + 1
        vectors = np.cumsum(sizes[order], axis=1)[:, min(F, order.shape[1]) - 1].mean()
        out.append((F, float(vectors)))
    return out


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def tuned_families(feats: dict, sizes: np.ndarray, rms: np.ndarray) -> None:
    """Spread-corrected scores d² - α·spread; α tuned per target on train (min vectors), reported on test."""
    log_n = np.sqrt(2 * np.log(np.maximum(sizes, 2)))[None, :]
    families = {
        "iso  d²-α·d·rms": lambda f, a: f["full"] - a * np.sqrt(np.maximum(f["full"], 0)) * rms[None, :],
        "iso·√(2ln n)": lambda f, a: f["full"] - a * np.sqrt(np.maximum(f["full"], 0)) * rms[None, :] * log_n,
        "aniso d²-α·σ(u)d": lambda f, a: f["full"] - a * f["aniso"],
        "aniso·√(2ln n)": lambda f, a: f["full"] - a * f["aniso"] * log_n,
    }
    for name, score in families.items():
        print(f"\n[{name}]  α chosen on train → test cells / vectors")
        train = {a: cells_needed(score(feats["train"], a), feats["train"]["gt_cells"], sizes) for a in ALPHAS}
        line = []
        for t, target in enumerate(TARGETS):
            best = min(ALPHAS, key=lambda a: train[a][t][1])
            F, v = cells_needed(score(feats["test"], best), feats["test"]["gt_cells"], sizes)[t]
            line.append(f"R{target}: α={best:4.2f} F={F:5d} vec={v:7.0f} (train {train[best][t][1]:7.0f})")
        print("  " + "   ".join(line))


def main() -> None:
    export = Path(sys.argv[1])
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    a = load(export)
    vectors = np.asarray(a["vectors"])
    n, dim = vectors.shape
    fine_offsets = np.asarray(a["fine_offsets"])
    n_fine = len(fine_offsets) - 1
    m = n_fine // a["coarse_z"].shape[0]
    sizes = np.diff(fine_offsets)
    cell_of_row = np.repeat(np.arange(n_fine), sizes)
    row_of_id = np.empty(n, dtype=np.int64)
    row_of_id[np.asarray(a["ids"])] = np.arange(n)
    norm2 = np.einsum("ij,ij->i", vectors, vectors)

    mean, proj = np.asarray(a["mean"]), np.asarray(a["proj"])
    z = np.empty((n, proj.shape[0]), dtype=np.float32)
    for s in range(0, n, 65536):
        z[s : s + 65536] = (vectors[s : s + 65536] - mean) @ proj.T
    nonempty = sizes > 0
    def cell_means(x):
        sums = np.add.reduceat(x, fine_offsets[:-1][nonempty], axis=0)
        out = np.zeros((n_fine, x.shape[1]), dtype=np.float32)
        out[nonempty] = sums / sizes[nonempty, None]
        return out
    cent_full, cent_z = cell_means(vectors), cell_means(z)
    cent_full_n2, cent_z_n2 = (cent_full ** 2).sum(1), (cent_z ** 2).sum(1)
    # spread features: rms radius (full dims) per cell
    member_d2 = norm2 - 2 * np.einsum("ij,ij->i", vectors, cent_full[cell_of_row]) + cent_full_n2[cell_of_row]
    rms = np.sqrt(np.maximum(np.bincount(cell_of_row, member_d2, n_fine) / np.maximum(sizes, 1), 0))
    # per-cell variance per PCA dim and mean residual variance per non-PCA dim (anisotropic spread)
    dz = z - cent_z[cell_of_row]
    var_z = np.zeros_like(cent_z)
    var_z[nonempty] = np.add.reduceat(dz * dz, fine_offsets[:-1][nonempty], axis=0) / sizes[nonempty, None]
    pca_d2 = (dz * dz).sum(1)
    var_res = np.maximum(np.bincount(cell_of_row, member_d2 - pca_d2, n_fine), 0) / np.maximum(sizes, 1) / (dim - z.shape[1])
    # 4-bit per-dim SQ of the PCA-256 cell means (same quantizer as the router's fine centroids)
    lo, hi = cent_z[nonempty].min(0), cent_z[nonempty].max(0)
    step = np.maximum(hi - lo, 1e-12) / 15
    cent_q4 = lo + step * np.clip(np.rint((cent_z - lo) / step), 0, 15)
    cent_q4_n2 = (cent_q4 ** 2).sum(1)

    fine_deq = np.asarray(a["fine_lo"]) + np.asarray(a["fine_step"]) * unpack4(np.asarray(a["fine_codes"]))
    fine_bias = np.where(nonempty, np.asarray(a["fine_bias"]), np.inf)
    coarse_z = np.asarray(a["coarse_z"])

    all_q = load_dbpedia_queries(1000).astype(np.float32)
    splits = {"train": all_q[:500], "test": np.asarray(a["queries"])}
    test_gt_rows = row_of_id[np.asarray(a["gt"])]
    feats = {}
    for split, queries in splits.items():
        Q = len(queries)
        gt_rows = np.empty((Q, K), dtype=np.int64)
        sc = {k: np.empty((Q, n_fine), dtype=np.float32) for k in ("router", "router_all", "q4_256", "pca256", "full", "aniso", "oracle")}
        coarse_rank = np.empty((Q, n_fine), dtype=np.int32)
        for s in range(0, Q, CHUNK):
            q = queries[s : s + CHUNK]
            d = sq_dists(q, vectors, norm2)
            gt_rows[s : s + CHUNK] = np.argpartition(d, K, axis=1)[:, :K] if split == "train" else test_gt_rows[s : s + CHUNK]
            mins = np.full((len(q), n_fine), np.inf, dtype=np.float32)
            mins[:, nonempty] = np.minimum.reduceat(d, fine_offsets[:-1][nonempty], axis=1)
            sc["oracle"][s : s + CHUNK] = mins
            zq = (q - mean) @ proj.T
            sc["pca256"][s : s + CHUNK] = np.where(nonempty, sq_dists(zq, cent_z, cent_z_n2), np.inf)
            sc["full"][s : s + CHUNK] = np.where(nonempty, sq_dists(q, cent_full, cent_full_n2), np.inf)
            sc["q4_256"][s : s + CHUNK] = np.where(nonempty, sq_dists(zq, cent_q4, cent_q4_n2), np.inf)
            # σ_c(u)·d: sqrt(Σ_pca var_cd (zq_d - c_d)² + var_res_c · d_res²)
            weighted = (zq * zq) @ var_z.T - 2.0 * zq @ (var_z * cent_z).T + (var_z * cent_z * cent_z).sum(1)[None, :]
            d_res2 = np.maximum(sc["full"][s : s + CHUNK] - sc["pca256"][s : s + CHUNK], 0)
            sc["aniso"][s : s + CHUNK] = np.sqrt(np.maximum(weighted + var_res[None, :] * d_res2, 0))
            fine = fine_bias[None, :] - 2.0 * zq[:, :128] @ fine_deq.T
            coarse = (coarse_z ** 2).sum(1)[None, :] - 2.0 * zq[:, :64] @ coarse_z.T
            crank = np.argsort(np.argsort(coarse, axis=1), axis=1)
            coarse_rank[s : s + CHUNK] = np.repeat(crank, m, axis=1)
            sc["router_all"][s : s + CHUNK] = fine
            sc["router"][s : s + CHUNK] = np.where(coarse_rank[s : s + CHUNK] < T, fine, np.inf)
            print(f"{split} {s + len(q)}/{Q}", file=sys.stderr, flush=True)
        gt_cells = cell_of_row[gt_rows]
        print(f"\n[{split}] cells / vectors scanned to cover true top-10 at mean recall {TARGETS}")
        for name, scores in sc.items():
            if name == "aniso":
                continue
            res = cells_needed(scores, gt_cells, sizes)
            print(f"  {name:11s} " + "   ".join(f"F={F:6d} vec={v:8.0f}" for F, v in res))
        distinct = np.array([len(set(r)) for r in gt_cells])
        print(f"  distinct GT cells per query: mean {distinct.mean():.2f}, median {np.median(distinct):.0f}")
        feats[split] = dict(gt_cells=gt_cells, coarse_rank=coarse_rank, **sc)
    tuned_families(feats, sizes, rms)
    np.savez(export / "rank_gap.npz", sizes=sizes, rms=rms,
             **{f"{sp}_{k}": v for sp, f in feats.items() for k, v in f.items()})


if __name__ == "__main__":
    main()
