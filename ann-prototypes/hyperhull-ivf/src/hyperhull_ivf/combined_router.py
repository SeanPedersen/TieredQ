"""Combined hull router: coarse hull → quantized fine hull → (two-pass) quantized member sketch.

Index (from two_stage.build_two_level): L coarse k-means cells, each split into m fine cells,
vectors stored coarse-major / fine-minor. One global PCA basis (fit on a sample of the data);
every query is projected once and all routing stages read prefixes of that projection.

Per query and configuration:
  1. coarse hull : coarse-centroid distances in r_c dims (float)      → shortlist the T nearest coarse cells
  2. fine hull   : fine-centroid distances in r_f dims, centroids stored as float or b-bit SQ
  3. member sketch for the F best fine cells, per member vector
                   est_x = ||z_q - ẑ_x||^2 + o_q^2 + o_x^2 - 2·GAMMA·o_q·o_x
                   ẑ_x = s-dim prefix of a 4-bit SQ sketch, o = exact residual norm outside the prefix
       single  : one pass with s dims
       two-pass: 32 dims over all members, then the top P re-estimated with 128 dims
  4. scan      : sketched vectors in estimate order, then the remaining shortlisted fine cells,
                 then non-shortlisted coarse cells; exact distances.
Stopping:
  fixed    : uniform vector budget V per query (vectors for recall t = t-quantile of vec_needed)
  adaptive : per query V_q = #{sketched x : est_x <= c·est_(K)}, est_(K) = K-th smallest estimate
Cost (distance-equivalents; a D-dim float dot product = 1): projection max(dims used) + coarse L·r_c/D
  + fine cells·r_f·bits/(32D) + sketch reads (s·4/(32D) per vector) + exact vectors.
Configs (incl. stopping rule) are chosen per recall target on train queries, reported on test queries.

Usage: uv run hyperhull-combined [n_coarse] [m] [n_queries]   (defaults 1024 32 1000; half train / half test)
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .evaluate import K
from .ivf import sq_dists
from .real_data import load_dbpedia
from .two_stage import TwoLevel, build_two_level, ground_truth

PCA_DIMS = 128
PCA_SAMPLE = 50_000
COARSE_DIMS = (64, 128)
SHORTLISTS = (32, 64, 128, 256)
FINE_HULLS = ((128, 32), (128, 4), (64, 4), (32, 4))  # (dims, bits)
SKETCHED_CELLS = (0, 128, 256, 512, 1024, 2048, 4096)
SKETCH_BITS = 4
SINGLE_DIMS = (32, 64, 128)
TWO_PASS = (32, 128)  # first-pass dims, second-pass dims
TWO_PASS_KEEP = (256, 1024)
ADAPTIVE_C = (1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5)
GAMMA = 0.5
TARGETS = (0.90, 0.95, 0.99)
WORKERS = min(32, os.cpu_count() or 1)


@dataclass(frozen=True)
class Space:
    """Global PCA space: query projection basis, quantized sketches and centroids."""

    mean: np.ndarray
    basis: np.ndarray  # (PCA_DIMS, D)
    z_sketch: np.ndarray  # (N, PCA_DIMS) dequantized SKETCH_BITS codes
    norm2_data: np.ndarray  # (N,) ||x - mean||^2
    cum_z2_data: dict  # s -> (N,) exact sum of first s squared coords
    z_coarse: np.ndarray  # (L, PCA_DIMS) float
    z_fine: dict  # bits -> (L·m, PCA_DIMS), dequantized for bits < 32


def _quantize(z: np.ndarray, bits: int) -> np.ndarray:
    """Per-dimension uniform scalar quantization; returns the dequantized values."""
    if bits >= 32:
        return z
    lo, hi = z.min(0), z.max(0)
    levels = 2**bits - 1
    step = np.maximum(hi - lo, 1e-12) / levels
    codes = np.clip(np.rint((z - lo) / step), 0, levels)
    return (lo + codes * step).astype(np.float32)


def build_space(index: TwoLevel, seed: int = 0) -> Space:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(index.vectors), size=min(PCA_SAMPLE, len(index.vectors)), replace=False)
    mean = index.vectors[rows].mean(0)
    basis = np.linalg.svd(index.vectors[rows] - mean, full_matrices=False)[2][:PCA_DIMS].astype(np.float32)
    centered = index.vectors - mean
    z = (centered @ basis.T).astype(np.float32)
    z2 = z * z
    z_fine = ((index.fine_centroids - mean) @ basis.T).astype(np.float32)
    sketch_dims = set(SINGLE_DIMS) | set(TWO_PASS)
    return Space(
        mean=mean,
        basis=basis,
        z_sketch=_quantize(z, SKETCH_BITS),
        norm2_data=(centered * centered).sum(1),
        cum_z2_data={s: z2[:, :s].sum(1) for s in sketch_dims},
        z_coarse=((index.coarse_centroids - mean) @ basis.T).astype(np.float32),
        z_fine={b: _quantize(z_fine, b) for b in {bits for _, bits in FINE_HULLS}},
    )


# Globals for forked workers (read-only, shared copy-on-write).
_STATE: dict = {}


def _rows_of(cells: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated row ids of `cells` (in order) and per-cell start offsets into that array."""
    starts, ends = offsets[cells], offsets[cells + 1]
    lengths = ends - starts
    cell_starts = np.concatenate([[0], np.cumsum(lengths)])
    rows = np.repeat(starts - cell_starts[:-1], lengths) + np.arange(cell_starts[-1])
    return rows, cell_starts


def _adaptive_budgets(sorted_est: np.ndarray) -> np.ndarray:
    n = len(sorted_est)
    if n == 0:
        return np.zeros(len(ADAPTIVE_C))
    kth = sorted_est[min(K, n) - 1]
    counts = np.searchsorted(sorted_est, np.array(ADAPTIVE_C) * kth, side="right")
    return np.clip(counts, min(K, n), n).astype(float)


def evaluate_query(qi: int) -> dict:
    """All configs for one query → {config: (vec_needed per gt, routing, adaptive budgets per c)}."""
    s_ = _STATE
    index: TwoLevel = s_["index"]
    space: Space = s_["space"]
    m, dim = index.m, index.vectors.shape[1]
    n_coarse = len(index.coarse_centroids)
    q = s_["queries"][qi]
    gt = s_["gt"][qi]
    gt_fine, gt_coarse = s_["row_fine"][gt], s_["row_coarse"][gt]
    fine_sizes, coarse_sizes = s_["fine_sizes"], s_["coarse_sizes"]
    sketch_read = SKETCH_BITS / (32 * dim)

    vq = q - space.mean
    zq = vq @ space.basis.T
    nq2 = float(vq @ vq)
    sketch_dims = sorted(set(SINGLE_DIMS) | set(TWO_PASS))
    o2q = {sd: max(nq2 - float((zq[:sd] ** 2).sum()), 0.0) for sd in sketch_dims}
    no_budget = np.full(len(ADAPTIVE_C), np.nan)
    out = {}
    for r_c in COARSE_DIMS:
        coarse_order = np.argsort(((space.z_coarse[:, :r_c] - zq[:r_c]) ** 2).sum(1))
        coarse_rank = np.empty(n_coarse, dtype=np.int64)
        coarse_rank[coarse_order] = np.arange(n_coarse)
        for t_short in SHORTLISTS:
            short = coarse_order[:t_short]
            fine_ids = (short[:, None] * m + np.arange(m)[None]).ravel()
            fine_ids = fine_ids[fine_sizes[fine_ids] > 0]
            in_short = coarse_rank[gt_coarse] < t_short
            rest_sizes = coarse_sizes[coarse_order[t_short:]]
            for r_f, f_bits in FINE_HULLS:
                fine_d = ((space.z_fine[f_bits][fine_ids, :r_f] - zq[:r_f]) ** 2).sum(1)
                fine_order = fine_ids[np.argsort(fine_d)]
                hull_routing = n_coarse * r_c / dim + len(fine_ids) * r_f * f_bits / (32 * dim)
                gt_fine_pos = np.full(len(gt), -1)
                for j in np.flatnonzero(in_short):
                    gt_fine_pos[j] = int(np.flatnonzero(fine_order == gt_fine[j])[0])
                cum_fine = np.cumsum(fine_sizes[fine_order])
                cum_rest = cum_fine[-1] + np.cumsum(rest_sizes)
                base_needed = np.array([
                    cum_fine[gt_fine_pos[j]] if in_short[j] else cum_rest[coarse_rank[gt_coarse[j]] - t_short]
                    for j in range(len(gt))
                ], dtype=float)
                head = (r_c, t_short, r_f, f_bits)
                out[head + (0, "none", 0, 0)] = (base_needed, max(r_c, r_f) + hull_routing, no_budget)

                fmax = min(max(SKETCHED_CELLS), len(fine_order))
                rows_max, cell_starts = _rows_of(fine_order[:fmax], index.fine_offsets)
                gt_row_idx = np.full(len(gt), -1)
                for j in np.flatnonzero((gt_fine_pos >= 0) & (gt_fine_pos < fmax)):
                    gt_row_idx[j] = int(np.flatnonzero(rows_max == gt[j])[0])
                est_by_dims = {}
                for sd in sketch_dims:
                    o2x = np.maximum(space.norm2_data[rows_max] - space.cum_z2_data[sd][rows_max], 0.0)
                    resid = o2q[sd] + o2x - 2 * GAMMA * np.sqrt(o2q[sd] * o2x)
                    est_by_dims[sd] = ((space.z_sketch[rows_max, :sd] - zq[:sd]) ** 2).sum(1) + resid

                for f_cells in SKETCHED_CELLS[1:]:
                    f = min(f_cells, len(fine_order))
                    n_rows = int(cell_starts[f])
                    sketched = (gt_fine_pos >= 0) & (gt_fine_pos < f)
                    own_idx = gt_row_idx[sketched]
                    for sd in SINGLE_DIMS:
                        est = est_by_dims[sd][:n_rows]
                        sorted_est = np.sort(est)
                        needed = base_needed.copy()
                        needed[sketched] = np.searchsorted(sorted_est, est[own_idx], side="right")
                        routing = max(r_c, r_f, sd) + hull_routing + n_rows * sd * sketch_read
                        out[head + (f_cells, "single", sd, 0)] = (needed, routing, _adaptive_budgets(sorted_est))
                    s1, s2 = TWO_PASS
                    est1 = est_by_dims[s1][:n_rows]
                    est2_all = est_by_dims[s2][:n_rows]
                    for keep in TWO_PASS_KEEP:
                        p = min(keep, n_rows)
                        top = np.argpartition(est1, p - 1)[:p] if p < n_rows else np.arange(n_rows)
                        in_top = np.zeros(n_rows, dtype=bool)
                        in_top[top] = True
                        sorted_top = np.sort(est2_all[top])
                        sorted_rest = np.sort(est1[~in_top])
                        needed = base_needed.copy()
                        own_top = in_top[own_idx]
                        idx_s = np.flatnonzero(sketched)
                        needed[idx_s[own_top]] = np.searchsorted(sorted_top, est2_all[own_idx[own_top]], side="right")
                        needed[idx_s[~own_top]] = p + np.searchsorted(sorted_rest, est1[own_idx[~own_top]], side="right")
                        routing = (max(r_c, r_f, s2) + hull_routing + n_rows * s1 * sketch_read
                                   + p * (s2 - s1) * sketch_read)
                        out[head + (f_cells, "two", s1, keep)] = (needed, routing, _adaptive_budgets(sorted_top))
    return out


def _evaluate_all(query_ids) -> dict:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        per_query = list(pool.map(evaluate_query, query_ids, chunksize=4))
    return {
        c: tuple(np.stack([pq[c][i] for pq in per_query]) if i != 1 else np.array([pq[c][1] for pq in per_query])
                 for i in range(3))
        for c in per_query[0]
    }


def fixed_at_recall(needed: np.ndarray, routing: np.ndarray, target: float) -> tuple[float, float]:
    flat = np.nan_to_num(needed.ravel(), nan=np.inf)
    vectors = float(np.quantile(flat, target, method="inverted_cdf"))
    return vectors, vectors + float(routing.mean())


def adaptive_points(needed, routing, budgets) -> list[tuple[int, float, float, float]]:
    """(c index, recall, vectors, total) per adaptive threshold."""
    if np.isnan(budgets).all():
        return []
    found = np.nan_to_num(needed, nan=np.inf)
    return [
        (ci, float((found <= budgets[:, ci : ci + 1]).mean()), float(budgets[:, ci].mean()),
         float(budgets[:, ci].mean() + routing.mean()))
        for ci in range(budgets.shape[1])
    ]


def operating_cost(result, target: float, mode) -> tuple[float, float, float] | None:
    """(vectors, total, recall) for a stopping mode: 'fixed' or an adaptive c index."""
    needed, routing, budgets = result
    if mode == "fixed":
        v, tot = fixed_at_recall(needed, routing, target)
        return v, tot, target
    _, rec, v, tot = adaptive_points(needed, routing, budgets)[mode]
    return (v, tot, rec) if rec >= target else None


def best_choice(results: dict, target: float, keep) -> tuple:
    best, best_cost = None, np.inf
    for c, res in results.items():
        if not keep(c):
            continue
        modes = ["fixed"] + ([ci for ci, rec, _, _ in adaptive_points(*res) if rec >= target] if keep(c, "adaptive") else [])
        for mode in modes:
            cost = operating_cost(res, target, mode)
            if cost and cost[1] < best_cost:
                best, best_cost = (c, mode), cost[1]
    return best


def _baselines(index: TwoLevel, queries, gt, fine_min, rows) -> dict:
    """Flat IVF over coarse cells (full-dim centroids) and the fine-cell oracle, fixed budget."""
    coarse_sizes, fine_sizes = np.diff(index.coarse_offsets), np.diff(index.fine_offsets)
    out = {}
    for name, scores, row_unit, sizes, routing in (
        ("flat IVF coarse (full-dim centroids)", sq_dists(queries[rows], index.coarse_centroids), _STATE["row_coarse"], coarse_sizes, len(coarse_sizes)),
        ("oracle fine cells (perfect hull)", fine_min[rows], _STATE["row_fine"], fine_sizes, 0.0),
    ):
        order = np.argsort(scores, axis=1)
        cum = np.cumsum(sizes[order], axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(order.shape[1])[None].repeat(len(order), 0), 1)
        unit = row_unit[gt[rows]]
        needed = np.take_along_axis(cum, np.take_along_axis(rank, unit, 1), 1).astype(float)
        out[name] = (needed, np.full(len(rows), routing, dtype=float))
    return out


def _label(c, mode) -> str:
    r_c, t, r_f, f_bits, f, kind, s, keep = c
    head = f"rc={r_c} T={t} fine={r_f}d/{f_bits}b"
    sketch = {"none": " no sketch", "single": f" F={f} s={s}", "two": f" F={f} 32→128 P={keep}"}[kind]
    stop = "" if mode == "fixed" else f" adapt c={ADAPTIVE_C[mode]}"
    return head + sketch + stop


VARIANTS = {
    # name: keep(config, mode="fixed") → bool
    "v2: float fine hull, s=128 4-bit, fixed": lambda c, mode="fixed": mode == "fixed" and c[0] == 128 and c[2:4] == (128, 32) and c[5] in ("none", "single") and c[6] in (0, 128),
    "+ cheap fine hull (dims/bits)": lambda c, mode="fixed": mode == "fixed" and c[5] in ("none", "single") and c[6] in (0, 128),
    "+ small sketch (s=32/64)": lambda c, mode="fixed": mode == "fixed" and c[5] in ("none", "single"),
    "+ two-pass sketch": lambda c, mode="fixed": mode == "fixed",
    "+ adaptive stopping (best overall)": lambda c, mode="fixed": True,
}


def main() -> None:
    n_coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    t0 = time.perf_counter()
    d = load_dbpedia(n_queries=n_queries)
    index = build_two_level(np.asarray(d.base, dtype=np.float32), n_coarse, m)
    space = build_space(index)
    gt, _, fine_min = ground_truth(index, d.queries)
    _STATE.update(
        index=index, space=space, queries=d.queries, gt=gt,
        row_fine=np.repeat(np.arange(n_coarse * m), np.diff(index.fine_offsets)),
        row_coarse=np.repeat(np.arange(n_coarse), np.diff(index.coarse_offsets)),
        fine_sizes=np.diff(index.fine_offsets), coarse_sizes=np.diff(index.coarse_offsets),
    )
    print(f"DBpedia-1536 N={len(index.vectors)}: {n_coarse}x{m} cells, PCA-{PCA_DIMS}, "
          f"{n_queries // 2} train / {n_queries - n_queries // 2} test queries, setup {time.perf_counter() - t0:.0f}s", flush=True)

    train, test = np.arange(n_queries // 2), np.arange(n_queries // 2, n_queries)
    t = time.perf_counter()
    res_train, res_test = _evaluate_all(train), _evaluate_all(test)
    print(f"  evaluated {len(res_test)} configs in {time.perf_counter() - t:.0f}s", flush=True)
    base_test = _baselines(index, d.queries, gt, fine_min, test)

    print("\n  test queries: vectors scanned / routing / total (recall reached); configs tuned on train")
    for target in TARGETS:
        print(f"\n  R>={target}")
        for name, (needed, routing) in base_test.items():
            v, tot = fixed_at_recall(needed, routing, target)
            print(f"    {name:<38}{'':<60}{v:7.0f} {routing.mean():6.0f} {tot:7.0f}")
        for name, keep in VARIANTS.items():
            choice = best_choice(res_train, target, keep)
            if choice is None:
                print(f"    {name:<38}not reached")
                continue
            c, mode = choice
            res = res_test[c]
            if mode == "fixed":
                v, tot = fixed_at_recall(res[0], res[1], target)
                rec = target
            else:
                _, rec, v, tot = adaptive_points(*res)[mode]
            print(f"    {name:<38}{_label(c, mode):<60}{v:7.0f} {res[1].mean():6.0f} {tot:7.0f}  (R={rec:.3f})", flush=True)


if __name__ == "__main__":
    main()
