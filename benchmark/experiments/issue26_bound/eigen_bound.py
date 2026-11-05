"""Issue #26 bound: how much distortion does a per-coordinate cap above 8 bits buy at 6-8 bits/dim?

Model: PCA ranks are independent Gaussians with variance sigma2_j. A rank coded with b bits has distortion
sigma2_j * d(b), where d is the exact unit-Gaussian distortion of a Lloyd-Max book for b <= 8 (TieredQ's
stored books). For b > 8 two table-free wide quantizers are modelled:
  uniform : optimally loaded uniform grid (the quantizer #26 proposes)
  compand : Panter-Dite Gaussian compander, levels sqrt(3) * Phi^-1((i + 1/2) / 2^b) (closed form, no stored book)
Bits are spread by greedy marginal allocation (optimal for convex d) under each per-coordinate cap.

Objectives:
  mse : sum_j sigma2_j d(b_j)                     reconstruction error
  ip  : sum_j E[q_j^2] sigma2_j d(b_j)            inner-product error for the held-out query distribution

Baselines at the same mean budget: a random rotation makes every coordinate's variance trace/D, so
uniform bits are optimal there; "rot+lloyd" approximates TurboQuant, "rot+uniform" a plain RaBitQ-style grid
(without RaBitQ's per-vector rescaling). Numbers are an upper bound on what wider codes can give TieredQ
(Gaussian marginals, per-rank widths, no tier grouping).

Usage: python eigen_bound.py NAME=PATH[:ARG] ...
  PATH a .npz with a base array; ARG names its query array (default "queries")
  PATH a dir with base.f32/queries.f32; ARG is the dimension
"""

import heapq
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfinv, ndtr

LLOYD_MAX_WIDEST_BITS = 8
WIDEST_BITS_MODELLED = 16
BUDGETS = (4, 6, 7, 8)
# (column label, wide-quantizer table, per-coordinate cap); the last entry also reports its allocation shape.
VARIANTS = (("cap 8", "compand", 8), ("uni 10", "uniform", 10), ("uni 12", "uniform", 12),
            ("cmp 10", "compand", 10), ("cmp 12", "compand", 12), ("cmp 16", "compand", 16))
WIDTH_THRESHOLDS = (8, 10, 12)
LLOYD_ITERATIONS = 3000
GAUSSIAN_LOADING_SIGMAS = 4.5
STEP_SEARCH_FACTOR = 4
COVARIANCE_CHUNK_ROWS = 65536
DB_PER_BIT = 20 * np.log10(2)


def _gaussian_cell_moments(edges):
    """Zeroth, first and second moments of N(0, 1) over cells [edges[i], edges[i+1])."""
    finite = np.where(np.isinf(edges), 0.0, edges)
    density = np.exp(-0.5 * finite**2) / np.sqrt(2 * np.pi) * np.isfinite(edges)
    m0 = np.diff(ndtr(edges))
    m1 = density[:-1] - density[1:]
    m2 = m0 + finite[:-1] * density[:-1] - finite[1:] * density[1:]
    return m0, m1, m2


def _nearest_level_edges(levels):
    return np.concatenate([[-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]])


def nearest_level_distortion(levels):
    m0, m1, m2 = _gaussian_cell_moments(_nearest_level_edges(levels))
    return float(np.sum(m2 - 2 * levels * m1 + levels**2 * m0))


def compander_levels(bits):
    count = 1 << bits
    return np.sqrt(3) * np.sqrt(2) * erfinv(2 * (np.arange(count) + 0.5) / count - 1)


def lloyd_max_levels(bits):
    levels = compander_levels(bits)
    for _ in range(LLOYD_ITERATIONS):
        m0, m1, _ = _gaussian_cell_moments(_nearest_level_edges(levels))
        levels = m1 / m0
    return levels


def lloyd_max_distortion(bits):
    return nearest_level_distortion(lloyd_max_levels(bits))


def uniform_distortion(bits):
    count = 1 << bits
    offsets = np.arange(count) - (count - 1) / 2
    guess = 2 * GAUSSIAN_LOADING_SIGMAS / count
    bounds = (guess / STEP_SEARCH_FACTOR, guess * STEP_SEARCH_FACTOR)
    return minimize_scalar(lambda step: nearest_level_distortion(offsets * step), bounds=bounds, method="bounded").fun


def distortion_tables():
    """Unit-Gaussian distortion per width 0..WIDEST_BITS_MODELLED; TieredQ tables keep Lloyd-Max up to 8 bits."""
    wide = range(1, WIDEST_BITS_MODELLED + 1)
    lloyd = np.array([1.0] + [lloyd_max_distortion(b) for b in range(1, LLOYD_MAX_WIDEST_BITS + 1)])
    uniform = np.array([1.0] + [uniform_distortion(b) for b in wide])
    compand = np.array([1.0] + [nearest_level_distortion(compander_levels(b)) for b in wide])
    above_lloyd = slice(LLOYD_MAX_WIDEST_BITS + 1, None)
    return {"lloyd": lloyd, "uniform": uniform, "compand": compand,
            "tieredq_uniform": np.concatenate([lloyd, uniform[above_lloyd]]),
            "tieredq_compand": np.concatenate([lloyd, compand[above_lloyd]])}


def greedy_allocation(weights, table, total_bits, cap):
    bits = np.zeros(len(weights), dtype=int)
    heap = [(-(w * (table[0] - table[1])), j) for j, w in enumerate(weights)]
    heapq.heapify(heap)
    for _ in range(total_bits):
        _, j = heapq.heappop(heap)
        bits[j] += 1
        if bits[j] < cap:
            heapq.heappush(heap, (-(weights[j] * (table[bits[j]] - table[bits[j] + 1])), j))
    return bits


def load(spec):
    name, _, rest = spec.partition("=")
    path, _, arg = rest.partition(":")
    path = Path(path).expanduser()
    if path.suffix == ".npz":
        with np.load(path) as data:
            return name, data["base"], data[arg or "queries"]
    d = int(arg)
    base = np.fromfile(path / "base.f32", dtype=np.float32).reshape(-1, d)
    queries = np.fromfile(path / "queries.f32", dtype=np.float32).reshape(-1, d)
    return name, base, queries


def pca(base):
    """Mean, descending eigenvalues and matching eigenvector columns of the base covariance."""
    mean = base.mean(axis=0, dtype=np.float64)
    gram = np.zeros((base.shape[1],) * 2)
    for start in range(0, len(base), COVARIANCE_CHUNK_ROWS):
        chunk = base[start:start + COVARIANCE_CHUNK_ROWS].astype(np.float64) - mean
        gram += chunk.T @ chunk
    sigma2, basis = np.linalg.eigh(gram / len(base))
    order = np.argsort(sigma2)[::-1]
    return mean, np.clip(sigma2[order], 0, None), basis[:, order]


def pca_spectrum(base, queries):
    _, sigma2, basis = pca(base)
    query_moment = np.mean((queries.astype(np.float64) @ basis) ** 2, axis=0)
    return sigma2, query_moment


def _db_cell(gain_db):
    return f"{gain_db:+.2f} ({gain_db / DB_PER_BIT:+.2f})"


def report(name, n, sigma2, query_moment, tables):
    dims = len(sigma2)
    trace = sigma2.sum()
    positive = sigma2[sigma2 > 0]
    am_gm_db = 10 * np.log10(positive.mean() / np.exp(np.mean(np.log(positive))))
    print(f"\n## {name}  (n={n}, D={dims}, AM/GM of eigenvalues = {am_gm_db:.1f} dB = {am_gm_db / DB_PER_BIT:.2f} bits)")
    objectives = {"mse": sigma2, "ip": query_moment * sigma2}
    rotation_scale = {"mse": trace, "ip": query_moment.sum() * trace / dims}
    widths = " / ".join(f">{t}" for t in WIDTH_THRESHOLDS)
    for objective, weights in objectives.items():
        print(f"\n### {objective}: gain over rot+lloyd in dB (+ = better), equivalent bits/dim in ()")
        header = ("| bits/dim | rot+uniform | " + " | ".join(label for label, _, _ in VARIANTS)
                  + f" | widest ({VARIANTS[-1][0]}) | ranks {widths} |")
        print(header)
        print("|" + "---|" * (header.count("|") - 1))
        for budget in BUDGETS:
            reference = rotation_scale[objective] * tables["lloyd"][budget]
            cells = [10 * np.log10(tables["lloyd"][budget] / tables["uniform"][budget])]
            allocations = []
            for _, wide, cap in VARIANTS:
                table = tables[f"tieredq_{wide}"]
                bits = greedy_allocation(weights, table, budget * dims, cap)
                cells.append(10 * np.log10(reference / np.sum(weights * table[bits])))
                allocations.append(bits)
            shape = allocations[-1]
            over = " / ".join(str(int(np.sum(shape > t))) for t in WIDTH_THRESHOLDS)
            print(f"| {budget} | " + " | ".join(map(_db_cell, cells)) + f" | {shape.max()} | {over} |")


def main():
    tables = distortion_tables()
    for key in ("lloyd", "uniform", "compand"):
        print(f"unit-Gaussian distortion {key:>8} b=1..: " + " ".join(f"{v:.3e}" for v in tables[key][1:]))
    for spec in sys.argv[1:]:
        name, base, queries = load(spec)
        sigma2, query_moment = pca_spectrum(base, queries)
        report(name, len(base), sigma2, query_moment, tables)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
