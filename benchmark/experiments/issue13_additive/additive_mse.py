"""Issue #13 feasibility: Gaussian MSE of additive codebooks level = a[hi] + b[lo] against Lloyd-Max at 5-8 bits.

"free" lets sums interleave, which needs a code permutation (as the 3-bit additive book does): alternating nearest-level
cells with exact Gaussian least squares, then L-BFGS. "nested" keeps every lo run inside one hi cell, so codes stay
sorted: L-BFGS on positive gaps, which encode that order.
Writes the best book per width and form as JSON for the recall experiment. Usage: out.json [seed] [restarts] [forms]
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc

MAX_PART_BITS = 4
WIDTHS = (3, 5, 6, 7, 8)
DEFAULT_RESTARTS = 12
ITERATIONS = 2000
CONVERGED = 1e-13
INIT_NOISE = 0.15
LLOYD_ITERATIONS = 20000
MIN_GAP = 1e-6
POLISH_ITERATIONS = 500
DEFAULT_SEED = 0
FORMS = "nested,free"


def pdf(x):
    return np.where(np.isfinite(x), np.exp(-np.minimum(x * x, 1e4) / 2) / np.sqrt(2 * np.pi), 0.0)


def cdf(x):
    return 0.5 * erfc(-x / np.sqrt(2.0))


def cells(levels):
    """Mass and first moment of each sorted level's Voronoi cell."""
    edges = np.concatenate(([-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]))
    return cdf(edges[1:]) - cdf(edges[:-1]), pdf(edges[:-1]) - pdf(edges[1:])


def distortion(levels):
    levels = np.sort(levels)
    edges = np.concatenate(([-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]))
    xpdf = np.zeros_like(edges)
    xpdf[1:-1] = edges[1:-1] * pdf(edges[1:-1])
    mass, first = cells(levels)
    return float(np.sum(mass + xpdf[:-1] - xpdf[1:] - 2 * levels * first + levels * levels * mass))


def lloyd_max(bits):
    levels = np.linspace(-3, 3, 1 << bits)
    for _ in range(LLOYD_ITERATIONS):
        mass, first = cells(levels)
        levels = first / mass
    return levels


def fit_free(hi_bits, lo_bits, start_a, start_b):
    """Weighted least squares on (a, b) with cells fixed until the distortion stops improving, then a joint polish."""
    a, b = start_a, start_b
    best = np.inf
    hi_count, lo_count = 1 << hi_bits, 1 << lo_bits
    cell = np.arange(hi_count * lo_count)
    design = np.zeros((cell.size, hi_count + lo_count))
    design[cell, cell // lo_count] = 1
    design[cell, hi_count + cell % lo_count] = 1
    for _ in range(ITERATIONS):
        sums = (a[:, None] + b[None, :]).ravel()
        order = np.argsort(sums, kind="stable")
        mass_sorted, first_sorted = cells(sums[order])
        mass, first = np.empty_like(sums), np.empty_like(sums)
        mass[order], first[order] = mass_sorted, first_sorted
        weight = np.sqrt(np.maximum(mass, 1e-300))
        target = first / np.maximum(mass, 1e-300)
        solution = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
        a, b = solution[:hi_count], solution[hi_count:]
        value = distortion((a[:, None] + b[None, :]).ravel())
        if value >= best - CONVERGED:
            break
        best = value
    result = minimize(lambda x: distortion((x[:hi_count, None] + x[hi_count:][None, :]).ravel()),
                      np.concatenate((a, b)), method="L-BFGS-B", options={"maxiter": POLISH_ITERATIONS})
    return result.fun, (result.x[:hi_count], result.x[hi_count:])


def nested_pair(x, hi_count):
    """Decode (start, a gaps beyond the lo span, lo gaps) so that all sums increase with the code."""
    lo_gaps = np.exp(x[hi_count:])
    b = np.concatenate(([0.0], np.cumsum(lo_gaps)))
    a = x[0] + np.concatenate(([0.0], np.cumsum(b[-1] + MIN_GAP + np.exp(x[1:hi_count]))))
    return a, b


def fit_nested(hi_bits, lo_bits, start_a, start_b):
    hi_count = 1 << hi_bits
    span = np.ptp(start_b)
    lo_gaps = np.maximum(np.diff(np.sort(start_b)), MIN_GAP)
    hi_gaps = np.maximum(np.diff(np.sort(start_a)) - span, MIN_GAP)
    x = np.concatenate(([np.min(start_a) + np.min(start_b)], np.log(hi_gaps), np.log(lo_gaps)))
    result = minimize(lambda v: distortion(np.ravel(np.add.outer(*nested_pair(v, hi_count)))), x, method="L-BFGS-B",
                      options={"maxiter": POLISH_ITERATIONS})
    return result.fun, nested_pair(result.x, hi_count)


def starts(hi_bits, lo_bits, reference, rng, restarts):
    """Deterministic nested start from Lloyd-Max cells, then jittered ones with overlapping lo spans."""
    hi_count, lo_count = 1 << hi_bits, 1 << lo_bits
    grid = reference.reshape(hi_count, lo_count)
    a, b = grid.mean(axis=1), (grid - grid.mean(axis=1, keepdims=True)).mean(axis=0)
    yield a, b
    for _ in range(restarts):
        stretch = rng.uniform(0.5, 2.0)
        yield a + rng.normal(0, INIT_NOISE, a.size), b * stretch + rng.normal(0, INIT_NOISE / lo_count, b.size)


def best_split(bits, form, reference, rng, restarts):
    fit = fit_nested if form == "nested" else fit_free
    best = (np.inf, None, None, None)
    for lo_bits in range(1, min(bits, MAX_PART_BITS + 1)):
        hi_bits = bits - lo_bits
        if hi_bits > MAX_PART_BITS or hi_bits < 1:
            continue
        for a, b in starts(hi_bits, lo_bits, reference, rng, restarts):
            value, (fit_a, fit_b) = fit(hi_bits, lo_bits, a, b)
            if value < best[0]:
                best = (value, hi_bits, fit_a, fit_b)
    return best


def main(out_path, seed=DEFAULT_SEED, restarts=DEFAULT_RESTARTS, forms=FORMS):
    rng = np.random.default_rng(int(seed))
    restarts = int(restarts)
    books = {}
    print("bits\tlloyd\tform\tsplit\tadditive\tratio\tdB_loss\tuniform_ratio")
    for bits in WIDTHS:
        reference = lloyd_max(bits)
        lloyd = distortion(reference)
        uniform = min(distortion(np.linspace(-s, s, 1 << bits)) for s in np.linspace(2, 5, 3001))
        for form in forms.split(","):
            value, hi_bits, a, b = best_split(bits, form, reference, rng, restarts)
            print(f"{bits}\t{lloyd:.6f}\t{form}\t{hi_bits}+{bits - hi_bits}\t{value:.6f}\t{value / lloyd:.3f}\t"
                  f"{10 * np.log10(value / lloyd):.2f}\t{uniform / lloyd:.3f}", flush=True)
            books[f"{bits}-{form}"] = {"hi_bits": hi_bits, "a": a.tolist(), "b": b.tolist(),
                                       "levels": np.sort((a[:, None] + b[None, :]).ravel()).tolist(),
                                       "distortion": value, "lloyd": lloyd}
    with open(out_path, "w") as output:
        json.dump(books, output)


if __name__ == "__main__":
    main(*sys.argv[1:])
