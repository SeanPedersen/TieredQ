"""Issue #13: 7-bit additive book with parts of 4, 2 and 1 bits (no nibble padding), by jittered L-BFGS from the 4+3 book.

Usage: fit_seven.py BEST.json OUT.json [workers]
"""
import json
import multiprocessing
import sys

import numpy as np
from scipy.optimize import minimize

from additive_mse import POLISH_ITERATIONS, distortion

PART_BITS = (4, 2, 1)
RESTARTS = 96
JITTER = 0.08


def sums(parts):
    total = parts[0]
    for part in parts[1:]:
        total = np.add.outer(total, part)
    return total.ravel()


def unpack(x):
    parts, at = [], 0
    for bits in PART_BITS:
        parts.append(x[at:at + (1 << bits)])
        at += 1 << bits
    return parts


def attempt(job):
    seed, x = job
    rng = np.random.default_rng(seed)
    start = x + (rng.normal(0, JITTER, x.size) * np.abs(x).mean() if seed else 0)
    result = minimize(lambda v: distortion(sums(unpack(v))), start, method="L-BFGS-B", options={"maxiter": 4 * POLISH_ITERATIONS})
    return result.fun, result.x.tolist()


def main(best_path, out_path, workers="8"):
    book = json.load(open(best_path))["7"]
    a, b = np.array(book["a"]), np.array(book["b"])
    if a.size != 1 << PART_BITS[0]:
        a, b = b, a
    b = np.sort(b)
    mid = np.array([b[:2].mean(), b[2:4].mean(), b[4:6].mean(), b[6:].mean()])
    single = np.array([0.0, (b[1::2] - b[0::2]).mean()])
    x = np.concatenate((a, mid, single))
    with multiprocessing.Pool(int(workers)) as pool:
        results = pool.map(attempt, [(seed, x) for seed in range(RESTARTS)])
    value, solution = min(results, key=lambda item: item[0])
    parts = unpack(np.array(solution))
    print(f"7 bits 4+2+1 distortion {value:.7f} lloyd {book['lloyd']:.7f} ratio {value / book['lloyd']:.3f}")
    json.dump({"part_bits": PART_BITS, "parts": [p.tolist() for p in parts], "distortion": value, "lloyd": book["lloyd"]}, open(out_path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
