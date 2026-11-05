"""Issue #26 recall check: does a per-coordinate cap of 10/12 bits recover flat-scan recall at 8 bits/dim?

Simulates TieredQ full mode without tiers: PCA ranks are centred, scaled by sigma_j and coded with Gaussian
Lloyd-Max books (<= 8 bits) or the closed-form Gaussian compander (> 8 bits, see eigen_bound.py). Widths come
from greedy allocation on variance^power, TieredQ's default power per metric unless a variant overrides it.
Scores are asymmetric: float queries against reconstructed vectors, exhaustive scan, compared with exact truth.

Baseline "rot+lloyd": random rotation, per-vector norm scaling, uniform 8-bit Lloyd-Max codes (TurboQuant-like).

Usage: python recall_sim.py [--bits-per-dim 8] [--l2 NAME]... NAME=PATH[:ARG] ...   (PATH/ARG as in eigen_bound.py)
"""

import argparse
import sys

import numpy as np

from eigen_bound import (LLOYD_MAX_WIDEST_BITS, WIDEST_BITS_MODELLED, compander_levels, distortion_tables,
                         greedy_allocation, load, lloyd_max_levels, pca)

DEFAULT_POWER = {"ip": 1.5, "l2": 1.0}
# (label, per-coordinate cap, allocation power or None for the metric default)
VARIANTS = (("cap 8", 8, None), ("cap 10", 10, None), ("cap 12", 12, None), ("cap 12 p2", 12, 2.0))
RECALL_DEPTHS = (10, 100)
ERROR_SAMPLE_ROWS = 20000
ENCODE_CHUNK_ROWS = 65536
SCAN_CHUNK_QUERIES = 100
ROTATION_SEED = 26


def code_books():
    books = {b: lloyd_max_levels(b) for b in range(1, LLOYD_MAX_WIDEST_BITS + 1)}
    books.update({b: compander_levels(b) for b in range(LLOYD_MAX_WIDEST_BITS + 1, WIDEST_BITS_MODELLED + 1)})
    return books


def quantize_columns(z, bits, books):
    """Nearest-level reconstruction of standardized columns z[:, j] with a bits[j]-wide book."""
    out = np.zeros_like(z)
    for width in np.unique(bits[bits > 0]):
        levels = books[width].astype(z.dtype)
        cols = bits == width
        edges = (levels[:-1] + levels[1:]) / 2
        out[:, cols] = levels[np.searchsorted(edges, z[:, cols])]
    return out


def pca_reconstruct(base, mean, sigma, basis, bits, books):
    mean, sigma, basis = mean.astype(np.float32), sigma.astype(np.float32), basis.astype(np.float32)
    safe_sigma = np.where(sigma > 0, sigma, 1)
    out = np.empty_like(base)
    for start in range(0, len(base), ENCODE_CHUNK_ROWS):
        z = ((base[start:start + ENCODE_CHUNK_ROWS] - mean) @ basis) / safe_sigma
        out[start:start + ENCODE_CHUNK_ROWS] = (quantize_columns(z, bits, books) * sigma) @ basis.T + mean
    return out


def rotation_reconstruct(base, bits_per_dim, books):
    dims = base.shape[1]
    rotation, _ = np.linalg.qr(np.random.default_rng(ROTATION_SEED).standard_normal((dims, dims)))
    rotation = rotation.astype(np.float32)
    uniform_bits = np.full(dims, bits_per_dim)
    out = np.empty_like(base)
    for start in range(0, len(base), ENCODE_CHUNK_ROWS):
        y = base[start:start + ENCODE_CHUNK_ROWS] @ rotation
        scale = np.linalg.norm(y, axis=1, keepdims=True) / np.sqrt(dims)
        out[start:start + ENCODE_CHUNK_ROWS] = (quantize_columns(y / scale, uniform_bits, books) * scale) @ rotation.T
    return out


def scores(queries, base, metric):
    ip = queries @ base.T
    return ip if metric == "ip" else 2 * ip - np.sum(base * base, axis=1)


def top_ids(queries, base, metric, depth):
    ids = []
    for start in range(0, len(queries), SCAN_CHUNK_QUERIES):
        s = scores(queries[start:start + SCAN_CHUNK_QUERIES], base, metric)
        part = np.argpartition(-s, depth, axis=1)[:, :depth]
        order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
        ids.append(np.take_along_axis(part, order, axis=1))
    return np.concatenate(ids)


def recall(approx, exact, depth):
    hits = [len(np.intersect1d(a[:depth], e[:depth], assume_unique=True)) for a, e in zip(approx, exact)]
    return float(np.mean(hits)) / depth


def score_error(queries, base, reconstructed, metric):
    """RMS score error relative to the spread of exact scores, on a fixed base sample."""
    rows = slice(0, min(ERROR_SAMPLE_ROWS, len(base)))
    exact = scores(queries, base[rows], metric)
    return float(np.sqrt(np.mean((scores(queries, reconstructed[rows], metric) - exact) ** 2)) / np.std(exact))


def evaluate(name, base, queries, metric, bits_per_dim, tables, books):
    base, queries = np.ascontiguousarray(base, np.float32), np.ascontiguousarray(queries, np.float32)
    dims = base.shape[1]
    exact = top_ids(queries, base, metric, max(RECALL_DEPTHS))
    mean, sigma2, basis = pca(base)
    candidates = [("rot+lloyd", None, lambda: rotation_reconstruct(base, bits_per_dim, books))]
    for label, cap, power in VARIANTS:
        weights = sigma2 ** (power or DEFAULT_POWER[metric])
        bits = greedy_allocation(weights, tables["tieredq_compand"], bits_per_dim * dims, cap)
        candidates.append((label, bits, lambda b=bits: pca_reconstruct(base, mean, np.sqrt(sigma2), basis, b, books)))

    print(f"\n## {name}  (n={len(base)}, D={dims}, metric={metric}, {bits_per_dim} bits/dim, {len(queries)} queries)")
    header = "| codec | " + " | ".join(f"recall@{k}" for k in RECALL_DEPTHS) + " | rel. score RMS | vs rot+lloyd | widest | ranks >8 |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    reference_error = None
    for label, bits, reconstruct in candidates:
        reconstructed = reconstruct()
        approx = top_ids(queries, reconstructed, metric, max(RECALL_DEPTHS))
        error = score_error(queries, base, reconstructed, metric)
        reference_error = reference_error or error
        widths = "- | -" if bits is None else f"{bits.max()} | {int(np.sum(bits > LLOYD_MAX_WIDEST_BITS))}"
        cells = " | ".join(f"{recall(approx, exact, k):.5f}" for k in RECALL_DEPTHS)
        gain_db = 20 * np.log10(reference_error / error)
        print(f"| {label} | {cells} | {error:.3e} | {gain_db:+.2f} dB | {widths} |")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits-per-dim", type=int, default=8)
    parser.add_argument("--l2", action="append", default=[], help="dataset name scored by L2 distance (repeatable)")
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    tables, books = distortion_tables(), code_books()
    for spec in args.datasets:
        name, base, queries = load(spec)
        evaluate(name, base, queries, "l2" if name in args.l2 else "ip", args.bits_per_dim, tables, books)


if __name__ == "__main__":
    main()
