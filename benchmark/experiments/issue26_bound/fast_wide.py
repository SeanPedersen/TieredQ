"""Issue #26: flat fast mode with fine tiers (up to 12 bits over a scan prefix at the automatic cap) at the same B*D
code budget (real codec and kernels).

Variants: "full" is full mode (exact FP32 scoring of Lloyd-Max codes, no fine tiers), "exact" a fast index over a
full-target (Lloyd-Max) model with exact_fast_scores, "off" is today's fast mode
(automatic cap max(6, ceil(B)) for inner product), "auto" lets training decide (policy C: fine tiers when the allocator
predicts a clear weighted-error drop), "on" / "on:N" forces fine tiers up to 12 / N bits; a "/pW" suffix sets the scan
cap (prefix width) to W instead of the automatic cap.
Fine-tier indexes scan each code's prefix with the unchanged nibble scan to depth max(32, 1.5k) and rescore those
candidates exactly from prefix plus fine bits.

Seed s > 0 permutes the base rows with that seed before indexing (the bindings fix the training seed, so this
varies the training sample and k-means initialisation); hits are mapped back to source ids.

Per variant: the scan prefix width of fine-tier models (0: none), add() seconds and rows/s (training + encoding
+ fast storage), realised code bits, bytes/vector (scanned storage + fine bits + metadata), resident bytes/vector
(storage_bytes / n), recall@k with missed top-k hits, median ms/query of a batch search of all queries, and median
single-query latency (one search call per query, first LATENCY_QUERIES queries).

Usage: python fast_wide.py NAME=PATH[:ARG]... [--bits-per-dim 8] [--variants off auto on:10 on:12] [--k 10 100]
       [--seeds 0 1 2]  (datasets first: --variants takes every following word)
"""

import argparse
import time

import numpy as np
import tieredq

from eigen_bound import load
from recall_sim import top_ids

WIDEST_FINE_BITS = 12
FULL_VARIANT = "full"
EXACT_VARIANT = "exact"
TIMING_REPEATS = 3
LATENCY_QUERIES = 200


def misses(ids, exact, k):
    hits = sum(len(np.intersect1d(row[:k], truth[:k])) for row, truth in zip(ids, exact))
    return exact.shape[0] * k - hits


def timed(index, queries, k):
    index.search(queries[:1], k)  # warm-up
    elapsed = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        elapsed.append(time.perf_counter() - start)
    return np.median(elapsed) * 1e3 / len(queries), ids.astype(np.int64)


def single_latency_ms(index, queries, k):
    elapsed = []
    for query in queries[:LATENCY_QUERIES]:
        start = time.perf_counter()
        index.search(query[None, :], k)
        elapsed.append(time.perf_counter() - start)
    return np.median(elapsed) * 1e3


def fine_options(variant):
    """MODE[:MAX_TIER_BITS][/pPREFIX]: fine_tiers mode, fine cap and an explicit scan cap (prefix width)."""
    head, _, prefix = variant.partition("/p")
    mode, _, bits = head.partition(":")
    return {"fine_tiers": mode, "max_tier_bits": int(bits) if bits else WIDEST_FINE_BITS, "max_scan_bits": int(prefix or 0)}


def index_options(variant):
    """"full": full mode; "exact": fast index with exact_fast_scores; every other variant is a fast fine-tier setting."""
    if variant == FULL_VARIANT:
        return {"mode": "full", "fine_tiers": "off"}
    if variant == EXACT_VARIANT:
        return {"mode": "fast", "exact_fast_scores": True, "fine_tiers": "off"}
    return {"mode": "fast", **fine_options(variant)}


def build(base, bits_per_dim, variant):
    index = tieredq.TieredQIndex(base.shape[1], bits_per_dim * base.shape[1], metric="ip", **index_options(variant))
    start = time.perf_counter()
    index.add(base)
    return index, time.perf_counter() - start


def evaluate(name, base, queries, bits_per_dim, variants, depths, seeds):
    base, queries = np.ascontiguousarray(base, np.float32), np.ascontiguousarray(queries, np.float32)
    dims = base.shape[1]
    exact = top_ids(queries, base, "ip", max(depths))
    print(f"\n## {name}  (n={len(base)}, D={dims}, {bits_per_dim} bits/dim = {bits_per_dim * dims} code bits, {len(queries)} queries; "
          f"misses out of {len(queries)} * k; ms/query median of {TIMING_REPEATS})")
    cells = " | ".join(f"recall@{k} (misses) | batch ms/q@{k} | latency ms@{k}" for k in depths)
    print(f"| seed | variant | fine prefix | add s (rows/s) | code bits | bytes/vector | resident bytes/vector | {cells} |")
    print("|---|---|---|---|---|---|---|" + "---|---|---|" * len(depths), flush=True)
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(base)) if seed else np.arange(len(base))
        rows = base[order] if seed else base
        for variant in variants:
            index, seconds = build(rows, bits_per_dim, variant)
            row = []
            for k in depths:
                ms, ids = timed(index, queries, k)
                missed = misses(order[ids], exact, k)
                row += [f"{1 - missed / (len(queries) * k):.4f} ({missed})", f"{ms:.3f}", f"{single_latency_ms(index, queries, k):.3f}"]
            print(f"| {seed} | {variant} | {index.fast_prefix_bits} | {seconds:.1f} ({len(rows) / seconds:,.0f}) | {index.code_bits} | "
                  f"{index.bytes_per_vector} | {index.storage_bytes / len(rows):.1f} | " + " | ".join(row) + " |", flush=True)
            del index
        del rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits-per-dim", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=["off", "auto", "on:10", "on:12"])
    parser.add_argument("--k", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    for spec in args.datasets:
        name, base, queries = load(spec)
        evaluate(name, base, queries, args.bits_per_dim, args.variants, args.k, args.seeds)


if __name__ == "__main__":
    main()
