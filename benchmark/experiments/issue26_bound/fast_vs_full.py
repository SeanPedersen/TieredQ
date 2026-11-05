"""Issue #26: how much recall does fast mode's representation lose against full mode at the same budget?

Three scorers per dataset on the unmodified codec, flat scan, exact float32 truth:
  full        full-target model (Lloyd-Max books), exact fp32 scorer
  full+cache  the same model through its auxiliary fast cache (build_fast_scan / search_fast)
  fast        fast-target model: additive wide books scored through int8 nibble tables

Usage: python fast_vs_full.py [--bits-per-dim 8] [--l2 NAME]... NAME=PATH[:ARG] ...
"""

import argparse
import sys
import time

import numpy as np

from eigen_bound import load
from recall_sim import RECALL_DEPTHS, recall, top_ids


def scorers(dims, budget, metric):
    import tieredq

    def full(base):
        index = tieredq.TieredQIndex(dims, budget, mode="full", metric=metric)
        index.add(base)
        return index.search

    def full_cache(base):
        index = tieredq.TieredQIndex(dims, budget, mode="full", metric=metric)
        index.add(base)
        index.build_fast_scan()
        return index.search_fast

    def fast(base):
        index = tieredq.TieredQIndex(dims, budget, mode="fast", metric=metric)
        index.add(base)
        return index.search

    return {"full": full, "full+cache": full_cache, "fast": fast}


def evaluate(name, base, queries, metric, bits_per_dim):
    base, queries = np.ascontiguousarray(base, np.float32), np.ascontiguousarray(queries, np.float32)
    dims, depth = base.shape[1], max(RECALL_DEPTHS)
    exact = top_ids(queries, base, metric, depth)
    print(f"\n## {name}  (n={len(base)}, D={dims}, metric={metric}, {bits_per_dim} bits/dim, {len(queries)} queries)")
    print("| scorer | " + " | ".join(f"recall@{k}" for k in RECALL_DEPTHS) + " | missed@10 | search s |")
    print("|---|" + "---|" * (len(RECALL_DEPTHS) + 2))
    for label, build in scorers(dims, bits_per_dim * dims, metric).items():
        search = build(base)
        start = time.perf_counter()
        _, ids = search(queries, depth)
        searched = time.perf_counter() - start
        cells = " | ".join(f"{recall(ids, exact, k):.5f}" for k in RECALL_DEPTHS)
        missed = round((1 - recall(ids, exact, RECALL_DEPTHS[0])) * len(queries) * RECALL_DEPTHS[0])
        print(f"| {label} | {cells} | {missed} | {searched:.1f} |")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits-per-dim", type=int, default=8)
    parser.add_argument("--l2", action="append", default=[], help="dataset name scored by L2 distance (repeatable)")
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    for spec in args.datasets:
        name, base, queries = load(spec)
        evaluate(name, base, queries, "l2" if name in args.l2 else "ip", args.bits_per_dim)


if __name__ == "__main__":
    main()
