"""Compare nested IVF suffix widths on a cached FP32 dataset with identical prefix candidates."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from tieredq import TieredQNestedIVFIndex

K = 10
LISTS = 256
PREFIX_BITS_PER_DIM = 2
REPEATS = 5
BATCH_SIZE = 32
CANDIDATE_DEPTHS = (100, 400)
PROBES = (32, LISTS)


def recall(ids, truth):
    return (ids[:, :, None] == truth[:, None, :]).any(axis=2).mean()


def search_batches(index, queries, probes, depth):
    return np.concatenate([
        index.search(queries[first:first + BATCH_SIZE], k=K, nprobe=probes, candidates=depth)[1]
        for first in range(0, len(queries), BATCH_SIZE)
    ])


def timed(call):
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def build(base, bits, fine):
    index, build_seconds = timed(lambda: TieredQNestedIVFIndex.build(
        base, prefix_bits=PREFIX_BITS_PER_DIM * base.shape[1], total_bits=bits * base.shape[1],
        lists=LISTS, fine_tiers=fine))
    tier, encode_seconds = timed(lambda: index.encode_tier(base))
    index.attach_tier(tier)
    return index, dict(build_seconds=build_seconds, encode_seconds=encode_seconds,
                       total_bits=index.total_bits, prefix_bits=index.prefix_bits,
                       max_refinement_bits=index.max_refinement_bits,
                       tier_bytes_per_vector=index.tier_bytes_per_vector,
                       ivf_bytes_per_vector=index.storage_bytes / len(base))


def measure_pair(base, queries, truth, bits):
    indexes, metadata = {}, {}
    for fine in (False, True):
        print(f"Building {bits} bits/dim fine_tiers={fine}", flush=True)
        indexes[fine], metadata[fine] = build(base, bits, fine)
        print(metadata[fine], flush=True)
    results = []
    for probes in PROBES:
        for depth in CANDIDATE_DEPTHS:
            candidates = {fine: index.candidates(queries, k=depth, nprobe=probes)
                          for fine, index in indexes.items()}
            np.testing.assert_array_equal(candidates[False], candidates[True])
            ceiling = (candidates[False][:, :, None] == truth[:, None, :]).any(axis=1).mean()
            samples = {False: [], True: []}
            ids = {}
            for fine, index in indexes.items():
                search_batches(index, queries, probes, depth)
            for repeat in range(REPEATS):
                for fine in ((False, True) if repeat % 2 == 0 else (True, False)):
                    ids[fine], seconds = timed(lambda: search_batches(indexes[fine], queries, probes, depth))
                    samples[fine].append(seconds / len(queries) * 1000)
            for fine in (False, True):
                row = dict(bits_per_dim=bits, fine_tiers=fine, nprobe=probes, candidates=depth,
                           recall_at_10=float(recall(ids[fine], truth)), candidate_recall_ceiling=float(ceiling),
                           ms_per_query=statistics.median(samples[fine]), samples_ms=samples[fine],
                           **metadata[fine])
                print(json.dumps(row), flush=True)
                results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    with np.load(args.dataset) as data:
        base = np.ascontiguousarray(data["base"], dtype=np.float32)
        queries = np.ascontiguousarray(data["queries"], dtype=np.float32)
        truth = np.ascontiguousarray(data["truth"][:, :K])
    report = dict(platform=platform.platform(), machine=platform.machine(), rows=len(base),
                  dimensions=base.shape[1], queries=len(queries), k=K, lists=LISTS,
                  batch_size=BATCH_SIZE, repeats=REPEATS, omp_threads=os.environ.get("OMP_NUM_THREADS"),
                  base_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
                  queries_sha256=hashlib.sha256(queries.tobytes()).hexdigest(), results=[])
    for bits in args.bits:
        report["results"].extend(measure_pair(base, queries, truth, bits))
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
