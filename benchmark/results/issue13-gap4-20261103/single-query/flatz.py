"""Flat single-query (1 thread, batch 1) ms/query and R@10 per pruning arm (TQ_FLAT_Z / TQ_FLAT_PREFIX hook),
one index, arms interleaved over rounds."""
import os, statistics, sys, time
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
from measure import recall
ARMS = [("z0-p4", "0", "4"), ("z8-p3", "8", "3"), ("z4-p3", "4", "3"), ("z2-p3", "2", "3"), ("z8-p2", "8", "2"), ("z4-p2", "4", "2"), ("z2-p2", "2", "2")]
dataset, data, code_bits, rounds = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
base, queries, truth = prepare(data, dataset)
index = codec_native.Index(base, -(-code_bits // base.shape[1]), 0, True, False, True, code_bits, True)
ids = np.empty((len(queries), 10), dtype=np.uint32)
times = {arm: [] for arm, _, _ in ARMS}; recalls = {}
for _ in range(rounds):
    for arm, z, prefix in ARMS:
        os.environ["TQ_FLAT_Z"], os.environ["TQ_FLAT_PREFIX"] = z, prefix
        start = time.perf_counter()
        for q in range(len(queries)):
            index.search(queries[q:q + 1], ids[q:q + 1], 1)
        times[arm].append((time.perf_counter() - start) / len(queries) * 1e3)
        recalls[arm] = recall(truth, ids)
for arm, _, _ in ARMS:
    print(f"{dataset} {code_bits} {arm}: {statistics.median(times[arm]):.3f} ms/query, R@10 {recalls[arm]:.4f}", flush=True)
