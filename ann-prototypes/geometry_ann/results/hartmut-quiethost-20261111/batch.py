"""Batch QPS of the low-RAM SSD configuration (hartmut-ssdtier follow-up, #52): the saved nested index with its tier on
the NVMe SSD (direct reads, 32 in flight per query), all held-out queries in one search call on the process's
threads (OMP_NUM_THREADS), best of PASSES, at each nprobe. Also records recall@10 and anonymous RSS.
Usage: batch.py INDEX.tq TIER.rows QUERIES.fvecs TRUTH.ivecs COUNT NPROBES OUT.csv"""
import sys
import time

import numpy as np
from tieredq import TieredQNestedIVFIndex

TOP_K, PASSES = 10, 3


def vecs(path, dtype, count, keep):
    raw = np.fromfile(path, dtype=np.int32)
    return raw.reshape(-1, int(raw[0]) + 1)[:count, 1:1 + keep].copy().view(dtype)


def anonymous_rss():
    for line in open('/proc/self/status'):
        if line.startswith('RssAnon:'):
            return int(line.split()[1]) * 1024
    return 0


index_path, tier_path, queries_path, truth_path, count, nprobes, out = sys.argv[1:8]
count = int(count)
index = TieredQNestedIVFIndex.load(index_path)
index.attach_tier(tier_path, reads='direct')
queries = np.ascontiguousarray(vecs(queries_path, np.float32, count, index.dimensions))
truth = vecs(truth_path, np.int32, count, TOP_K)
with open(out, 'w') as report:
    report.write('nprobe,recall10,qps,anon_bytes\n')
    for nprobe in map(int, nprobes.split(',')):
        index.search(queries, k=TOP_K, nprobe=nprobe)  # warm
        best, ids = 0.0, None
        for _ in range(PASSES):
            started = time.perf_counter()
            _, ids = index.search(queries, k=TOP_K, nprobe=nprobe)
            best = max(best, count / (time.perf_counter() - started))
        hits = sum(len(np.intersect1d(found, true)) for found, true in zip(ids, truth))
        report.write(f'{nprobe},{hits / (count * TOP_K):.4f},{best:.1f},{anonymous_rss()}\n')
        report.flush()
