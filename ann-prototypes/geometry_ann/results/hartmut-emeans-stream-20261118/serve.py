"""Serves the saved index with its tier on the NVMe SSD (direct reads) and measures held-out recall@10 and per-query
latency at the default candidates, one query at a time; run alone on one pinned core with one OpenMP thread. Anonymous
RSS (/proc/self/status) after each nprobe is the serving RAM.
Usage: serve.py INDEX.tq TIER.rows QUERIES.fvecs TRUTH.ivecs COUNT NPROBES OUT.csv [CANDIDATES] (default: the index's)"""
import sys, time
import numpy as np
from tieredq import TieredQNestedIVFIndex

TOP_K = 10


def vecs(path, dtype, count, keep):
    raw = np.fromfile(path, dtype=np.int32)
    width = int(raw[0])
    return raw.reshape(-1, width + 1)[:count, 1:1 + keep].copy().view(dtype)


def anonymous_rss():
    for line in open('/proc/self/status'):
        if line.startswith('RssAnon:'):
            return int(line.split()[1]) * 1024
    return 0


index_path, tier_path, queries_path, truth_path, count, nprobes, out = sys.argv[1:8]
options = dict(candidates=int(sys.argv[8])) if len(sys.argv) > 8 else {}
count = int(count)
index = TieredQNestedIVFIndex.load(index_path)
index.attach_tier(tier_path, reads='direct')
queries = vecs(queries_path, np.float32, count, index.dimensions)
truth = vecs(truth_path, np.int32, count, TOP_K)
with open(out, 'w') as report:
    report.write('nprobe,recall10,p50_us,p99_us,mean_us,anon_bytes\n')
    for nprobe in map(int, nprobes.split(',')):
        index.search(queries[0], k=TOP_K, nprobe=nprobe, **options)  # warm
        micros, hits = np.empty(count), 0
        for q in range(count):
            started = time.perf_counter()
            _, ids = index.search(queries[q], k=TOP_K, nprobe=nprobe, **options)
            micros[q] = (time.perf_counter() - started) * 1e6
            hits += len(np.intersect1d(ids, truth[q]))
        report.write(f'{nprobe},{hits / (count * TOP_K):.4f},{np.percentile(micros, 50):.1f},'
                     f'{np.percentile(micros, 99):.1f},{micros.mean():.1f},{anonymous_rss()}\n')
        report.flush()
