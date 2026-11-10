"""Coarse router duel (#50): Faiss HNSW over the centroids (IndexHNSWFlat, and IndexHNSWSQ 8-bit with an exact re-rank
of its first 256 lists) against TieredQ's two-level router (TieredQIVFIndex.probe_lists with the router; an IVF whose
lists are fixed to the centroids, built over the centroids themselves). Every query is routed alone on one pinned
core. Two quality measures per depth L: coverage, the share of each query's true top-10 neighbours whose list is among
the router's first L (when row labels are given), and routing recall, the share of the exact first L lists found.
Writes one CSV row per router setting."""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import tieredq

HNSW_NEIGHBORS, HNSW_CONSTRUCTION_EF = 32, 100  # as the Faiss IVF-HNSW quantizer of the scorecard
RERANK = 256
DEPTHS = (64, 128, 256, 512, 1024, 2048)
EF_SEARCH = (64, 128, 256, 512, 1024, 2048, 4096)
SUPER_PROBES = (8, 16, 32, 64, 128, 256, 512, 1024)


def read_vecs(path: Path, dtype, rows: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    width = raw[0]
    table = raw.reshape(-1, width + 1)[:rows, 1:]
    return table.view(dtype) if dtype != np.int32 else table


def exact_order(centroids: np.ndarray, queries: np.ndarray, depth: int) -> np.ndarray:
    """First `depth` lists by exact L2, per query (blocked matmul)."""
    norms = (centroids ** 2).sum(1)
    out = np.empty((len(queries), depth), dtype=np.int64)
    for first in range(0, len(queries), 64):
        block = norms[None, :] - 2 * queries[first:first + 64] @ centroids.T
        part = np.argpartition(block, depth, axis=1)[:, :depth]
        order = np.take_along_axis(block, part, 1).argsort(1)
        out[first:first + 64] = np.take_along_axis(part, order, 1)
    return out


def quality(orders: list[np.ndarray], exact: np.ndarray, true_lists: np.ndarray | None) -> dict:
    result = {}
    for depth in DEPTHS:
        recall = np.mean([len(np.intersect1d(order[:depth], exact[q, :depth])) / depth for q, order in enumerate(orders)])
        result[f'recall@{depth}'] = recall
        if true_lists is not None:
            result[f'coverage@{depth}'] = np.mean([np.isin(true_lists[q], order[:depth]).mean() for q, order in enumerate(orders)])
    return result


def timed(route, queries: np.ndarray) -> tuple[list[np.ndarray], float]:
    route(queries[0])
    orders, start = [], time.perf_counter()
    for query in queries:
        orders.append(route(query))
    return orders, (time.perf_counter() - start) * 1000 / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centroids', type=Path, required=True, help='lists x dim float32')
    parser.add_argument('--labels', type=Path, help='list of every base row (uint32), for coverage')
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--nq', type=int, default=750)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--search-cpu', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    centroids = np.fromfile(args.centroids, dtype=np.float32).reshape(-1, args.dim)
    queries = np.ascontiguousarray(read_vecs(args.queries, np.float32, args.nq)[:, :args.dim])
    truth = read_vecs(args.truth, np.int32, args.nq)[:, :10]
    lists = len(centroids)
    true_lists = np.fromfile(args.labels, dtype=np.uint32)[truth] if args.labels else None
    exact = exact_order(centroids, queries, max(DEPTHS))
    print(f'lists={lists} queries={len(queries)}', file=sys.stderr, flush=True)

    faiss.omp_set_num_threads(args.threads)
    builds = {}
    start = time.perf_counter()
    flat = faiss.IndexHNSWFlat(args.dim, HNSW_NEIGHBORS)
    flat.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
    flat.add(centroids)
    builds['hnsw-flat'] = time.perf_counter() - start
    start = time.perf_counter()
    sq = faiss.IndexHNSWSQ(args.dim, faiss.ScalarQuantizer.QT_8bit, HNSW_NEIGHBORS)
    sq.hnsw.efConstruction = HNSW_CONSTRUCTION_EF
    sq.train(centroids)
    sq.add(centroids)
    builds['hnsw-sq8'] = time.perf_counter() - start
    start = time.perf_counter()
    ivf = tieredq.TieredQIVFIndex(args.dim, 2 * args.dim, lists=lists, metric='l2', max_scan_bits=4)
    ivf.set_lists(centroids)
    # Each centroid is its own list's only row; restricted assignment finds it in its own super.
    ivf.set_coarse_training(assignment_supers=max(1, lists // 128), assignment_probes=8)
    ivf.build(centroids)
    ivf.build_router(min_super_probes=1)
    builds['tieredq-router'] = time.perf_counter() - start
    print(f'builds {builds}', file=sys.stderr, flush=True)

    faiss.omp_set_num_threads(1)
    os.sched_setaffinity(0, {args.search_cpu})
    rows = []

    def record(router, setting, route):
        orders, ms = timed(route, queries)
        rows.append(dict(router=router, setting=setting, route_ms=ms, build_s=builds.get(router, 0),
                         **quality(orders, exact, true_lists)))
        print(rows[-1], file=sys.stderr, flush=True)

    for ef in EF_SEARCH:
        flat.hnsw.efSearch = ef
        depth = min(ef, max(DEPTHS))
        record('hnsw-flat', f'ef={ef}', lambda q, d=depth: flat.search(q[None, :], d)[1][0])
        sq.hnsw.efSearch = ef

        def sq_route(q, d=depth):
            found = sq.search(q[None, :], d)[1][0]
            head = found[:RERANK]
            exact_head = head[np.argsort(((centroids[head] - q) ** 2).sum(1))]
            return np.concatenate([exact_head, found[RERANK:]])
        record('hnsw-sq8', f'ef={ef}', sq_route)
    # Routed to the first 256 lists (shallow probing) and to 2048 (deep); quality beyond nprobe counts as missing.
    for nprobe in (RERANK, max(DEPTHS)):
        for probes in (p for p in SUPER_PROBES if p <= max(1, lists // 128)):
            record('tieredq-router', f'nprobe={nprobe} super_probes={probes}',
                   lambda q, n=nprobe, p=probes: ivf.probe_lists(q, nprobe=n, routing='router', super_probes=p))
    record('exact-binary16', 'all', lambda q: ivf.probe_lists(q, nprobe=max(DEPTHS), routing='exact'))

    with args.output.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
