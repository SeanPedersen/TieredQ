"""Routing fidelity and single-thread scan time of TieredQ centroid tables vs bit budget and per-coordinate scan cap.
`max_scan_bits=4` keeps every tier inside the x86 nibble-lookup kernel; the default (0 = 8 bits for L2) does not.
Inputs (in DATA): centroids.f32 cached by study.py, labels.u32, calibration queries/truth. Same metrics as study.py."""
import json, os, struct, sys, time
from pathlib import Path
import numba
import numpy as np

DATA, OUTPUT = Path(sys.argv[1]), Path(sys.argv[2])
DIM, LEAVES, QUERIES, RECALL_K = 1024, 32768, 250, 10
BUDGETS = [0.25, 0.5, 1, 2, 4]
SCAN_CAPS = [0, 4, 2]
RERANK = [0, 512, 2048]
OVERLAP_K = [128, 1024]
CAPS = [35000, 88408, 250000]
TIMING_K, TIMING_QUERIES = 4096, 50


@numba.njit(cache=True)
def taken_leaves(order, sizes, cap):
    taken = np.zeros(sizes.shape[0], dtype=np.bool_)
    spent = 0
    for leaf in order:
        if spent >= cap:
            break
        if sizes[leaf] <= cap - spent:
            spent += sizes[leaf]; taken[leaf] = True
    return taken


def main():
    from tieredq import TieredQIndex
    centroids = np.fromfile(DATA / 'centroids.f32', dtype='<f4').reshape(LEAVES, DIM)
    labels = np.fromfile(DATA / 'labels.u32', dtype='<u4').astype(np.int64)
    sizes = np.bincount(labels, minlength=LEAVES)
    raw = np.fromfile(DATA / 'calibration_queries.fvecs', dtype='<i4', count=QUERIES * (DIM + 1)).reshape(QUERIES, DIM + 1)
    queries = raw[:, 1:].view('<f4').copy()
    width = struct.unpack('<i', (DATA / 'calibration_truth.ivecs').read_bytes()[:4])[0]
    truth = np.fromfile(DATA / 'calibration_truth.ivecs', dtype='<i4', count=QUERIES * (width + 1)).reshape(QUERIES, width + 1)[:, 1:1 + RECALL_K]
    truth_leaves = labels[truth]
    q64, c64 = queries.astype(np.float64), centroids.astype(np.float64)
    exact = (q64 * q64).sum(1)[:, None] + (c64 * c64).sum(1)[None, :] - 2 * q64 @ c64.T
    exact_rank = np.argsort(exact, axis=1, kind='stable')
    coverage = lambda order, q, cap: float(taken_leaves(order, sizes, cap)[truth_leaves[q]].mean())
    exact_coverage = {cap: np.mean([coverage(exact_rank[q], q, cap) for q in range(QUERIES)]) for cap in CAPS}
    results = []
    for budget in BUDGETS:
        for scan_cap in SCAN_CAPS:
            index = TieredQIndex(DIM, int(budget * DIM), metric='l2', mode='fast', max_scan_bits=scan_cap)
            index.add(centroids)
            scores, ids = index.search(queries, LEAVES)
            timing = []
            for q in range(TIMING_QUERIES):
                begin = time.perf_counter(); index.search(queries[q:q + 1], TIMING_K); timing.append(time.perf_counter() - begin)
            entry = dict(bits_per_dim=budget, max_scan_bits=scan_cap, realized_code_bits=int(index.code_bits),
                         single_query_ms_p50=1e3 * float(np.median(timing)), rerank={})
            for rerank in RERANK:
                metrics = {f'overlap@{k}': 0.0 for k in OVERLAP_K} | {f'coverage_loss@{c}': 0.0 for c in CAPS}
                for q in range(QUERIES):
                    order = ids[q].astype(np.int64)
                    if rerank:
                        head = order[:rerank].copy(); order = order.copy()
                        order[:rerank] = head[np.argsort(exact[q, head], kind='stable')]
                    for k in OVERLAP_K:
                        metrics[f'overlap@{k}'] += len(np.intersect1d(order[:k], exact_rank[q, :k])) / k / QUERIES
                    for cap in CAPS:
                        metrics[f'coverage_loss@{cap}'] += (exact_coverage[cap] - coverage(order, q, cap)) / QUERIES
                entry['rerank'][str(rerank)] = metrics
            print(json.dumps(entry), flush=True)
            results.append(entry)
    OUTPUT.write_text(json.dumps(dict(exact_coverage={str(k): v for k, v in exact_coverage.items()}, methods=results,
                                      omp_num_threads=os.environ.get('OMP_NUM_THREADS')), indent=2) + '\n')


main()
