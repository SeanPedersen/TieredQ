"""Offline fidelity of compressed leaf centroids for flat routing (Qwen SQ8, flat Lloyd 32K).
Centroids match the index: float64 means of decoded SQ8 rows per label, rounded to float32.
Each method ranks all leaves for the 250 calibration queries; optionally the approximate top R are re-ranked exactly.
Fidelity: overlap with the exact top-K leaves, and truth coverage under fixed vector caps (the index's skip-if-too-large
scan rule) against the exact order. Runs under the shared benchmark lock; resumable via cached centroids."""
import datetime, fcntl, importlib.metadata, json, struct, sys, time
from pathlib import Path
import numba
import numpy as np
import scipy.sparse as sparse

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
LABELS = EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32'
ROWS, DIM, LEAVES = 8840823, 1024, 32768
QUERIES, RECALL_K = 250, 10
CHUNK = 65536
SPECTRUM_SAMPLE = 200000
TIEREDQ_BITS = [1, 2, 3, 4, 8]
RERANK = [0, 128, 512, 2048]
OVERLAP_K = [128, 1024]
CAPS = [35000, 88408, 250000]
TIMING_K, TIMING_QUERIES = 2048, 100


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def read_fvecs(path, count):
    raw = np.fromfile(path, dtype='<i4', count=count * (DIM + 1)).reshape(count, DIM + 1)
    assert (raw[:, 0] == DIM).all()
    return raw[:, 1:].view('<f4').copy()


def read_ivecs(path, count):
    width = struct.unpack('<i', path.read_bytes()[:4])[0]
    return np.fromfile(path, dtype='<i4', count=count * (width + 1)).reshape(count, width + 1)[:, 1:1 + RECALL_K]


def decoded(codes, scales, offsets, begin, end):
    block = codes[begin:end].astype(np.float32); block *= scales; block += offsets
    return block


def compute_centroids(codes, scales, offsets, labels):
    cache = ROOT / 'centroids.f32'
    if cache.exists():
        return np.fromfile(cache, dtype='<f4').reshape(LEAVES, DIM)
    sums = np.zeros((LEAVES, DIM), dtype=np.float64)
    for begin in range(0, ROWS, CHUNK):
        end = min(begin + CHUNK, ROWS)
        onehot = sparse.csr_matrix((np.ones(end - begin), (labels[begin:end], np.arange(end - begin))), shape=(LEAVES, end - begin))
        sums += onehot @ decoded(codes, scales, offsets, begin, end).astype(np.float64)
    centroids = (sums / np.bincount(labels, minlength=LEAVES)[:, None]).astype(np.float32)
    centroids.tofile(cache)
    return centroids


def spectrum(matrix):
    centered = matrix - matrix.mean(axis=0)
    values = np.linalg.eigvalsh(centered.T.astype(np.float64) @ centered)[::-1]
    share = np.cumsum(values) / values.sum()
    return {str(m): float(share[m - 1]) for m in (16, 64, 128, 256, 512)}


def squared_distances(queries, points):
    q = queries.astype(np.float64); p = points.astype(np.float64)
    return (q * q).sum(1)[:, None] + (p * p).sum(1)[None, :] - 2 * q @ p.T


def int8_per_dimension(centroids):
    low, high = centroids.min(0), centroids.max(0)
    scale = np.maximum(high - low, 1e-30) / 255
    return (np.round((centroids - low) / scale) * scale + low).astype(np.float32)


@numba.njit(cache=True)
def taken_leaves(order, sizes, cap):
    """Leaves taken in `order` while skipping leaves that no longer fit the cap (the index's scan rule)."""
    taken = np.zeros(sizes.shape[0], dtype=np.bool_)
    spent = 0
    for leaf in order:
        if spent >= cap:
            break
        if sizes[leaf] <= cap - spent:
            spent += sizes[leaf]; taken[leaf] = True
    return taken


def scan_coverage(order, sizes, truth_leaves, cap):
    return float(taken_leaves(order, sizes, cap)[truth_leaves].mean())


def evaluate(name, approximate, exact, sizes, truth_leaves, exact_coverage, extra):
    exact_rank = np.argsort(exact, axis=1, kind='stable')
    approx_rank = np.argsort(approximate, axis=1, kind='stable')
    result = dict(name=name, **extra, rerank={})
    for rerank in RERANK:
        metrics = {f'overlap@{k}': 0.0 for k in OVERLAP_K} | {f'coverage@{c}': 0.0 for c in CAPS}
        for q in range(len(exact)):
            order = approx_rank[q].copy()
            if rerank:
                head = order[:rerank]
                order[:rerank] = head[np.argsort(exact[q, head], kind='stable')]
            for k in OVERLAP_K:
                metrics[f'overlap@{k}'] += len(np.intersect1d(order[:k], exact_rank[q, :k])) / k
            for cap in CAPS:
                metrics[f'coverage@{cap}'] += scan_coverage(order, sizes, truth_leaves[q], cap)
        metrics = {key: value / len(exact) for key, value in metrics.items()}
        metrics |= {f'coverage_loss@{cap}': exact_coverage[cap] - metrics[f'coverage@{cap}'] for cap in CAPS}
        result['rerank'][str(rerank)] = metrics
    print(json.dumps(result), flush=True)
    return result


def main():
    from tieredq import TieredQIndex
    codes = np.memmap(QWEN / 'base.i8', dtype='int8', mode='r', shape=(ROWS, DIM))
    scales = np.fromfile(QWEN / 'scales.f32', dtype='<f4'); offsets = np.fromfile(QWEN / 'offsets.f32', dtype='<f4')
    labels = np.fromfile(LABELS, dtype='<u4').astype(np.int64)
    status('centroids'); started = time.time()
    centroids = compute_centroids(codes, scales, offsets, labels)
    sizes = np.bincount(labels, minlength=LEAVES)
    queries = read_fvecs(QWEN / 'data/calibration_queries.fvecs', QUERIES)
    truth = read_ivecs(QWEN / 'data/calibration_truth.ivecs', QUERIES)
    truth_leaves = labels[truth]
    status('spectrum', centroid_s=time.time() - started)
    sample = np.random.default_rng(42).choice(ROWS, SPECTRUM_SAMPLE, replace=False); sample.sort()
    spectra = dict(centroids=spectrum(centroids), data=spectrum(codes[sample].astype(np.float32) * scales + offsets))
    exact = squared_distances(queries, centroids)
    exact_rank = np.argsort(exact, axis=1, kind='stable')
    exact_coverage = {cap: float(np.mean([scan_coverage(exact_rank[q], sizes, truth_leaves[q], cap) for q in range(QUERIES)])) for cap in CAPS}
    results = []
    status('controls')
    results.append(evaluate('binary16', squared_distances(queries, centroids.astype(np.float16).astype(np.float32)), exact, sizes,
                            truth_leaves, exact_coverage, dict(bits_per_dim=16, table_mb=LEAVES * DIM * 2 / 2**20)))
    results.append(evaluate('int8-per-dim', squared_distances(queries, int8_per_dimension(centroids)), exact, sizes,
                            truth_leaves, exact_coverage, dict(bits_per_dim=8, table_mb=LEAVES * DIM / 2**20)))
    for bits in TIEREDQ_BITS:
        status('tieredq', bits=bits)
        index = TieredQIndex(DIM, bits * DIM, metric='l2', mode='fast')
        build = time.time(); index.add(centroids); build = time.time() - build
        scores, ids = index.search(queries, LEAVES)
        approximate = np.full((QUERIES, LEAVES), np.inf)
        np.put_along_axis(approximate, ids.astype(np.int64), -scores.astype(np.float64), axis=1)
        timing = []
        for q in range(TIMING_QUERIES):
            begin = time.perf_counter(); index.search(queries[q:q + 1], TIMING_K); timing.append(time.perf_counter() - begin)
        results.append(evaluate(f'tieredq-{bits}b', approximate, exact, sizes, truth_leaves, exact_coverage,
                                dict(bits_per_dim=bits, realized_code_bits=int(index.code_bits), table_mb=bits * DIM * LEAVES / 8 / 2**20,
                                     train_s=build, single_query_ms_p50=1e3 * float(np.median(timing)))))
    summary = dict(spectra=spectra, exact_coverage={str(k): v for k, v in exact_coverage.items()}, methods=results,
                   tieredq_version=importlib.metadata.version('tieredq'))
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main(); status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=repr(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
