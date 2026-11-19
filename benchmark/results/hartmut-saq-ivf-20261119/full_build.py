"""End-to-end build times from raw rows in RAM (#60), one configuration per process, 32 threads.

  full_build.py tieredq DATA BITS OUT     TieredQIVFIndex as shipped: lists=None (automatic, 2,048 at 1.34M rows), its
                                          own coarse training, assignment, codec training and encoding; then
                                          recall@k at a few nprobe (batch search, recall only).
  full_build.py saq-prep DATA KMEANS OUT  SAQ's preprocessing as upstream python/ivf.py and pca.py do it with Faiss:
                                          k-means lists, every row's nearest list (IndexFlatL2), PCAMatrix over all
                                          dimensions. KMEANS 'upstream' trains IndexIVFFlat as ivf.py does (Faiss
                                          defaults); 'fast' uses the #55 partition settings (64 rows per list, 10
                                          iterations, seed 1), the lists SAQ was measured on. saq_ivf build (PCA_DIR)
                                          then times the transform and construct.
Writes OUT/timing.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

LISTS = 2048
FAST_ROWS_PER_LIST, FAST_ITERATIONS, FAST_SEED = 64, 10, 1
NPROBES = (16, 64, 256)


def load(data):
    rows, d, k, _ = (data / 'meta.txt').read_text().split()
    base = np.fromfile(data / 'base.f32', dtype=np.float32).reshape(int(rows), int(d))
    queries = np.fromfile(data / 'queries.f32', dtype=np.float32).reshape(-1, int(d))
    truth = np.fromfile(data / 'truth.i32', dtype=np.int32).reshape(len(queries), int(k))
    return base, queries, truth


def recall(ids, truth):
    return float(np.mean([len(np.intersect1d(found, expected)) for found, expected in zip(ids, truth)])) / truth.shape[1]


def tieredq(data, bits, out):
    from tieredq import TieredQIVFIndex
    base, queries, truth = load(data)
    code_bits = round(base.shape[1] * float(bits))
    started = time.perf_counter()
    index = TieredQIVFIndex(base.shape[1], code_bits, metric='ip')
    index.build(base)
    total_s = time.perf_counter() - started
    points = []
    for nprobe in NPROBES:
        _, ids = index.search(queries, truth.shape[1], nprobe=nprobe)
        points.append(dict(nprobe=nprobe, recall=recall(ids, truth)))
    return dict(method='tieredq', bits=float(bits), code_bits=code_bits, lists=index.lists, total_s=total_s,
                stages=dict(index.build_stats), storage_bytes=index.storage_bytes, points=points)


def saq_prep(data, kmeans, out):
    import faiss
    base, _, _ = load(data)
    d = base.shape[1]
    started = time.perf_counter()
    if kmeans == 'upstream':
        index = faiss.index_factory(d, f'IVF{LISTS},Flat')
        index.train(base)
        centroids = index.quantizer.reconstruct_n(0, LISTS)
    else:
        sample = np.random.default_rng(FAST_SEED).choice(len(base), LISTS * FAST_ROWS_PER_LIST, replace=False)
        clustering = faiss.Kmeans(d, LISTS, niter=FAST_ITERATIONS, seed=FAST_SEED, max_points_per_centroid=FAST_ROWS_PER_LIST)
        clustering.train(np.ascontiguousarray(base[np.sort(sample)]))
        centroids = clustering.centroids
    kmeans_s = time.perf_counter() - started
    started = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids)
    _, lists = quantizer.search(base, 1)
    assign_s = time.perf_counter() - started
    started = time.perf_counter()
    pca = faiss.PCAMatrix(d, d)
    pca.train(base)
    pca_s = time.perf_counter() - started
    matrix = faiss.vector_to_array(pca.PCAMat).reshape(d, d)
    np.ascontiguousarray(matrix.T, dtype=np.float32).tofile(out / 'pca.f32')
    faiss.vector_to_array(pca.mean).astype(np.float32).tofile(out / 'mean.f32')
    np.ascontiguousarray(centroids, dtype=np.float32).tofile(out / 'centroids.f32')
    lists.astype(np.uint32).ravel().tofile(out / 'assignment.u32')
    return dict(method='saq-prep', kmeans=kmeans, kmeans_s=kmeans_s, assign_s=assign_s, pca_s=pca_s,
                faiss=faiss.__version__, threads=faiss.omp_get_max_threads())


if __name__ == '__main__':
    mode, data, setting, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    result = tieredq(data, setting, out) if mode == 'tieredq' else saq_prep(data, setting, out)
    (out / 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
