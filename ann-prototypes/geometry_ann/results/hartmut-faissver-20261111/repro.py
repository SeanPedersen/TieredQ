"""Faiss IVF-SQ8 + IndexFlatL2 quantizer: one multi-threaded batch search vs one-query-at-a-time searches, per
parallel_mode. Synthetic data, no training: random centroids, random SQ8 codes added with random list ids (add_sa_codes).
Usage: repro.py DIM NLIST ROWS [NPROBE] [NQ] [THREADS] [ADD=sa|add]"""
import sys
import numpy as np
import faiss

dim, nlist, rows = (int(a) for a in sys.argv[1:4])
nprobe = int(sys.argv[4]) if len(sys.argv) > 4 else 64
nq = int(sys.argv[5]) if len(sys.argv) > 5 else 64
threads = int(sys.argv[6]) if len(sys.argv) > 6 else 32
add_mode = sys.argv[7] if len(sys.argv) > 7 else "sa"
k = 10
rng = np.random.default_rng(0)

quantizer = faiss.IndexFlatL2(dim)
quantizer.add(rng.standard_normal((nlist, dim), dtype=np.float32))
index = faiss.IndexIVFScalarQuantizer(quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2, False)
faiss.omp_set_num_threads(threads)
if add_mode == "sa":
    index.is_trained = True
    faiss.copy_array_to_vector(np.concatenate([np.full(dim, -4, np.float32), np.full(dim, 8, np.float32)]), index.sq.trained)
    prefix = index.coarse_code_size()
    batch = 1 << 18
    for start in range(0, rows, batch):
        count = min(batch, rows - start)
        lists = rng.integers(0, nlist, count, dtype=np.int64)
        packed = np.empty((count, prefix + dim), np.uint8)
        for byte in range(prefix):
            packed[:, byte] = (lists >> (8 * byte)) & 255
        packed[:, prefix:] = rng.integers(0, 256, (count, dim), dtype=np.uint8)
        index.add_sa_codes(packed)
else:
    data = rng.standard_normal((rows, dim), dtype=np.float32)
    index.train(data[: min(rows, 65536)])
    index.add(data)

queries = rng.standard_normal((nq, dim), dtype=np.float32)
index.nprobe = nprobe
faiss.omp_set_num_threads(1)
single_d, single_i = zip(*(index.search(q[None], k) for q in queries))
single_d, single_i = np.vstack(single_d), np.vstack(single_i)
faiss.omp_set_num_threads(threads)
results = []
for mode in (0, 1, 2, 3):
    index.parallel_mode = mode
    batch_d, batch_i = index.search(queries, k)
    same = float(np.mean(batch_i == single_i))
    distinct = float(np.mean([len(set(row)) for row in batch_i]))
    results.append(f"mode{mode}={same:.3f}(distinct/row {distinct:.1f})")
print(f"faiss={faiss.__version__} dim={dim} nlist={nlist} rows={rows} nprobe={nprobe} nq={nq} threads={threads} "
      f"add={add_mode} ntotal={index.ntotal} | identical ids vs single: " + " ".join(results), flush=True)
