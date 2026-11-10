"""Does Faiss IVF-SQ8 with a flat quantizer return the same ids for one batch search call as for single-query calls,
built the way faiss_reference builds it (add_sa_codes with packed list ids)?"""
import os
import faiss
import numpy as np

rng = np.random.default_rng(0)
n, d, nlist, nq, k = 300000, 1024, 4096, 750, 10
x = rng.normal(size=(n, d)).astype(np.float32)
q = rng.normal(size=(nq, d)).astype(np.float32)
for variant in ("ivf", "ivf-hnsw"):
    quantizer = faiss.IndexFlatL2(d) if variant == "ivf" else faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2, False)
    index.train(x[:160000])
    index.add(x)
    index.nprobe = 16
    faiss.omp_set_num_threads(1)
    single = np.vstack([index.search(row[None, :], k)[1] for row in q])
    os.sched_setaffinity(0, range(os.cpu_count()))
    faiss.omp_set_num_threads(32)
    batch = index.search(q, k)[1]
    print(variant, "identical ids:", np.mean(single == batch), "batch -1 share:", np.mean(batch == -1))
