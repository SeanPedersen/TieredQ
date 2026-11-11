# Faiss flat-quantizer batch bug on Faiss main, and 1.15.1 vs 1.16.0 (#52)

**The bug is fixed upstream.** `repro.py` is a self-contained repro on synthetic data: random centroids and SQ8 codes,
no training. It compares one multi-threaded batch call with one-query searches per `parallel_mode`. On Faiss 1.15.1,
mode 0 returns one repeated id and distance per query once nlist ≥ max(10,000, threads × 1,024): 16,384 lists pass and
32,768 fail at 32 threads, and 65,536 lists fail even at 2 threads with 2 queries, at any dimension, with `add` or
`add_sa_codes`.

Cause: in mode 0 every OpenMP worker calls the flat quantizer, and `knn_L2sqr` takes its database-parallel path
(added in #5000, shipped since v1.14.2), which assumed its own threads inside a nested region. This was already
reported as faiss#5702 and fixed by #5824 (7ac4a6c, 2026-10-30), released in v1.16.0. `build.sh` builds Faiss main
(ddc7680, 2026-11-10): the repro passes in every mode, so nothing was filed.

**1.15.1 vs 1.16.0** (`faissver.py`, `results/`; a Faiss-vs-Faiss diagnostic on the given 64K Lloyd partition, both
quantizers, 750 held-out queries):

| | 1.15.1 | 1.16.0 |
|---|---|---|
| IVF-HNSW single p50 at 0.948 / 0.963 / 0.984 | 3.31 / 6.61 / 24.2 ms | 3.13 / 6.07 / 23.4 ms |
| Flat single p50 at 0.960 / 0.979 / 0.991 | 20.8 / 28.7 / 60.2 ms | 20.6 / 28.3 / 58.3 ms |
| Flat batch QPS, best mode, nprobe 128 / 512 | 2,545 / 876 | 2,656 / 899 |
| Flat batch, mode 0 | recall 0 (bug) | correct, but slower than modes 2-3 |
| IVF-HNSW batch QPS, best mode, nprobe 128 / 512 | 2,983 / 793 | 3,003 / 808 |

Recall is identical at every budget. 1.16.0 is 1-8% faster on single queries and the same on batches, so the 1.15.1
baselines of the #52 scorecard stand. Against 1.16.0, our narrowest latency margin would shrink by about 5%. The full
1.16.0 curve rerun was cancelled as not worth an hour of hartmut.
