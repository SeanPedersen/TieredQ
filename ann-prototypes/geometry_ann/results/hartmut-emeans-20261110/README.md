# Flat partitions: Faiss Lloyd vs E-means through our index

Status: complete (exit 0); see `REPORT.md`. A full-data E-means run was stopped as unfair and kept in `stopped-emeans-32k-e1-fulldata/`. The 8K pair was skipped. On identical 1.31M training rows, E-means reaches a lower objective at 2 and 4 epochs but lower held-out recall per scanned vector than Lloyd (95% target: Lloyd 0.9580 at 35K vs E-means e2 0.9507 and e4 0.9496). It is not adopted.
Remote root: `/home/sean/experiments/TieredQ-issue44-emeans-20261110` (`status.json`, `job.exitcode`).

## Why

The assignment study showed partition quality decides scanned vectors. Our hierarchical k-means with leaf
reassignment works, but at 32K leaves it builds in 1,540 s (1,313 s reassignment), because our exact double kernel
runs at about 1 GMAC/s per thread. E-means (Meta FAIR, arXiv 2611.09501; `faiss/contrib/e_means`) is a mini-batch k-means
with EMA centroids and fitted step/batch laws. It claims about 10× fewer passes than Lloyd at equal loss, and its CPU path assigns
with BLAS (`faiss.knn`). Training flat partitions externally removes the hierarchy restriction entirely.

## Setup

- `train_partitions.py` (uv, faiss-cpu 1.15.1, 32 threads) trains on **decoded INT8** rows, streamed in 64K-row
  chunks, so the 36 GB FP32 corpus is never materialized. Methods:
  - `lloyd`: `faiss.Kmeans`, 4 iterations, 40 rows per centroid (Faiss IVF training).
  - `emeans`: `NumpyEmeans` caller-driven loop over all rows, 1 or 2 passes, default scaling laws.
  Both end with one exact assignment pass against the final centroids. Writes `labels.u32` and `partition.json`
  (times, objective, empty clusters, memory peaks).
- E-means is not in the faiss-cpu 1.15.1 wheel. The three pure-NumPy files (`__init__.py`, `base.py`,
  `e_means.py`, MIT) are vendored on hartmut at Faiss commit `fc1095a9314f892938bf83f59e7ac966b5095def` (2026-11-09).
  Hashes are in `vendor-sha256.txt`; they are not committed to this repo.
- `graph_bench --assignments labels.u32` builds our index from the labels. Coarse parents for the graph come
  from clustering leaf centroids. Same search, protocol and settings as the assignment study: flat and graph routing, 10 vector
  caps, calibration (250) → frozen selections → held-out (750). Search is pinned to one core.

Models: `lloyd-32k`, `emeans-32k-e1`, `emeans-32k-e2`, then `lloyd-8k`, `emeans-8k-e2`.
Compare: build time (train + assign + index) and vectors scanned at 95/99% against `ours-32k-p8`,
`ours-8k-p4`, and Faiss IVF-SQ8 at 8K/32K in `../hartmut-assignment-20261110/`.
