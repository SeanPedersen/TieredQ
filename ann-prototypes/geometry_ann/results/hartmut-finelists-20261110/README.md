# Experiment 3, stage A: 64K/128K Lloyd labels and Faiss IVF-SQ8 baselines

Remote root: `/home/sean/experiments/TieredQ-issue44-finelists-20261110`. Steps: Lloyd 64K/128K (4 iterations, 40 rows
per centroid, exact assignment; labels stay on hartmut), then Faiss IVF-SQ8 with an exact (flat) and an HNSW quantizer at
each list count. The queue was reordered so that the 128K exact-quantizer run comes last. Status: complete.

| Step | Build s | Serving RSS | Held-out recall @ p50 / p99 |
|---|---:|---:|---|
| Lloyd 64K | 1,635 (894 train + 742 assign) | – | – |
| Lloyd 128K | 5,060 (3,569 train + 1,490 assign) | – | – |
| Faiss IVF-SQ8 64K, exact quantizer | 1,587 | 14.4 GB | 0.951 @ 19.8 / 21.2 ms; 0.989 @ 49.7 / 54.4 ms |
| Faiss IVF-SQ8 64K, HNSW quantizer | 1,611 | 13.7 GB | 0.949 @ 3.3 / 4.2 ms; 0.985 @ 24.5 / 28.4 ms |
| Faiss IVF-SQ8 128K, HNSW quantizer | 3,065 | 14.0 GB | 0.943 @ 2.0 / 2.7 ms; 0.987 @ 21.9 / 25.9 ms |
| Faiss IVF-SQ8 128K, exact quantizer | 4,729 | 14.7 GB | 0.957 @ 37.9 / 39.4 ms; 0.987 @ 52.6 / 56.3 ms; 0.993 @ 101.3 / 109.4 ms |

Full nprobe curves (250 calibration queries) are in each step's `calibration.csv`.
