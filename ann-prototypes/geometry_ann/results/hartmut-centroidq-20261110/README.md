# Compressed leaf centroids for flat routing: offline fidelity

Status: complete (exit 0); see `summary.json`. Remote root: `/home/sean/experiments/TieredQ-issue44-centroidq-20261110`
(the 128 MB `centroids.f32` cache is not mirrored).

Setup: flat Lloyd 32K labels, Qwen SQ8. Centroids are float64 means of decoded SQ8 rows rounded to float32, as in the index.
The exact-order coverage matches the earlier policy simulation (0.9644 / 0.9836 / 0.9940 at the 35K / 88K / 250K caps). The 250 calibration queries rank
all 32,768 leaves with each method. The approximate top R are then re-ranked with exact distances. "Coverage loss" is the change in true-neighbor
coverage under the index's fixed-cap scan rule (one true neighbor = 0.0004). TieredQ is the PyPI package 0.1.1 (`metric="l2"`, `mode="fast"`), trained on the centroids.

PCA variance share in the top 16/64/128/256/512 components: centroids 0.30/0.58/0.75/0.89/0.97, data 0.18/0.42/0.60/0.79/0.94.

| Method | Table | Top-128 kept (R=0) | Top-1024 kept (R=0) | Coverage loss R=0 (35K/88K/250K) | R=512: top-128 kept | Coverage loss R=512 |
|---|---:|---:|---:|---|---:|---|
| binary16 | 64 MB | 1.000 | 1.000 | 0 / 0 / 0 | 1.000 | 0 / 0 / 0 |
| int8 per dim | 32 MB | 0.997 | 0.997 | 0 / 0 / .0004 | 1.000 | 0 / 0 / .0004 |
| TieredQ 1 bit/dim | 4 MB | 0.937 | 0.947 | .0048 / .0004 / .0004 | 1.000 | 0 / 0 / .0004 |
| TieredQ 2 bits/dim | 8 MB | 0.976 | 0.980 | 0 / .0004 / 0 | 1.000 | 0 / 0 / 0 |
| TieredQ 3 bits/dim | 12 MB | 0.989 | 0.990 | .0004 / 0 / 0 | 1.000 | 0 / 0 / 0 |
| TieredQ 4 bits/dim | 16 MB | 0.994 | 0.995 | 0 / 0 / .0004 | 1.000 | 0 / 0 / .0004 |
| TieredQ 8 bits/dim | 32 MB | 0.995 | 0.996 | 0 / 0 / .0004 | 1.000 | 0 / 0 / .0004 |

With R=2048, every method keeps the exact top 1024 (overlap 1.000) and loses no coverage.

Findings: centroid variance is more concentrated than the data's, as expected. TieredQ at 2 bits/dim (8 MB, smaller than the
~25 MB L3) with an exact re-rank of the top 512 keeps the exact top 128 order and loses no coverage. Even 1 bit/dim
loses at most one true neighbor. The Python single-query timing (~6.5–6.9 ms for k=2048, flat across bit widths) is dominated by
binding and top-k selection overhead, not by the table scan, and does not indicate C++ routing cost.

## Scan-width cap (local follow-up)

`scan_cap_fidelity.py` (local run in `scan_cap_fidelity.local.txt`) repeats the fidelity test with `max_scan_bits` ∈ {default, 4, 2}.
At 2 bits/dim, a 4-bit cap lowers the exact top-128 overlap without re-rank from 0.976 to 0.947, but R=512 still loses no coverage.
In the C++ single-thread benchmark, the cap makes the 2-bit search about 22× faster (1.6 vs 35 ms at k=10 on an i7-8550U).
