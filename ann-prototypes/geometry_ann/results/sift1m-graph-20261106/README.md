# SIFT1M: centroid-graph routing and geometric bounds against Faiss (#44)

Status: complete. This is the first real-data run of the geometry prototype, committed in 0128676 without a README.
This README was written afterwards from the committed files. The host and the exact build are not recorded, so treat
the latencies as relative.

**Data** (`dataset-manifest.json`): SIFT1M from `sift-128-euclidean.hdf5`, 1,000,000 × 128, squared L2, no
normalization. The manifest records the seed-42 split, the hashes and the query ids. 250 calibration queries pick the
settings (`calibration*.csv`, `selected-settings.*`); `heldout.csv` replays them on the disjoint held-out queries.

**Index:** 8,192 leaves. Settings sweep the routing (graph over the leaf centroids, flat, hierarchy), the bounds
(none, radius, pca, pca-max, shared) and the variance weight 0 / 0.5 / 1, at vector caps of 20K and 50K. The build
takes 15 s (7.4 s of it clustering) and adds 24.5 MB over the 512 MB of raw vectors.
`navigation-only-heldout.csv` repeats the replay with metadata built only for navigation.

## Held-out results (selected settings)

| Index | Recall@10 | p50 / p99 µs | Vectors or distances |
|---|---:|---:|---:|
| Graph routing, radius bounds, cap 20K | 0.9643 | 1,092 / 1,308 | 20K vectors |
| Flat routing, cap 20K | 0.9671 | 1,756 / 2,688 | 20K |
| Graph routing, no bounds, cap 50K | 0.9950 | 2,699 / 3,169 | 50K |
| Flat routing, cap 50K | 0.9955 | 2,929 / 4,003 | 50K |
| Faiss HNSW (M 32, efC 100), ef 80 / 160 | 0.975 / 0.992 | 170 / 314 | 1.8K / 3.1K distances |
| Faiss IVF-Flat 8,192, nprobe 128 / 256 | 0.9817 / 0.9953 | 422 / 729 | 18.6K / 36K |
| Faiss IVF-HNSW 8,192, nprobe 128 / 256 | 0.9821 / 0.9950 | 416 / 734 | 18.6K / 36K |

## Findings

- Geometric bounds prune almost nothing. Radius bounds keep recall but barely change the work. PCA bounds add about
  0.5 ms per query. A variance weight of 1 costs 0.03–0.14 recall.
- Graph routing is about 40% cheaper than flat routing at the same cap and recall.
- Faiss IVF needs fewer vectors than the prototype for the same recall: 18.6K at 0.98, against more than 20K for
  0.964. Its query is also 2–4× faster.

These results pointed the later studies (`../hartmut-simd-20261107` onward) at the scan kernel and the partition
quality.
