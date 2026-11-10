# Router duel: coarse routers over the leaf centroids

Status: complete. Remote root: `/home/sean/experiments/TieredQ-issue44-routerduel-20261110`. `duel.cpp` builds every router
over the list means of the Lloyd 32K/64K/128K partitions (`centroid_means.py`) and measures 750 held-out queries on one
pinned core. Coverage is the share of each query's true top-10 whose list is in the router's first L lists. Each router
re-ranks its first 256 lists exactly. Raw tables: `results/<size>/duel.txt`.

Coverage at 256 lists:

| Router | 32K | 64K | 128K |
|---|---|---|---|
| Exact FP32 scan | 9.40 ms, 0.9776 | 18.8 ms, 0.9751 | 37.7 ms, 0.9725 |
| TieredQ 2-bit full scan, k=1024 | 2.45 ms, 0.9787 | 3.18 ms, 0.9747 | 5.03 ms, 0.9728 |
| Two-level, SQ8 members, 64 groups | **1.24 ms, 0.9745** | **1.42 ms, 0.9716** | **1.62 ms, 0.9625** |
| Graph (exact 48-NN + RNG, degree 16), SQ4, beam 512 | 1.13 ms, 0.9740 | 1.29 ms, 0.9649 | 1.59 ms, 0.9612 |
| TieredQIVFIndex (L2), 64 of 256 lists | 1.08 ms, 0.8529 | | |

The last row exposed the L2 coarse-list bug in `TieredQIVFIndex`, now fixed: coverage at 16 of 256 lists went from
0.57 to 0.96. The first 128K attempt was killed by systemd-oomd: the graph builder kept an n-wide scratch buffer per row,
137 GB in total. That is fixed, and the rerun completed.
