# TieredQ's own IVF on Qwen 8.84M x 1024

Remote root: `/home/sean/experiments/TieredQ-issue44-tieredqivf-20261110`. `tieredq_ivf_qwen.cpp` builds
`TieredQIVFIndex` from the SQ8 store with the streaming `build(RowLoader)`, on the same Lloyd partitions via
`set_lists` (the list means from the router duel). It builds once per list count: a 2-bit prefix IVF, measured alone,
with an exact SQ8 re-rank of the top 40/100, and as `TieredQNestedIVFIndex` with a suffix tier to 4 bits/dim.
Inner product; 4-bit scan cap; one pinned core. A first run that trained its own lists (20 Lloyd iterations) was
stopped as impractically slow.

64K lists: the build took 1,382 s (assign and encode) plus 115 s (tier). Codes 2.73 GB; tier 2.30 GB. Held out:

| Mode | nprobe | Recall | p50 / p99 ms |
|---|---:|---:|---|
| 2-bit IVF + SQ8 re-rank, top 40 | 96 | 0.950 | 9.8 / 10.6 |
| 2-bit IVF + SQ8 re-rank, top 40 | 256 | 0.971 | 11.2 / 11.9 |
| 2-bit IVF + SQ8 re-rank, top 40 | 1024 | 0.986 | 17.4 / 18.5 |
| 2-bit IVF alone | 1536 | 0.935 (cal) | 21.3 |
| Nested, suffix to 4 bits/dim | 1536 | 0.943 (cal) | 21.7 |

About 9 ms of every query is the exact coarse ranking over 64K centroids (#47).

128K lists: the build took 2,674 s (assign and encode) plus 127 s (tier). Codes 3.14 GB; tier 2.30 GB; no empty lists,
largest list 875. Held out, 2-bit IVF + SQ8 re-rank of the top 40 (top 100 gave the same recall at every nprobe):

| nprobe | Recall | p50 / p99 ms |
|---:|---:|---|
| 64 | 0.939 | 19.1 / 20.1 |
| 128 | 0.957 | 19.4 / 20.3 |
| 256 | 0.970 | 20.4 / 21.3 |
| 512 | 0.979 | 22.1 / 23.2 |
| 1024 | 0.985 | 25.3 / 26.5 |
| 2048 | 0.989 | 31.8 / 33.3 |
| 3072 | 0.991 | 38.0 / 39.5 |

About 19 ms of every query is the exact coarse ranking over 128K centroids, twice the 64K cost. What remains, the scan
and the re-rank, is about 0.4 ms at nprobe 128 and about 6 ms at 1024. At equal vectors scanned (twice the nprobe), 128K lists
reach higher recall than 64K: 0.970 at nprobe 256 against 0.960 at 128, and 0.989 at 2048 against 0.986 at 1024. The 2-bit IVF alone plateaus at 0.941 and the nested 2+2-bit index at 0.948. The two-level router is
measured end to end in `hartmut-router-20261110`.
`run.json`'s serving RSS counts touched pages of the memory-mapped SQ8 file, so use the index byte counts.
