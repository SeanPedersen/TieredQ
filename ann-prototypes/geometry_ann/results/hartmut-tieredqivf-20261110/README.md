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

About 9 ms of every query is the exact coarse ranking over 64K centroids (#47). 128K results: pending, see `results/lloyd-128k`.
`run.json`'s serving RSS counts touched pages of the memory-mapped SQ8 file, so use the index byte counts.
