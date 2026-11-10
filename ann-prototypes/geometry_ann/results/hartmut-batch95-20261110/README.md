# 32-thread batch QPS near recall 0.95 (#51)

Remote root: `/home/sean/experiments/TieredQ-issue44-batch95-20261110`. The router study's scratch indexes, with exact
GEMM ranking and the router at expansion 64, top-40 SQ8 re-rank, 750 held-out queries per batch call on 32 threads.

| Lists | nprobe | Exact GEMM ranking | Router (expansion 64) |
|---|---:|---|---|
| 64K | 96 | 0.9523 @ 3,268 QPS | 0.9457 @ 5,280 QPS |
| 64K | 128 | 0.9591 @ 3,160 | 0.9544 @ 4,452 |
| 64K | 192 | 0.9661 @ 2,683 | 0.9643 @ 3,293 |
| 128K | 192 | 0.9660 @ 1,860 | 0.9607 @ 3,559 |
| 128K | 256 | 0.9709 @ 1,762 | 0.9663 @ 2,832 |

The router beats the exact GEMM ranking for batches as well as for single queries, so `Mode::automatic` now routes
batches too. Against Faiss's best batch configuration (flat 64K in parallel modes 2/3, see
`hartmut-faisscurve-20261110`; IVF-HNSW 128K), it is 1.4–1.45× faster: 4,452 vs 3,091 QPS at 0.95, and 3,559 vs 2,591
at 0.96.
