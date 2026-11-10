# TieredQ refinement tiers in place of the SQ8 re-rank (#49)

Remote root: `/home/sean/experiments/TieredQ-issue44-nested-20261110`. The router study's scratch 64K index (router at
expansion 64) is measured with the exact SQ8 re-rank of its top 40 and with `TieredQNestedIVFIndex` suffix tiers held in
RAM, on 750 held-out queries, one pinned core. Serving RSS is read from `/proc/self/status` per configuration, the same
way `compare_faiss` reads it for Faiss.

| nprobe | SQ8 re-rank | 4 bits | 5 bits | 6 bits | 8 bits | 6 bits, fine | 8 bits, fine |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | 0.9544 @ 2.46 ms | 0.9407 | 0.9493 | 0.9512 @ 2.77 | 0.9508 | 0.9503 | 0.9532 @ 2.80 |
| 256 | 0.9697 @ 3.80 | 0.9551 | 0.9640 | 0.9665 @ 4.14 | 0.9663 | 0.9656 | 0.9685 @ 4.12 |
| 512 | 0.9788 @ 6.40 | 0.9636 | 0.9727 | 0.9755 @ 6.75 | 0.9752 | 0.9747 | 0.9776 @ 6.75 |
| 1024 | 0.9856 @ 11.4 | 0.9699 | 0.9792 | 0.9823 @ 11.75 | 0.9821 | 0.9816 | 0.9844 @ 11.8 |
| Tier bytes per vector | 1,024 (SQ8 store) | 249 | 381 | 514 | 772 | 509 | 765 |
| Serving RAM | 3.1 + 9.05 GB | 5.5 GB | 6.7 GB | **7.9 GB** | 10.2 GB | 7.8 GB | **10.1 GB** |

The top 100 gave the same recall as the top 40 everywhere. Without fine tiers, 6 and 8 bits plateau at the same recall,
0.003 below SQ8. An L2 codec does not change that (`hartmut-l2fix-20261110`). Fine tiers let the top PCA ranks take up
to 12 bits, and at 8 bits they come within 0.0012 of SQ8. For comparison, Faiss IVF-SQ8 serves at 13.6–14.4 GB (anon).
