# L2 codec check (#49)

Remote root: `/home/sean/experiments/TieredQ-issue44-l2fix-20261110`. Hypothesis: the refinement tiers' 0.003 recall
plateau came from the inner-product codec, since the ground truth is exact squared L2 on unnormalized vectors. The 64K
index was rebuilt from scratch with an L2 codec, same recipe: 240 s total, 2.73 GB.

| nprobe | SQ8 re-rank, IP / L2 | 6-bit tier, IP / L2 | p50, IP / L2 (6-bit tier) |
|---:|---|---|---|
| 128 | 0.9544 / 0.9544 | 0.9512 / 0.9516 | 2.77 / 3.39 ms |
| 256 | 0.9697 / 0.9697 | 0.9665 / 0.9668 | 4.14 / 4.78 ms |
| 1024 | 0.9856 / 0.9856 | 0.9823 / 0.9824 | 11.75 / 12.64 ms |

Rejected: the metric does not explain the plateau, and the L2 query lift costs about 0.6 ms. The cause was the per-rank
bit cap; fine tiers address it (`hartmut-nested-20261110`). The shipping index stays inner product.
