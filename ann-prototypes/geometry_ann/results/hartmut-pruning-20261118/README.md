# #45 follow-ups from SuperEMeans: revival threshold and PCA lower-bound pruning (hartmut, 2026-11-18)

Source `dd4999d` plus two experiment-only overrides (not in the repo): `TQ_EMEANS_UNUSED_THRESHOLD` (e-means revival
threshold) and `TQ_SAVE_CENTROIDS` in `examples/coarse_training_benchmark.cpp`.

## Check 1: revival threshold (`revival.sh`, `revival.jsonl`)

8.84M x 1024 Qwen/MS MARCO, 64K lists, all rows x 2 epochs, 256 supers x 4 probes, 32 threads; held-out 131K rows,
exact assignment. Imbalance k * sum n^2 / N^2 = expected rows scanned per probe over a balanced partition's.

| threshold | seed | train_s | train RSS | objective | imbalance |
|---:|---:|---:|---:|---:|---:|
| 0.1 (default) | 0 | 60.9 | 1.55 GB | 0.43781 | 1.763 |
| 0.2 | 0 | 56.2 | 1.55 GB | 0.43785 | 1.716 |
| **0.3** | 0 | 55.0 | 1.55 GB | 0.43872 (+0.2%) | **1.671 (-5.2%)** |
| 0.4 | 0 | 55.5 | 1.55 GB | 0.44111 (+0.8%) | 1.647 (-6.6%) |
| 0.5 | 0 | 56.0 | 1.55 GB | 0.44724 (+2.2%) | 1.689 |
| 0.1 (default) | 7 | 55.8 | 1.55 GB | 0.43777 | 1.757 |
| **0.3** | 7 | 55.8 | 1.54 GB | 0.43894 (+0.3%) | **1.675 (-4.7%)** |

Same time and RAM. 0.2-0.4 trade objective for balance; 0.5 loses both. Whether 5% fewer rows per probe beats a 0.2%
worse objective in recall at equal latency needs a full build + serve (same-window, 2 seeds).

## Check 2: exact PCA lower-bound pruning rate (`pruning_rate.py`, `pruning.jsonl`)

Seed-0 default centroids (64K), 4,000 held-out rows, candidates = members of the row's nearest 4 (training) or 16
(final assignment) of 256 supers. Bound: |P(x-c)|^2 + (|r_x| - |r_c|)^2 with P the top d' PCA directions (exact:
never drops the true nearest). Survivors are rescored in full after exact scoring of the 4 best-bound candidates.

| probes | candidates | d' | survivors (mean / p90) | FLOP ratio vs exact candidates |
|---:|---:|---:|---:|---:|
| 4 | 1,236 | 64 | 668 / 1,429 | 0.66 |
| 4 | 1,236 | **128** | 150 / 476 | **0.35** |
| 4 | 1,236 | 256 | 8.6 / 19 | 0.47 |
| 16 | 4,831 | 64 | 1,966 / 5,002 | 0.48 |
| 16 | 4,831 | **128** | 273 / 732 | **0.21** |
| 16 | 4,831 | 256 | 9.2 / 19 | 0.31 |

Data PCA beats residual PCA slightly; the norm term halves survivors at d' = 128. FLOP ratio counts the projection
(2 d d' per row), the d'-dim bound GEMM and full rescoring of survivors, all at equal efficiency. Survivor rescoring is
an irregular gather, not a GEMM, so realized gains will be smaller than 2.9x (training) / 4.8x (final assignment).

## Check 3: C++ prototype in training's restricted assignment (`proto/tieredq_train_emeans.h`, `prune64k.sh`)

`SuperRestriction::assign_pruned` (experiment-only, `TQ_PRUNE_DIMS`): PCA basis of the centroids at regroup; the first
probed super is scored in full (sgemm, a per-row upper bound); the other 3 probes get a d'-dim sgemm of the bound and an
AVX-512 full dot for candidates below it. Exact: d' = 64 / 128 / 192 give identical partitions. 8.84M x 1024, 64K lists,
all rows x 2 epochs, 256 x 4, 32 threads, order 0, 128, 192, 128, 0.

| d' | train_s | assign stage | rescored (of the pruned probes' candidates) | train RSS | objective | imbalance |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (current) | 55.1 / 55.5 | 42.7 / 42.9 s | - | 1.55 GB | 0.43781 | 1.7626 |
| 128 | 76.1 / 76.0 | 62.4 / 62.4 s | 20.5% | 1.73 GB | 0.43762 | 1.7612 |
| 192 | 56.9 | 43.2 s | 5.9% | 1.75 GB | 0.43762 | 1.7612 |

No gain in training: at 4 probes the exact first probe is a quarter of the work, and the projection, the narrow
d'-GEMMs and the per-candidate rescoring cost what the bound saves (192: ~2x fewer FLOPs, equal time). The final
assignment (16 probes, 4,831 candidates, ~53 s per build) has a 4x better ratio and remains the one open candidate.

## Check 4: C++ prototype in the final restricted assignment (`proto/tieredq_ivf_assign.h`, `assign64k*.sh`)

`RestrictedAssigner::assign_pruned` (experiment-only, `TQ_PRUNE_DIMS`; `TQ_PRUNE_ASSIGN_DIMS` in the full-build copy):
PCA basis of the rounded centroids at setup, per super FP32 panels of the members' projections and a row-major binary16
copy of the members (+128 MB at 64K x 1024) for rescoring. The first probed super is scored by the existing binary16
panel kernel (per-row upper bound), the other 15 probes by the d'-dim bound panel GEMM, survivors by an AVX-512
binary16 dot. `restricted_assign_benchmark.cpp` assigns all 8.84M rows to the saved seed-0 64K centroids as
`encode_lists` does (65,536-row chunks from the SQ8 loader), 256 supers x 16 probes, 32 threads.

| d' | assign_s | setup_s | rescored (of probes 2-16 candidates) |
|---:|---:|---:|---:|
| 0 (current) | 54.24 / 54.41 | 1.4 | - |
| 128 | 52.01 | 2.1 | 12.2% |
| 160 | 41.98 | 2.1 | 5.5% |
| 192 | 37.67 / 38.09 | 2.1 | 2.5% |
| **224** | **36.50** | 2.1 | 1.3% |
| 256 | 36.97 | 2.1 | 0.7% |

Labels: the two d' = 0 runs are byte-identical; d' = 128 and 192 differ from them on the same 6 of 8,840,823 rows
(summation-order ties of the rescoring dot, not pruning). d' = 224: 1.49x faster assignment, -17.7 s per build.
