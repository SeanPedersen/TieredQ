# Restricted assignment, end to end

Remote root: `/home/sean/experiments/TieredQ-issue44-assignfast-20261110`. Status: complete.

The study labels all 8.84M rows against the 32K Lloyd centroids (seed 42, 40 rows per centroid, 4 iterations) by
restricted assignment. Each row is assigned among the members of its P nearest of 256 supers, using `fast_assign` with
grouped GEMMs on 32 threads. Label agreement and objective are checked on 50,000 rows against exact assignment. The
geometry_ann index (compressed routing, fixed caps and patience) is then replayed on each label set, plus Faiss
IVF-SQ8 on the P=16 labels, with 750 held-out queries on one pinned core.

| Assignment | Candidates per row | Wall time (32 thr) | Exact labels | Objective |
|---|---:|---:|---:|---:|
| Exhaustive (P=256, control) | 32,768 | 1,079.6 s | 100% (reproduces the existing exact labels: 99.9998%) | 1 |
| P=16 | 2,397 | 81.1 s (13.3×) | 98.89% | +0.13% |
| P=8 | 1,216 | 44.9 s (24×) | 96.68% | +0.41% |

Held-out recall@10 of the same search settings on each partition:

| Setting | Exhaustive | P=16 | P=8 |
|---|---:|---:|---:|
| cap 35,000 | 0.9580 | 0.9576 | 0.9560 |
| patience 48 | 0.9505 | 0.9485 | 0.9477 |
| patience 96 | 0.9660 | 0.9651 | 0.9652 |
| cap 88,408 | 0.9737 | 0.9725 | 0.9728 |
| patience 192 | 0.9759 | 0.9752 | 0.9745 |
| cap 250,000 | 0.9851 | 0.9849 | 0.9844 |
| patience 768, cap 350,000 | 0.9867 | 0.9865 | 0.9855 |
| patience 768 | 0.9879 | 0.9879 | 0.9863 |

Latency and vectors scanned are unchanged within noise. P=16 costs 0.0000–0.0020 recall (mean 0.0008); P=8 costs
0.0007–0.0028. Faiss IVF-SQ8 on the P=16 labels reaches 0.957 at nprobe 96, 0.985 at 768 and 0.990 at 1536.

**Conclusion:** restricted assignment with 16 of 256 probes is a practical replacement for exact assignment. It is
13× faster at 32K lists and costs under 0.002 recall. `TieredQIVFIndex` ships it as
`set_coarse_training({iterations, 256, 16})` (#48). The router study's `scratch-*` models measure it in the library
at 64K and 128K lists, with list training included.
