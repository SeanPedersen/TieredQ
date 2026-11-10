# Cell ceiling: how many clusters can hyperplane bounds rule out?

Status: complete (2026-11-10 03:00 UTC, exit 0). See `REPORT.md`; hull-rank (added after this run) not yet measured.
Remote root: `/home/sean/experiments/TieredQ-issue44-cells-20261110` (`status.json`, `job.exitcode`).

## Question

The goal is to scan fewer clusters by using cluster geometry, in particular the hyperplanes
that k-means implies between neighbouring clusters. Earlier runs used only the single best plane
per cluster. They were neutral. This study measures the **ceiling**: how far any
hyperplane/convex metadata could go, before investing in a query-time implementation.

## Bounds compared (all are valid lower bounds on min_{x in C} ||q-x||²)

| Score | Metadata | Meaning |
|---|---|---|
| `centroid` | none | Baseline order; certifies nothing. |
| `ball` | radius | Enclosing ball. |
| `knnK-single` | K slabs | Best single plane toward one of the K nearest centroids (previous `support` idea). |
| `knnK-joint4` | K slabs + K×K Gram | Joint polytope bound after 4 coordinate-ascent sweeps (deployable cost). |
| `knnK-joint` | K slabs + K×K Gram | Converged joint bound: distance to the intersection of all K slabs. |
| `hull` | all members | Distance to the convex hull of the members: the ceiling for **any** convex/hyperplane summary. |
| `nearest-member` | all members | Exact nearest-member distance (non-convex oracle). |

Why slabs from member extremes instead of Voronoi bisectors: stored leaf centroids are member
means after the final assignment, not the assignment centroids, and hierarchical assignment is
restricted to the coarse parent. Bisectors of stored centroids are therefore **not** valid
boundaries. The same normals with offsets from actual member extremes are always valid and at least as tight.

Joint bound: with [lo_e, hi_e] enclosing (x-q)·a_e, every weight vector w gives
||x-q||² >= S(w)²/(wᵀGw), S(w) = Σ min(w_e lo_e, w_e hi_e). So any iterate is safe. Coordinate
ascent on 2S(w) - wᵀGw converges to the exact distance from q to the slab polytope. Query
projections reuse full flat centroid distances: no extra 1024-D work.

## Metrics (first 50 calibration queries, K = 8 and 32)

- **Certified leaves/vectors**: leaves with bound <= reference (max SQ8 distance of the 10 supplied
  true IDs). An exact search using that bound alone cannot skip fewer.
- **Pure/hybrid vectors**: whole-leaf prefix cost to cover 10/10 truth IDs when ranking by bound
  (ties by centroid) or 0.5·centroid + 0.5·bound. Compare with `oracle_vectors_10`.
- Hull decisions use Gilbert iterations (128 max). Undecided leaves are reported separately.
  The hull ceiling uses double-accumulated dot products without padding. It is a diagnostic, not a pruning rule.
- `unsafe` must be zero (no bound exceeds a true neighbour's distance). `summarize.py` enforces it.

Same partitions as earlier studies: V0 fixed hierarchical k-means, 64×128 = 8192 leaves, 4
iterations, seed 42, decoded INT8 members. SIFT1M sanity and Qwen/MS MARCO 8.84M×1024.
The diagnostic runs with 32 threads over queries. The bound ns/leaf figures are indicative, not timed search.

## Decision rule

- If even `hull` certifies thousands of leaves, convex geometry cannot cut scans at 1024-D. Drop it.
- If `knn32-joint` or `hull` certifies close to `nearest-member`, build the joint bound into search.
- If `hull` is good but `knnK-joint` is not, the gap shows what richer (but still cheap) metadata could recover.
