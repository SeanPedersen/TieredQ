# Ranking study: statistical leaf scores and finer partitions

Status: A complete (statistical scores lose to centroid order on SIFT1M and Qwen); B stopped during the single-threaded 32K build and superseded by `../hartmut-partitions-20261110/` and `../hartmut-assignment-20261110/`.
Remote root: `/home/sean/experiments/TieredQ-issue44-ranking-20261110` (`status.json`, `job.exitcode`).

Context: the cell study (`../hartmut-cells-20261110/`) showed that geometric bounds cannot cut scans on
Qwen 8.84M×1024. The gap is ranking: centroid order needs 130K vectors for 10/10 versus 8.6K for the oracle.

## A. Statistical leaf scores (offline, `cell_ceiling --slabs 0 --hull 0`)

Rank leaves by an estimate of the expected nearest-member squared distance (`rank_scores.{h,cpp}`):

    score = D + r̄² − 2β·√D·s·√(2 ln n)

Here D is the centroid distance, r̄² the mean squared member radius, and n the leaf size. s² is the member variance along
the query direction: isotropic r̄²/d (`evt-iso`), or a shared 8-axis PCA basis with per-leaf axis
variances plus an isotropic residual (`evt-pca8`, O(8) per leaf after 8 query projections). β=0 is
`D + r̄²`. These are estimates, not bounds, so they certify nothing. The key metric is coverage: the mean
fraction of the 10 true IDs scanned when whole leaves are taken in score order under the timed-search
vector caps (leaves that no longer fit are skipped). Same 64×128 V0 partitions, 250 calibration queries.

## B. Finer partitions (real search, `graph_bench`)

128×256 = 32,768 and 256×256 = 65,536 leaves, centroid order, graph degree 8, training sample 8192
(the same 32 samples per fine centroid as the 4096/128 baseline). Settings: flat with all leaves, and graph
with 2,048/4,096/8,192/16,384 metadata evaluations × caps 35K/88K/177K/442K. Calibration selects the
fastest settings reaching 95/99/99.5%; held-out runs 750 queries × 3 repeats on CPU 2. Compared with the
8,192-leaf baseline: 95.57% at 27.3 ms (flat), 95.04% at 25.8 ms (graph), and a best of 97.65% at 61.5 ms.

Outputs: `statistical/REPORT.md` (A) and `REPORT-partitions.md` (B), each with a JSON summary.
