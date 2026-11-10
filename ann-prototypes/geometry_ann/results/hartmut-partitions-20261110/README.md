# Finer partitions with multi-threaded k-means

Status: stopped after 32K calibration (`results/qwen-32k/calibration.csv` kept). The hierarchical assignment turned out to be the scan-count bottleneck, so this was superseded by `../hartmut-assignment-20261110/`.
Remote root: `/home/sean/experiments/TieredQ-issue44-partitions-20261110` (`status.json`, `job.exitcode`).
Replaces the partition phase of `../hartmut-ranking-20261110/`, which was stopped during its single-threaded 32K build.

## Protocol

- k-means assignment uses 32 threads (`--build-threads 32`). Assignments are computed in parallel; centroid
  sums and group order stay serial, so partitions are bit-identical to single-threaded builds.
  CTest checks this on fixtures. **Gate:** SIFT1M is built with 1 and with 32 threads; all deterministic
  search columns (recall, vectors, metadata, leaves, graph distances) must match before Qwen runs.
- After the build, `--search-cpu 2` pins the process to one core: every calibration and held-out
  measurement is single-core, as before.
- Build times here are **32-thread** k-means (graph construction, geometry and packing stay serial).
  They are not comparable with single-threaded Faiss. Final candidates get a separate like-for-like build timing.

Partitions: 128×256 = 32,768 and 256×256 = 65,536 leaves, V0 fixed hierarchical k-means, 4 iterations, training
sample 8192 (32 per fine centroid, as in the 4096/128 baseline), graph degree 8. Settings: flat with all
leaves, and graph with 2,048/4,096/8,192/16,384 metadata evaluations × caps 35K/88K/177K/442K. Calibration
(250 queries) selects the fastest settings reaching 95/99/99.5%. Held-out: 750 queries × 3 repeats.

Baseline (8,192 leaves): 95.57% at 27.3 ms flat, 95.04% at 25.8 ms graph, best 97.65% at 61.5 ms.
Output: `REPORT-partitions.md`, `summary-partitions.json`.
