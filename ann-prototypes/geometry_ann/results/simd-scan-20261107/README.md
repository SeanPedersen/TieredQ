# SIMD scan filter control (#44)

Linux x86_64, Intel i7-8550U, GCC 16.2.1, CMake Release, single-threaded.
Three alternating reference/fast process runs; no CPU pinning or frequency locking.
Both builds pass CTest. The AArch64 implementation was not executed here.

Build as in the prototype README, using `/tmp/geometry-fast` for the default
and `/tmp/geometry-reference` with `-DGEOMETRY_FAST_SCAN=OFF`.
Run each executable three times with:

```sh
geometry_bench --n 20000 --dim 128 --nq 100 --variant V1 --split kmeans \
  --coarse 16 --leaves 256 --target-size 64 --leaf-scans 256 \
  --vectors 1000 --ordering centroid
```

| Kernel | p50 across three runs (us) | Median p50 (us) |
| --- | --- | --- |
| Original double | 71.410, 69.959, 70.209 | 70.209 |
| FP32 filter + double finalists | 47.693, 45.953, 50.800 | 47.693 |

Median p50 improves by 32% (1.47x). All runs have recall@10 = 1,
833 mean vector scans, 125.72 metadata evaluations, and 112.27 pruned nodes.
This easy clustered synthetic dataset verifies the optimization and harness;
it does not establish a gain over Faiss or address partition quality. No
SIFT/Instagram benchmark exports were available in this workspace. Rerun
held-out real-data settings and Faiss on the same machine before comparing
with the older measurements from issue #44.

The filter follows the production IVF code's runtime SIMD dispatch pattern,
but stays standalone and does not import its quantized scan or panel layout.
It uses a deliberately loose dimension-dependent FP32 rounding allowance;
only rejected candidates avoid double scoring. Training and metadata remain
unchanged. Tests cover exact complete-search results, duplicates, SIMD tails,
near-equal coordinates, overflow, and underflow.
