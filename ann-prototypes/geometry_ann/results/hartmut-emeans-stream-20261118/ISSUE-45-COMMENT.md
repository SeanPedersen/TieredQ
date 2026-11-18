**E-means is now the automatic list training** (streamed batches, Lloyd stays available as `TieredQCoarseTraining.lloyd(lists)`).

The previous attempt (64 rows/list x 2.5 epochs from an FP32 sample) gained recall at 8.84M / 64K lists but built 15%
slower at +7.7 GB peak RSS. Two fixes:
- **Streaming:** e-means reads each batch through the build's row loader (the SQ8 memmap) in 32K-row chunks and keeps
  per-list sums, never a sample: 1.6 GB while training instead of Lloyd's 11.6 GB sample, so it can train on every row.
- **Grouped restricted assignment:** centroids are copied into super-member order once per batch instead of once per
  256-row work item. A new `emeans_probes` (4) lets batch rows probe fewer supers than the final assignment (16).

**8.84M x 1024 (Qwen/MS MARCO), 64K lists, nested low-RAM build, 2 seeds** — e-means 2 epochs over all rows vs fast Lloyd:

| | Lloyd | E-means |
|---|---:|---:|
| Build (32 threads, tier write excluded) | 196-203 s | **175-179 s** |
| Peak build RSS | 20.9-21.1 GB | **15.4 GB** |
| Serving RAM | 1.58-1.59 GB | 1.58-1.59 GB |
| Recall@10 at nprobe 96 / 256 / 1024 (seed 0) | 0.938 / 0.962 / 0.980 | **0.947 / 0.970 / 0.984** |
| Recall@10 at nprobe 96 / 256 / 1024 (seed 7) | 0.941 / 0.966 / 0.984 | **0.945 / 0.968 / 0.985** |

Per-query latency is 0-5% higher at equal nprobe; at equal p50 e-means stays ahead on both seeds.

**1M TieredQIVFIndex (2,048 and 8,192 lists, 2 seeds):** 8-18% faster builds, equal or higher recall@10 per nprobe.

**Defaults:** from 8,192 lists, 2 epochs over 256 rows per list (all rows at 64K), 256 supers, 4 probes in training
and 16 in the final assignment; below that, exact e-means 2.5 epochs over 256 rows per list. `TieredQIVFFlatIndex`
trains by e-means (256 rows per list x 2.5 epochs) by default.

Study: `ann-prototypes/geometry_ann/results/hartmut-emeans-stream-20261118/README.md`.
