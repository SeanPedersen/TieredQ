# Corrected intermediate result: SIFT1M, 1 bit/dimension

This is one budget on one dataset, not the complete issue #4 matrix. Earlier IVF-based / size-optimized runs are invalidated and are not included here.

Both codecs use the same 1,000,000 original SIFT base rows, 1,000 queries and exact FP32 L2 truth. Recall is mean ± population standard deviation over seeds 0/1/2. Timing uses seed 0, three warmup queries and three repeats. QPS below uses batch 32; latency uses batch 1. CPU affinity is eight physical cores, 0–7.

| Codec | Effective bytes/vector | Recall@10 | QPS, 1 thread | QPS, 8 threads | Single-query p50 / p95, 1 thread | Build seconds |
|---|---:|---:|---:|---:|---:|---:|
| TieredQ fast-native | 26.580 | 0.3635 ± 0.0006 | 391 | 2,750 | 2.506 / 2.826 ms | 0.939 |
| RaBitQ direct-flat, HACC | 28.001 | 0.2495 ± 0.0047 | 302 | 2,257 | 3.326 / 3.506 ms | 0.116 |

At these measured points TieredQ has higher recall, 5.1% lower effective storage and about 1.22× higher eight-thread batch throughput. RaBitQ builds about 8.1× faster. This does not establish a winner at higher recall, higher budgets or other datasets. The earlier reported 4.4× throughput advantage was invalid.

## Fairness checks

- Both codec translation units' effective optimization is audited as `-O3`; nanobind uses `NOMINSIZE`, with `-march=native -ffp-contract=off` shared by both codecs.
- RaBitQ uses official native packed encoding/scoring kernels, default high-accuracy lookup and native extra-bit bounds. The measured adapter constructs no IVF container, posting lists, stored ID array or ANN routing. IDs are row positions.
- RaBitQ's rankings match an independent official one-list reference caller for all four budgets, L2/IP and 128/200 dimensions. The reference is outside benchmark timing and storage.
- Storage growth is tested against native code/factor bytes, so an ID array cannot silently enter accounting.
- SIFT uses native 128-dimensional RaBitQ L2. TieredQ's required L2-to-IP transformation is included in its serving latency and extra-coordinate storage, with the same original-dimension code budget.
- Dataset source SHA256, split seed and base/query hashes are in `data-manifest.json`; exact compiler commands, commits and source hashes are in `meta.json`.

Core selected-matrix time was 65.25 seconds, excluding download, ground-truth preparation and native compilation. Separate preprocessing diagnostics took 21.34 seconds. All 24 selected matrix rows completed; no full matrix was started afterward.

## Reproduce

Use the TieredQ source commit recorded in `meta.json`, on Linux x86-64 with the prerequisites in `../../README.md`:

```sh
cd benchmark
./validate.sh
taskset -c 0-7 ./run.sh --datasets sift --bits 1 --output output-corrected-sift-1bit
```

All required executable code is in `benchmark/`. The dependency lock, upstream revision and dataset catalogue are pinned. `as-run-harness.tar.gz` preserves the exact benchmark sources used for this measurement, before the build workflow was factored into the shared validation helper. Native codec sources and kernels are unchanged by that orchestration refactor.

The CSV, metadata, summary and separate preprocessing measurements are retained alongside the three figures. Timings can vary with machine load and CPU frequency; input data and seed choices remain fixed.
