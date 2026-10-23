# Bounded-memory exhaustive fast search

Native TieredQ flat exhaustive search on hartmut (Intel i9-7980XE, 18 physical cores), without IVF. Baseline is commit 6bad55c; optimized source is the accompanying uncommitted change.

All indexes contain 1,183,514 normalized GloVe200 vectors. Each measurement uses the same first 500 held-out queries, k=10 and supplied angular ground truth. Single queries use one thread pinned to CPU 2; batches of 500 use 18 OpenMP workers with OMP_PLACES=cores, OMP_PROC_BIND=close, OMP_DYNAMIC=FALSE and OPENBLAS_NUM_THREADS=1. No other thread counts were benchmarked.

Three independent processes per point, each with eight untimed warmup queries and three timed passes. The table reports median aggregate QPS across processes. The first baseline process precedes implementation; the other two baseline/optimized pairs alternate execution order. Index loading, fast-cache construction and evaluation are outside timing. Timed calls include query preparation, all document scoring, top-k selection and search scratch allocation.

| Model/cache | Threads | Baseline QPS | Optimized QPS | Speedup | Recall@10 |
|---|---:|---:|---:|---:|---:|
| 800 bits, 200 × 4-bit | 1 | 88.0 | 92.3 | 1.05x | 0.8992 |
| 800 bits, 200 × 4-bit | 18 | 818.6 | 1360.7 | 1.66x | 0.8992 |
| 400 bits, mixed 1/2/3-bit | 1 | 153.3 | 161.1 | 1.05x | 0.6674 |
| 400 bits, mixed 1/2/3-bit | 18 | 1092.5 | 2420.5 | 2.22x | 0.6674 |
| 1,200 bits, 200 × 6-bit | 1 | 56.7 | 58.2 | 1.03x | 0.9638 |
| 1,200 bits, 200 × 6-bit | 18 | 1048.2 | 2205.4 | 2.10x | 0.9638 |

## Change

Fast search now scans bounded document tiles and performs incremental top-k selection before scratch is reused. A tile holds at most 4,096 document scores per query, with existing platform query reuse: one query for x86 byte lookup, up to 16 for x86 expanded levels and up to four on ARM. A dedicated single-query x86 specialization avoids carrying the batch accumulator array through the serial dot-product loop.

Persistent cache representations, numerical operations, query quantization and result ordering are preserved. The full queries-by-documents score matrix and the single-query full-document score buffer are removed. Shared selection and exception-safe query preparation moved out of the main implementation into tieredq_search.h; bounded scheduling lives in tieredq_search_fast.h. Worker errors propagate to the caller after leaving OpenMP.

At k=10, score scratch is bounded by workers × query tile × 4,096 floats, rather than queries × documents. For 18 workers this is at most 4.5 MiB for the expanded x86 scan, plus query state, top-k heaps and result arrays. The original 500-query score matrix alone was 2,367,028,000 bytes (2.20 GiB). Persistent index/cache memory and load-time allocations still scale with index size.

## Memory

| Model/cache | Baseline batch peak RSS MiB | Optimized batch peak RSS MiB |
|---|---:|---:|
| 800 bits, 200 × 4-bit | 2507.1 | 367.5 |
| 400 bits, mixed 1/2/3-bit | 2395.2 | 199.6 |
| 1,200 bits, 200 × 6-bit | 2676.0 | 592.8 |

Peak RSS is measured per process with wait4 and includes loading, cache construction and warmup. It is not just steady-state search scratch.

## Validation

For every model, IDs and float-score payloads are byte-identical across baseline/optimized, one core/18 threads and all three independent processes. Recall is unchanged. New tests cross both document and query tiles, cover zero-query ties and k near index size, validate wide dots against an independent scalar numerical-level calculation, and guard partial/tail stores. On ARM and x86: 25 Python tests and 2/2 C++ tests pass.

## Limits and next step

A separate one-core cycles profile of the final expanded-level path (200 queries, three passes, including startup/cache construction) attributes 84.99% of sampled cycles to scan_fast_avx512_tile_impl<1>, 8.53% to cache construction, and about 3.7% to loading. This is a whole-process profile, not a search-only percentage, but it identifies full numerical scoring as the dominant remaining single-core cost. Raw profile and text are in results/linear-expanded-profile.{data,txt} on hartmut.

One shared host and three model configurations; background services remain running. The single-core differences are small and should be treated cautiously. This change improves memory traffic and scheduling, not the number of numerical scores computed. Every document is still fully scored. Progressive screening is the next separate experiment; no pruning or recall tradeoff was introduced here.

An early fixture label called the 800-bit model "wide", but its model metadata confirms all coordinates are four-bit: it takes the compact byte-lookup path. The raw filenames retain that alias; structured results use lookup4. The actual expanded-level fixture is the 1,200-bit model, labeled expanded6. An accumulator-unrolling experiment was initially run on the lookup fixtures, so those runs provide no evidence about expanded-dot performance and are excluded from the table.

## Reproduction and artifacts

Everything is preserved under ~/experiments/TieredQ-linear-fast-20261023 on hartmut: baseline/ and optimized/ source snapshots and binaries, scripts/throughput.cpp, run.py, repeat.py, report.py, per-run JSON/IDs/scores/usage records and logs. The harness uses C++20, OpenMP and the normal CMake-built library, compiled with -O3 -ffp-contract=off. GCC 16.2.1, Release build.

After building both snapshots and linking the harness: python3 -u scripts/run.py baseline optimized; python3 -u scripts/repeat.py; python3 scripts/report.py. Existing raw filenames use wide=lookup4, compact=lookup2 and expanded=expanded6.

Local structured evidence: [linear-fast-scan-results.json](benchmarks/linear-fast-scan-results.json).
