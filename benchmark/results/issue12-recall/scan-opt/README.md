# Issue #12: precision-preserving scan optimization

Archived bytes/vector values use the earlier memory counters; see the
[final-review accounting correction](../../../../docs/high-bit-recall.md#final-review-accounting-correction).
Use the [refreshed current storage table](../storage-refresh/README.md) for the eleven reported configurations.

Follow-up to the [recall fix](../README.md). Fast scans of models with 7/8-bit tiers remain bit-identical to the scalar codec; the work removes gathers, register spills and scheduling limits. [Design and analysis](../../../../docs/high-bit-recall.md#scan-throughput-optimization).

## Primary results

Full datasets, 1,000 held-out queries, top-10, seeds 0/1/2 for recall, seed 0 timing, **8 threads, batch 32**, `taskset -c 0-7`, native RaBitQ pruning/HACC enabled. Recall and measured bytes/vector are identical to the published recall-fix run in all 36 rows.

| Point | Recall@10 mean | Bytes/vector | Published QPS | Same-session rerun | Optimized QPS | RaBitQ QPS (recall) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT 8-bit | 0.98853 | 141.859 | 266 | 202 | **1,569** | 1,644 (0.98820) |
| GloVe 8-bit | 0.99257 | 221.486 | 1,779 | 1,806 | **7,026** | 10,996 (0.98850) |
| OpenAI 8-bit | 0.99827 | 1798.814 | 233 | 201 | **868** | 4,021 (0.99670) |
| SIFT 4-bit | 0.93383 | 137.833 | 277 | 282 | **1,771** | 1,763 (0.84503) |
| GloVe 4-bit | 0.90153 | 121.580 | 10,944 | 10,914 | 11,072 | 11,883 (0.88183) |
| OpenAI 4-bit | 0.98387 | 1798.694 | 233 | 235 | **1,038** | 4,048 (0.96737) |

GloVe 4-bit uses the unchanged narrow lookup kernel with 32 single-query tiles, so document segmentation does not apply at batch 32.

## Single query (batch 1, 8-bit, seed 0)

| Dataset | Before, 1 / 8 threads | Optimized, 1 / 8 threads | p50 ms, 8 threads | RaBitQ QPS |
| --- | ---: | ---: | ---: | ---: |
| SIFT | 10.6 / 10.6 | 25.0 / 184 | 93.3 → 5.5 | 222–226 |
| GloVe | 69.7 / 70.1 | 169 / 959 | 14.1 → 1.1 | 1,628–1,643 |
| OpenAI | 9.1 / 9.1 | 21.3 / 112 | 109 → 8.9 | 545 |

## Evidence

- `final/`, `final-single/`: delivered implementation. `final/preprocessing.csv` includes stage counters (`profile.py --stages`).
- `baseline-rerun/`, `baseline-single/`: unchanged corrected implementation, same session, separate workspace sharing the dataset cache.
- `intermediate-scan-only/`: scan optimization before batched query projection; its throughput is within run noise of `final/`.
- `as-run-sources.tar.gz`: exact scan-stage compiler inputs and harness, recovered during final review from the original remote workspace. SHA-256 `8d84670f36a585fa33b803f0a393acb194fd8bdcb5ce979c5b285a6d230a15e9` matches the previously recorded digest; all 43 source hashes match `final/meta.json`.
- `microbench/`: kernel experiments. `final-microbench.log` gives single-core baseline/new kernel cost, phase breakdown and query projection; `kbench.cpp`/`codes_new.h` contain the exploratory gather, register-table and decode-once variants; `abench.cpp` the accumulation-shape study; `ebench*.log` synthetic index-level A/B with result hashes; `qdump.cpp` the bitwise `query_weights` comparison; `e2e.log`/`final.log` run logs.

Single core (AVX-512, 256 centroids, all-8-bit), ns per document·dimension·query:

| Dimensions | Baseline 1 / 32 queries | Optimized 1 / 32 queries | Optimized AVX2 1 / 32 |
| ---: | ---: | ---: | ---: |
| 132 | 0.740 / 0.205 | 0.279 / 0.032 | 0.481 / 0.074 |
| 200 | 0.732 / 0.217 | 0.281 / 0.031 | 0.482 / 0.066 |
| 1536 | 0.733 / 0.210 | 0.303 / 0.034 | 0.474 / 0.055 |

The baseline AVX2 kernel costs 1.23–1.41 ns at one query and 0.36–0.37 ns at 32.

## Validation

ARM and x86 C++ suites pass, including new checks: 11-query tiles across full and partial register tiles of every SIMD backend against the scalar codec, a model mixing 8- down to 1-bit tiers with 100 centroids, and batched versus single `query_weights`. A deliberately wrong centroid table size fails the new test. Python API: 42 tests. Benchmark validation: 16 tests. Synthetic A/B result hashes match between baseline and optimized libraries for all six shapes.

## Reproduction

```sh
cd benchmark
./validate.sh
taskset -c 0-7 ./run.sh --bits 4 8 --threads 8 --batches 32 --seeds 0 1 2 --output output-new
taskset -c 0-7 uv run --locked --python 3.12 python profile.py output-new --stages
taskset -c 0-7 ./run.sh --bits 8 --threads 1 8 --batches 1 --seeds 0 --output output-new-single
```

Remote workspace: `sean@hartmut:/home/sean/experiments/TieredQ-issue12-recall` (`scan-opt/` holds the baseline workspace, microbenchmarks and drivers). Two unrelated processes floated across cores 0–7 during measurement; compare within the same session.
