# Nested IVF fine suffixes: local Gemini comparison

At an 8-bit/dimension total budget, enabling fine suffix allocation reduces full-probe top-10 misses from 56 to 24 (57.1%) with essentially unchanged search time. The 4-bit/dimension control produces the same allocation and recall.

## Method

- Apple M1, macOS 27.0, ARM64; one OpenMP thread and one BLAS thread.
- 99,000 normalized Gemini-768 document vectors and 1,000 held-out document queries. Ground truth is the cached exhaustive FP32 inner-product top 10; source revision, split and hashes are in `dataset-manifest.json`.
- 256 IVF lists, 2-bit/dimension prefix, total budgets of 4 and 8 bits/dimension. RAM tier with `prefix_and_suffix` rows; inner-product metric.
- Each pair uses identical seeded training inputs. The runner asserts equality of the complete prefix candidate-ID arrays for every probe/depth setting.
- Query batches of 32 (last batch 8), one warmup pass per variant, then five alternating-order passes. Times below are median amortized milliseconds per query, including query projection, IVF scan, tier reads and rescoring.
- Build/encode durations in the raw JSON are informational: validation jobs overlapped training. Search measurements for the 8-bit comparison ran after validation finished.

## Search results

| Total budget (bits/dim) | nprobe | Candidates | Recall@10 off → on | Misses off → on | ms/query off → on | Time change |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 100 | 95.51% → 95.51% | 449 → 449 | 0.936 → 0.973 | +4.02% |
| 4 | 32 | 400 | 95.51% → 95.51% | 449 → 449 | 1.264 → 1.248 | -1.26% |
| 4 | 256 | 100 | 97.92% → 97.92% | 208 → 208 | 2.540 → 2.516 | -0.92% |
| 4 | 256 | 400 | 97.92% → 97.92% | 208 → 208 | 2.971 → 2.959 | -0.41% |
| 8 | 32 | 100 | 96.77% → 96.96% | 323 → 304 | 0.930 → 0.927 | -0.29% |
| 8 | 32 | 400 | 96.77% → 96.96% | 323 → 304 | 1.292 → 1.289 | -0.23% |
| 8 | 256 | 100 | 99.44% → 99.76% | 56 → 24 | 2.480 → 2.501 | +0.83% |
| 8 | 256 | 400 | 99.44% → 99.76% | 56 → 24 | 2.964 → 2.988 | +0.82% |

The 8-bit full-probe run gains 0.32 percentage points of recall. At 32 probes, recall improves by 0.19 percentage points; list selection limits attainable recall to 97.15%. The remaining rerank misses within that candidate set fall from 38 to 19. Increasing the candidate depth from 100 to 400 changes neither recall nor the candidate ceiling in this experiment.

The latency distributions overlap. At the 8-bit full-probe/100-candidate setting, the five baseline samples range from 2.461–2.538 ms/query and the fine samples from 2.475–2.521 ms/query; the median increase is 0.83%. This run does not establish a speed improvement or a material slowdown.

## Realized storage

| Budget | Fine suffixes | Combined bits/vector | Widest combined code | Tier bytes/vector |
|---|---|---:|---:|---:|
| 4 | off | 3032 | 8 | 388 |
| 4 | on | 3032 | 8 | 388 |
| 8 | off | 6144 | 8 | 777 |
| 8 | on | 6135 | 11 | 776 |

The prefix remains 1,536 code bits/vector, and IVF storage remains 218.69 bytes/vector in all cells. Tier rows include a copy of the prefix and metadata. At 8 bits/dimension, the fine model uses one fewer tier byte/vector, so its recall gain does not come from additional row storage. Different total code sizes result from rounding gains across fixed prefix blocks.

## Implementation and validation

`TieredQNestedIVFIndex.build(..., fine_tiers=True, max_tier_bits=12)` enables wider IP/cosine suffix allocation. The default remains off. `max_refinement_bits` exposes the realized maximum combined code width. C++ callers pass `fine_tiers` and `max_tier_bits` after the seed to `TieredQRefinement::train`.

The refinement trainer now separates model/layout construction from fitting. It reuses the existing fine-tier allocation, retains prefix-cell Lloyd fitting, and supports packed suffix fields through 12 bits. The AVX-512 extraction fallback uses masked gathers for wide fields and partial segments. Wider models use refinement format 2; format 1 stays readable and is still emitted for narrow models.

RAM, file-tier, external rescore and persistence parity are covered by tests, including both row layouts and IP/cosine. The host runs the scalar refinement scorer; the x86/AVX-512 path was syntax-checked with an x86_64 target, but its runtime speed was not measured. SSD latency and independently embedded retrieval queries were not benchmarked.

Final validation: 200 Python tests passed (3 skipped), both CTest targets passed, and the x86_64 syntax check passed. Wide-field regression coverage includes suffix fields spanning three bytes.

## Reproduction

```sh
uv pip install --python .venv/bin/python -e . --no-deps
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u benchmark/experiments/nested_fine/run.py \
  /path/to/gemini-prepared.npz benchmark/results/nested-fine-local-20261105/gemini.json
```

Raw data: `gemini.json`. Dataset provenance: `dataset-manifest.json`. Relevant source hashes: `source-hashes.json`.
