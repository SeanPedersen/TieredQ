# Issue #19.3: build peak RSS (2026-10-31)

Peak RSS growth per build stage, in bits per indexed vector, from
[`build_peak.cpp`](../../experiments/issue19_build/build_peak.cpp) (synthetic rows with a decaying spectrum, seeded
32,768-row training sample, 8 threads; Hartmut, reference BLAS/LAPACK). "Old" is the library at `de91723^`
(`a25ecc1`); "new" adds:
- training reads rows in place: cosine rows are normalized on access instead of copied (bit-identical models), and
  `TieredQ::train_sampled` trains on rows named by id without a gathered copy (the same model);
- bulk `TieredQIndex::add` into an empty fast index encodes 16,384 rows at a time into storage reserved once;
  `TieredQIVFIndex::build` fills its fast storage in list order 8,192 documents at a time (bit-identical storage).

100K x 1,536, cosine, flat fast index:

| Stage | Old, 1 bit/dim | New, 1 bit/dim | Old, 4 bits/dim | New, 4 bits/dim |
|---|---:|---:|---:|---:|
| caller's training sample copy | 16,171 | 75 (`train_sampled`) | 16,171 | 75 |
| training peak | +21,711 | +5,823 | +22,711 | +6,830 |
| add peak | +5,231 | +3,305 | +15,327 | +11,058 |
| index | 1,589 | 1,589 | 6,185 | 6,185 |
| **build peak (largest stage)** | **~38K** | **~6K (-84%)** | **~39K** | **~11K (-72%)** |

1M x 768, cosine, 4 bits/dim, flat add peak (local i7-8550U, MKL): 6,334 -> 3,427 bits/vector from chunked encoding;
the training stage is ~1K there (the fixed training cost is spread over 1M rows).

- The fixed training cost no longer holds the lifted 32,768-row sample (201 MB at d = 1,536); what remains is the
  FP64 covariance and eigendecomposition workspace (about 60 MB at d = 1,536) and k-means state.
- The remaining add overhead at 100K is the encoder's fixed state and per-thread buffers plus one chunk of codes.
- L2 models still lift their training sample once (the lift is a matrix product); only SIFT (d = 128) uses it here.
- The benchmark adapter and the Python binding now call `train_sampled`; the binding trains on a seeded
  32,768-row sample when the first `add()` is larger (models of smaller first batches are unchanged).
- Python tests: `test_l2_metric_scores_negative_squared_distances` failed on Hartmut with and without these changes.
  Its tolerance (5% of the mean top-3 distance) passed for 35% of seeds while the ranking was right for all; L2
  score errors scale with the squared row norms (max 0.06 x mean |x|^2 over 200 seeds at 4 bits/dim), and the test
  now bounds them by 0.1 x mean |x|^2 (`f26b89f`). All 61 Python tests pass.

## Build time (training)
Same driver (`TRAIN_ROWS` sets the sample), 100K x 1,536 cosine, 1 bit/dim, 8 threads alone on CPUs 0-7. Hartmut's
system BLAS/LAPACK is the unoptimized reference implementation (Arch `blas` / `lapack` 3.12), which every earlier
TieredQ build time on Hartmut used. `TIEREDQ_BLAS=/usr/lib/libopenblas.so` loads OpenBLAS at run time for the
eigendecomposition and the covariance, which then runs as BLAS `syrk` over 1,024-row chunks (no rows x d copy).

| Training | k-means | covariance | eigen | total | training peak |
|---|---:|---:|---:|---:|---:|
| default (reference LAPACK, own covariance loops) | 1.9 s | 7.1 s | 6.1 s | 15.5 s | 5.8K |
| `TIEREDQ_BLAS` OpenBLAS, eigen only | 1.3 s | 7.6 s | 0.35 s | 9.7 s | 7.5K |
| `TIEREDQ_BLAS` OpenBLAS (syrk + eigen) | 1.8 s | 0.38 s | 0.26 s | 2.8 s | 7.8K |
| same, 16,384 training rows | 0.9 s | 0.16 s | 0.23 s | 1.5 s | 7.8K |

At 4 bits/dim the OpenBLAS training takes 3.7 s and the add 4.4 s. The extra ~2K bits/vector of training peak with
OpenBLAS are its thread buffers (~25 MB), not the covariance chunk. 16K training rows reach the recall of 32K on
OpenAI-1M and MS MARCO-1M (#19.2 sweep).

## Flat build vs RaBitQ and TQ+ (OpenAI, real data)
`build_peak.cpp` with `DATA=` the OpenAI base and `CODE_BITS=` the #4 budgets matched to RaBitQ's steady RAM at
1 / 2 / 4 bits/dim, cosine, seeded 32,768-row `train_sampled`, chunked add, 8 threads on CPUs 0-7 (an untimed recall
job used CPUs 8-31 at the same time). Reference and earlier TieredQ build times are the #4 timing cells (seed 0, run
alone; [report](../../../docs/benchmark-report.md#build-cost)), not re-measured. Build = train + add, seconds.

| Set, bits/dim | TieredQ #4 (reference LAPACK) | TieredQ now, reference LAPACK | TieredQ now, OpenBLAS | RaBitQ | Qdrant TQ+ |
|---|---:|---:|---:|---:|---:|
| 100K, 1 | 12.4 | 13.8 | **3.3** | 0.1 | 3.2 |
| 100K, 2 | 12.9 | 14.1 | **3.5** | 2.5 | 4.3 |
| 100K, 4 | 13.6 | 15.0 | **4.7** | 4.1 | 5.8 |
| 1M, 1 | 21.4 | 25.7 | **15.4** | 1.0 | 31.8 |
| 1M, 2 | 24.1 | – | **17.5** | 24.9 | 40.5 |
| 1M, 4 | 34.0 | – | **28.5** | 41.9 | 51.7 |

Build peak RSS growth at 1M (bits/vector), now vs #4: 1,677 vs ~5,000 (index 1,593), 3,343 vs ~8,300 (index
3,176), 6,573 vs ~14,400 (index 6,216): within 2-6% of the index, as RaBitQ (1,647 / 3,250 / 6,322) and half of TQ+
(3,254 / 6,321 / 12,473).

- With an optimized LAPACK, TieredQ builds 1.4-1.5x faster than RaBitQ at 2 and 4 bits/dim on 1M and within
  1.1-1.4x at 100K; RaBitQ's 1-bit build (sign codes, no extra bits) stays 15-33x faster.
- Training is now a fixed ~2 s at d = 1,536; at 1M the encode (13-26 s) is the remaining build cost.

## Recall with OpenBLAS (default build)
The OpenBLAS training path (chunked `syrk` covariance, OpenBLAS `syevd`) against the reference-LAPACK cells
`TRAIN_ROWS=32768` of the [#19.2 sweep](../issue19-drift-20261031/README.md): IVF 256 lists, prefixes 768 / 1024,
nprobe 32 / 128, k' 20 / 50 / 100, INT8 and `tieredq-refinement:4` rerank, OpenAI-1M and MS MARCO-1M real queries.
The eval binary predates the budget-dependent α (`07b5d3f`), so the comparison isolates the BLAS change. Raw TSVs:
[`blascheck/`](blascheck/) (job `job_blascheck.sh`).

| Set, prefix, nprobe 128 | candidates@k'20 | INT8 R@10 @k'100 | refinement:4 R@10 @k'100 |
|---|---:|---:|---:|
| OpenAI-1M, 768 | 0.9562 vs 0.9564 | 0.9944 vs 0.9944 | 0.9845 vs 0.9829 |
| OpenAI-1M, 1024 | 0.9781 vs 0.9779 | 0.9948 vs 0.9949 | 0.9855 vs 0.9837 |
| MS MARCO real, 768 | 0.9462 vs 0.9445 | 0.9825 vs 0.9822 | 0.9726 vs 0.9710 |
| MS MARCO real, 1024 | 0.9761 vs 0.9767 | 0.9844 vs 0.9844 | 0.9743 vs 0.9753 |

OpenBLAS vs reference LAPACK. Over all 96 paired values (48 cells x candidates / reranked R@10) the difference is
at most 0.24 pt (mean +0.02 pt); candidate recall is within 0.17 pt. The models differ only by floating-point
summation order, and the largest gaps (refinement:4, +0.2 pt on OpenAI, -0.1 pt on MS MARCO 1024) change sign
between sets.
