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
- Python tests: `test_l2_metric_scores_negative_squared_distances` fails on Hartmut's reference BLAS with and without
  these changes (one estimate 0.85 off a zero distance); all other tests pass.
