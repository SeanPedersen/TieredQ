# Build peak: drop the duplicate training residual

Issue #4 follow-up (task 4c). TieredQ's build peak was about 1 GB for OpenAI at 100K vectors. This bundle shows where that peak comes from and removes the largest avoidable part without changing a single saved byte.

Hartmut (Intel i9-7980XE, AVX-512), CPUs 8–15, GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. Shared host: the issue #4 equal-memory job ran on CPUs 0–7 at the same time, so build times carry a few percent of noise. Training sample: 32,768 rows (the adapter's `training_limit`), k = 256, seed 0.

## Profile (before)

`benchmark/profile_build_memory.py` samples VmRSS every 1 ms during `codec_native.Index(...)`, with `DV_TIEREDQ_PROFILE` stage timings for alignment. The timeline is in [openai-1-timeline.json](openai-1-timeline.json) (every 10th sample). Peak growth over the loaded corpus is 982 MiB for OpenAI 1-bit (d = 1,536).

| Stage (cumulative s) | RSS growth | Held |
|---|---:|---|
| k-means (0–1.0) | 197 MiB | adapter's FP32 training sample copy, 32,768 × 1,536 × 4 B = 192 MiB |
| residual (1.0–2.0) | 965 MiB | + `residual` f64 rows × d (384 MiB) + `transposed` f64 d × rows (384 MiB) |
| covariance (2.0–8.1) | **984 MiB peak** | + covariance d × d f64 (18 MiB) |
| eigen (8.1–13.6) | 638 MiB | `transposed` freed |
| return from `train` | 245 MiB | `residual` freed (function scope) |
| encode + pack | 54 MiB at end | the index |

Encoding adds almost nothing. The default build has no `TIEREDQ_BLAS`, so `linalg.syrk` is null and training takes the fallback branch. That branch materialised `residual` only to compute the mean and fill `transposed`. `residual` itself is read again only when `DV_TIEREDQ_REFERENCE_STATS` is set.

## Change

`src/tieredq.cpp`: on the spectral-stats path, compute the mean from the per-element differences, then write `transposed` by recomputing the same differences instead of keeping a rows × d copy. The arithmetic and summation order are unchanged. The reference-stats path still keeps `residual`.

## Result

[model-hash.jsonl](model-hash.jsonl): "before" is the tree without this change. It contains the separate lookup-projection change, which does not touch training.

| Build | Saved index SHA-256 (prefix) | Build peak before → after |
|---|---|---:|
| OpenAI 1-bit | `b95f2f39dda75027` = same | 982 → 598 MiB (−39%) |
| Gemini 2-bit | `88f3e396f7692885` = same | 479 → 287 MiB (−40%) |
| GloVe 4-bit | `5302c775c8f6f8f7` = same | 124 → 74 MiB (−40%) |

Saved indexes are byte-identical, so search results and recall are unchanged. Build time is unchanged within noise. In the interleaved OpenAI 1-bit builds ([build-ab.jsonl](build-ab.jsonl)), before measured 14.99 and 15.18 s and after measured 14.50, 15.44 and 14.82 s. One "before" run aborted on a VmHWM reset race; the traceback is kept in the log.

The remaining ~600 MiB is the FP32 sample (192 MiB, held by the benchmark adapter) and the f64 `transposed` matrix (384 MiB). Removing either needs a different covariance summation order (row-chunked accumulation) or a smaller or float training sample. Both change the model, so they are out of scope for an ID-identical change.

Tests: `test_tieredq` (also under `DV_TIEREDQ_REFERENCE_STATS=1`), `test_training_rng`, `uv run --group dev pytest` (57 passed) and `benchmark/validate.py` (18 tests) all pass.
