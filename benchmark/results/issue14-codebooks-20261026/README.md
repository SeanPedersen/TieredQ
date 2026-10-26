# Issue #14: learned tier-shared codebook experiment

The initial experiment finds **no consistent recall improvement from replacing Gaussian codebooks with empirically fitted tier-shared tables**. Keep the production default. There are small gains in selected settings, particularly unrotated OpenAI, but these do not establish a general improvement or superiority over the rotated baseline.

## Primary results

Mean Recall@10 across seeds 0/1/2, 200 held-out queries per dataset. These are FP32 codec scores; the fast results are in the raw bundle. Budgets are bits per original dimension, not total bits.

| Dataset | Bits/dim | Rotate tiers | Gaussian recall | Learned recall | Change (percentage points) | Reconstruction MSE change |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| glove | 2 | off | 65.867% | 65.733% | -0.133 | -0.006% |
| glove | 2 | on | 65.967% | 66.217% | +0.250 | +0.002% |
| glove | 4 | off | 90.233% | 89.983% | -0.250 | -0.429% |
| glove | 4 | on | 90.567% | 90.400% | -0.167 | -0.245% |
| openai | 2 | off | 94.133% | 94.533% | +0.400 | +0.180% |
| openai | 2 | on | 94.417% | 94.167% | -0.250 | +0.319% |
| openai | 4 | off | 98.183% | 98.500% | +0.317 | +0.543% |
| openai | 4 | on | 98.483% | 98.333% | -0.150 | +0.968% |
| sift | 2 | off | 75.950% | 75.717% | -0.233 | +0.259% |
| sift | 2 | on | 75.600% | 75.600% | +0.000 | +0.616% |
| sift | 4 | off | 92.850% | 93.117% | +0.267 | -0.023% |
| sift | 4 | on | 93.250% | 92.850% | -0.400 | +0.130% |

At the default rotation setting and 4-bit budgets, learned tables reduce mean recall by 0.400 percentage points on SIFT, 0.167 on GloVe, and 0.150 on OpenAI. At 2 bits, rotated GloVe improves slightly, rotated SIFT is unchanged on average, and rotated OpenAI regresses. Disabling rotations changes which comparisons improve; it does not provide a consistent win across datasets and budgets. Lower scalar fitting distortion also does not guarantee lower reconstruction error after the full encoder or better ranking.

## Row-normalized fitting follow-up

The encoder normalizes standardized rows by their RMS, searches 49 scale-grid points, applies a radial factor, and stores a centroid correction. A simple marginal Lloyd fit does not jointly optimize those operations. A second variant fits RMS-normalized inputs with RMS-squared and coordinate-variance weights. This is a seed-0 diagnostic, not a three-seed result.

Default rotations, FP32 scoring:

| Dataset | Bits/dim | Gaussian recall | Learned recall | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| glove | 2 | 66.60% | 66.90% | +0.300 |
| glove | 4 | 90.75% | 90.90% | +0.150 |
| openai | 2 | 93.90% | 94.00% | +0.100 |
| openai | 4 | 98.50% | 98.20% | -0.300 |
| sift | 2 | 76.25% | 76.00% | -0.250 |
| sift | 4 | 93.35% | 93.35% | -0.000 |

Accounting for row normalization produces mixed results as well. It does not justify adopting learned tables. A future investigation could fit against the full encoder's reconstruction or score objective, using a larger calibration sample and more queries; these results do not rule out that approach or per-coordinate codebooks.

## Method and limits

- Full base datasets: SIFT 1,000,000 rows, GloVe-200 100,000 rows, OpenAI3-1536 100,000 rows. Use the existing cached base/query splits and exact top-10 truth.
- Train the common model on 8,192 uniformly sampled base rows, with 256 centroids. This is a screening experiment with a smaller training sample than the standard benchmark cap of 32,768.
- Fit one shared table per active bit tier, starting from the Gaussian levels. Minimize coordinate-variance-weighted squared error over standardized projected residuals; use up to one million seeded values per tier, up to 100 Lloyd steps, and a maximum-level-change threshold of 1e-6. Some fits reach the iteration cap; full convergence is not claimed.
- Within each pair, keep allocation, centroids, basis, means, sigma, codebook lengths, and serialized recipe size fixed. Only reconstruction values change. Production code and public API are unchanged by this experiment.
- Evaluate reconstruction on 2,048 additional base rows disjoint from training. SIFT uses the existing corrected query-metric L2 adapter; these rows are not necessarily held out from adapter calibration. Held-out queries and ground truth are never used for fitting.
- Evaluate both FP32 and fast serving, with rotations on/off and budgets 2/4. Primary study: three seeds; RMS-normalized follow-up: seed 0. Total: 192 evaluation rows / 96 paired scorer evaluations.
- Actual vector storage and model memory are identical within every Gaussian/learned pair. No extra codebook storage is needed for tier-shared tables.
- Timing uses eight threads, batches up to 32, one warmup, and three measured passes. Query transformation is outside the timed search. The shared host was not reserved; the RMS follow-up overlapped part of the primary study on a disjoint CPU set. Gaussian precedes learned. Timing noise is substantial, so no speedup or slowdown conclusion is supported.
- Only 200 queries were evaluated. Small changes are not claims of statistical significance. `returned_score_mse` uses each variant's own retrieved candidates and is not a controlled fixed-candidate score-error comparison.

## Reproduction and artifacts

The [experiment instructions](../../experiments/learned_codebooks/README.md) give build and run commands. Use the existing benchmark uv environment with Python 3.12, C++20, nanobind, LAPACK/BLAS, and OpenMP.

Remote workspace: `sean@hartmut:/home/sean/experiments/TieredQ-codebooks-20261026-AUmRXY`. The source was copied into this isolated workspace before compilation; its codec snapshot is authoritative even if the local checkout changes during the run. Raw `.bin` model recipes remain there for model-level validation.

Primary command (from remote workspace):

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 \
  /home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/.venv/bin/python \
  benchmark/experiments/learned_codebooks/run.py \
  --data /home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data \
  --native build --output results --bits 2 4 --seeds 0 1 2 --queries 200
```

For the follow-up, use `taskset -c 8-15`, `--output results-row-normalized --seeds 0 --fit-space row-normalized`, retaining the other flags. Both commands completed with exit code 0.

- [Primary raw measurements](standardized/results.jsonl), [summary](standardized/summary.json), and [manifest](standardized/manifest.json).
- [RMS-normalized raw measurements](row-normalized/results.jsonl), [summary](row-normalized/summary.json), and [manifest](row-normalized/manifest.json).
- [Validation and model recipe hashes](validation.json): pair invariants passed, plus byte-identical Gaussian model recipes across both studies for all 12 seed-0 settings.
- [Codec source and dataset provenance](provenance.json), [CPU metadata](cpu.json), and [codec source plus final experiment drivers](source-snapshot.tar.gz). The primary driver was later extended with the optional RMS fitting mode; its default standardized computation is unchanged.

The fitter also passed a synthetic skewed-distribution check: four-level reconstruction MSE fell from 0.25153 to 0.10889 on the calibration sample. Python compilation and whitespace checks passed. Native evaluation checked every model serialization round trip. These checks establish experiment integrity, not a production adoption recommendation.

Source archive SHA-256: `923bf617d59ab6fe1698a83bc008af7bba50eb699741e6166c52acf217bb67a8`.
