# Learned tier-shared codebook ablation

This isolated experiment replaces only reconstruction tables in a trained codec's serialized recipe. Production training, allocation, encoding, scoring, and model formats remain unchanged. The bridge uses public codec/index APIs.

Build with the benchmark's Python 3.12 environment (dependencies managed by uv):

```sh
cmake -S benchmark/experiments/learned_codebooks -B build-codebooks \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/benchmark/.venv/bin/python"
cmake --build build-codebooks -j8
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 benchmark/.venv/bin/python \
  benchmark/experiments/learned_codebooks/run.py \
  --data /path/to/existing/benchmark/data --native build-codebooks \
  --output /path/to/new/results --bits 2 4 --seeds 0 1 2 --queries 200 \
  --fit-space standardized
benchmark/.venv/bin/python benchmark/experiments/learned_codebooks/summarize.py \
  /path/to/new/results
```

The initial study uses full cached base datasets, 200 held-out queries, 8,192 uniformly sampled training rows, 256 centroids, 2/4 bits per original dimension, seeds 0/1/2, rotations on/off, and FP32/fast serving. A separate 2,048-row base sample, disjoint from training, evaluates reconstruction. SIFT uses the benchmark's query-metric L2 transform; its calibration uses base data, including potentially reconstruction evaluation rows. These are held out from codec/codebook fitting, not from adapter calibration.

Fitting starts from Gaussian reconstruction levels and runs at most 100 weighted Lloyd iterations, stopping when maximum level movement is below 1e-6. Each bit tier shares one sorted codebook. Its flattened standardized training values are weighted by coordinate variance, so the objective represents unstandardized coordinate squared error. Tiers exceeding one million values use a seeded uniform sample. Empty cells retain their previous level. Iteration counts are recorded; a capped fit is not claimed to have converged.

Run `--fit-space row-normalized` into a second fresh directory to account for the encoder's row RMS normalization. This divides standardized rows by RMS and weights samples by RMS squared as well as coordinate variance. It still does not jointly optimize the encoder's scale-grid selection, final radial factor, or centroid correction. Neither fitting objective guarantees improved score error or recall.

Each Gaussian/learned pair has byte-identical model metadata, allocation, centroids, basis, mean, and sigma; table lengths and serialized recipe size also match. The summarizer verifies these invariants and measured storage equality. The native bridge checks model round trips. Saved `.bin` files are raw codec recipes accepted by `TieredQ::deserialize`, not the checksummed containers accepted by `TieredQ::load`.

Search timing uses batches up to 32, one warmup, and three measured passes with eight threads. The last batch can be smaller. Timings exclude query transformation and initial index construction. Gaussian always precedes learned, and the shared host is not reserved, so these are exploratory paired kernel/serving measurements, not a deployment speed claim. `add_s` includes encoding and representation construction; `fit_s` includes residual assignment, projection, and fitting. `train_s` is the common Gaussian model training time; learned construction adds `fit_s`.

`returned_score_mse` evaluates only each variant's own retrieved top-10 candidates. Since those candidate sets differ, it is descriptive and not a fixed-candidate score-error comparison. Reconstruction and Recall@10 are the primary quality measurements. The study does not evaluate per-coordinate codebooks, joint allocation/fitting, absolute-error fitting, or the 64/80/128-total-bit regime.
