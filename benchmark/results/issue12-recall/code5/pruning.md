# Tier-prefix pruning feasibility

This Python-only study asks whether scoring the widest tiers first can remove enough work for a 2–4× serving gain. It evaluates complete SIFT1M, GloVe-200 and OpenAI3-1536 base sets, 200 held-out queries, seed 0, top 10, at nominal 4/8-bit budgets. SIFT uses the benchmark's fitted `L2Transform`, including its query-metric calibration.

## Bounds and score preservation

For a query, write the score as `base[doc] + factor[doc] * dot`, where `base` includes the centroid score and correction term. After scoring a prefix, bound each remaining tier using its codebook minimum/maximum and each signed query weight. This is equivalent to a codebook midpoint contribution plus `sum(abs(weight)) * half_range`. A negative document factor selects the lower dot bound rather than the upper one. The diagnostic adds a conservative FP32 roundoff allowance before pruning.

The optional Cauchy–Schwarz (CS) bound uses the remaining query-weight norm and an exact per-document norm of remaining decoded levels. It requires extra stored metadata for each chosen split; it is not a free improvement to the range bound. The study uses float64 residual norms. A native FP32 norm representation would need outward rounding and a new storage measurement.

Query projection, weights, dot products and metadata epilogue mirror native sequential FP32 operations, with separate multiply and add. Survivors continue the prefix accumulator in original coordinate order. For every prefix/query, the study checks that all reference top-10 candidates survive both bounds and that every CS survivor score has the same bits as the full sequential score. It does not time a native pruned scanner. A local synthetic check also matched native score bits for 90 returned hits; the production C++ regression suite independently checks code-lookup scores against the scalar codec.

Thresholds are either the final kth score (an unattainable oracle) or a streaming local heap: eight contiguous document segments, thresholds updated after chunks of 256 documents. Earlier chunks' exact top-k supplies the threshold; skipped documents below the current threshold cannot change that heap. Equality survives, preserving cutoff ties. A 200-query SIFT check found zero native hits below the reference cutoff and 1.0 native top-10 overlap under score/id ordering (`check-native.py`, `check-native.log`). The raw report's 0.999 SIFT 4-bit overlap uses arbitrary `argpartition` cutoff ties.

## Work model and limitations

If the prefix contains fraction `p` of coordinates and fraction `s` of 16-document blocks has any survivor, the optimistic work gain is `1 / (p + s * (1-p))`. The JSON also reports a packed-bit-weighted variant. Neither is measured QPS. Bounds, heaps, scratch storage, survivor compaction, metadata loads and partial decode overhead are excluded. The model is per query; a batched implementation sharing decoded blocks across queries would need the union of their surviving blocks, weakening these estimates further.

Whole tiers are the split unit. A one-tier model has no nonempty prefix with an unscored suffix; the study does not rule out splitting inside a tier or another representation.

## Reproduction

After building the existing benchmark extension, run from `benchmark/`:

```sh
taskset -c 0-7 uv run --locked --python 3.12 python prune_study.py \
  --native build-native --queries 200 --threads 8 --bits 4 8 \
  --output output-prune-study.json
uv run --locked --python 3.12 pytest -q test_prune_study.py
```

Use a fresh output path. `prune-study-initial.*` preserves the superseded float64/matrix-product estimate; use `prune-study-exact-final.*` for the FP32 continuation results.

## Results by tier prefix

Percentages average over 200 queries. “Blocks” is the percentage of 16-document blocks still requiring the suffix. Gains use coordinate work and streaming thresholds; CS requires the additional norms described above.

| Dataset / budget | Scored tier widths | Coordinates scored | Pruned, oracle range | Pruned, streaming range | Pruned, streaming CS | Blocks, range / CS | Ideal gain, range / CS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sift 4-bit | 8 | 1.6% | 15.309% | 13.151% | 48.025% | 99.92% / 98.70% | 1.00× / 1.01× |
| sift 4-bit | 8/7 | 11.1% | 52.756% | 48.186% | 77.106% | 98.68% / 87.79% | 1.01× / 1.12× |
| sift 4-bit | 8/7/6 | 23.8% | 86.278% | 80.146% | 95.347% | 84.52% / 39.88% | 1.13× / 1.85× |
| sift 4-bit | 8/7/6/5 | 39.7% | 99.186% | 97.137% | 99.127% | 28.55% / 8.99% | 1.76× / 2.22× |
| sift 4-bit | 8/7/6/5/4 | 57.1% | 99.968% | 99.446% | 99.604% | 5.03% / 2.90% | 1.69× / 1.71× |
| sift 4-bit | 8/7/6/5/4/3 | 79.4% | 99.998% | 99.707% | 99.713% | 1.48% / 1.40% | 1.26× / 1.26× |
| sift 4-bit | 8/7/6/5/4/3/2 | 93.7% | 99.999% | 99.733% | 99.733% | 1.10% / 1.10% | 1.07× / 1.07× |
| sift 8-bit | 8 | 94.6% | 99.999% | 99.728% | 99.733% | 1.18% / 1.11% | 1.06× / 1.06× |
| sift 8-bit | 8/7 | 99.2% | 99.999% | 99.736% | 99.736% | 1.07% / 1.06% | 1.01× / 1.01× |
| glove 4-bit | Single 4-bit tier | 100% | — | — | — | — | No tier split |
| glove 8-bit | Single 8-bit tier | 100% | — | — | — | — | No tier split |
| openai 4-bit | 7 | 0.8% | 0.000% | 0.000% | 0.000% | 100.00% / 100.00% | 1.00× / 1.00× |
| openai 4-bit | 7/6 | 9.3% | 0.000% | 0.000% | 0.148% | 100.00% / 100.00% | 1.00× / 1.00× |
| openai 4-bit | 7/6/5 | 29.9% | 69.674% | 18.107% | 78.524% | 99.66% / 80.98% | 1.00× / 1.15× |
| openai 4-bit | 7/6/5/4 | 64.8% | 99.944% | 96.168% | 97.014% | 21.73% / 13.91% | 1.38× / 1.43× |
| openai 4-bit | 7/6/5/4/3 | 95.9% | 99.989% | 97.562% | 97.567% | 7.66% / 7.60% | 1.04× / 1.04× |
| openai 4-bit | 7/6/5/4/3/2 | 99.9% | 99.990% | 97.587% | 97.587% | 7.35% / 7.35% | 1.00× / 1.00× |
| openai 8-bit | Single 8-bit tier | 100% | — | — | — | — | No tier split |

All 209,585,227 survivor score comparisons passed bit for bit; all checked top-10 candidates survived. Native top-10 overlap is 1.0 for five points and 0.999 for raw SIFT 4-bit candidate selection; the separate tie check resolves SIFT to 1.0 under score/id ordering.

## Decision

A general 2–4× win is not supported by whole-tier pruning. The strongest range-bound split is SIFT 4-bit at 1.76× optimistic coordinate work; a stored residual norm raises it to 2.22×. A narrow SIFT experiment may be worthwhile later, but that estimate leaves little headroom for overhead. OpenAI 4-bit reaches only 1.38× (1.43× with norms); SIFT 8-bit reaches 1.06×. GloVe 4/8-bit and OpenAI 8-bit have no whole-tier split. These results do not establish competitiveness with RaBitQ’s bound-pruned native serving; a within-tier or alternative bound scheme would require another study.
