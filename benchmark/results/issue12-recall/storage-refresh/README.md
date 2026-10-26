# Issue #12: corrected serving-memory measurements

Measured 2026-10-26 on hartmut from the final-review source snapshot. These are
storage-only measurements: no recall, throughput, latency or build-time results
were collected. All eleven reported TieredQ configurations were rebuilt on the
original full base sets at seeds 0/1/2 because no saved indexes were available.
The table reports the maximum bytes/vector across those seeds.

Each index is saved and reloaded. Effective index bytes are the larger of saved
file size and the reloaded fast-cache allocations plus serving codec memory.
SIFT uses `L2Transform.fit(base)` and additionally counts its scale and both
maps. The corrected counters include cache objects, geometry arrays, lookup
projection panels/groups/codebooks, and the lazy encoder object. Encoder buffers
are absent from the reloaded serving model. Allocator bookkeeping, shared-pointer
control blocks and transient query/top-k scratch are excluded; these are not RSS.

## Current serving footprint

The earlier column comes from `pack-opt/primary/results.csv` at budgets 4/8 and
`code5/new/*/results.csv` for the five mid-width points. Those historical files,
figures and timing results are preserved unchanged.

| Dataset / budget | Earlier bytes/vector | Corrected bytes/vector | Increase |
| --- | ---: | ---: | ---: |
| sift 2-bit | 41.319482 | **41.320308** | 0.000826 |
| sift 4-bit | 73.341674 | **73.342688** | 0.001014 |
| sift 8-bit | 137.346406 | **137.347415** | 0.001009 |
| glove 4-bit | 112.877320 | **118.705640** | 5.828320 |
| glove 5-bit | 140.774200 | **140.787760** | 0.013560 |
| glove 6-bit | 164.775480 | **164.789040** | 0.013560 |
| glove 8-bit | 212.783160 | **212.796720** | 0.013560 |
| openai 2-bit | 493.832720 | **493.905340** | 0.072620 |
| openai 3-bit | 695.473040 | **695.553300** | 0.080260 |
| openai 4-bit | 887.724240 | **887.804740** | 0.080500 |
| openai 8-bit | 1655.844280 | **1655.924640** | 0.080360 |

The largest change is GloVe 4-bit, whose narrow lookup cache owns a dense query
projection omitted by the earlier counter. Its corrected 118.706 bytes/vector
exceeds the historical RaBitQ 3-bit point at 116.012 bytes/vector: the earlier
claim that it fits inside that point's memory budget no longer holds. It still
fits inside the RaBitQ 4-bit point at 148.012 bytes/vector. Treat the old
whole-curve equal-memory summaries as historical, not current comparisons.

The approved threshold-5 policy is retained. In particular, the measured OpenAI
2/3-bit points now occupy 493.905/695.553 bytes/vector; both fit below the
historical RaBitQ 4-bit point at 788.070 bytes/vector. The old threshold-7 curve's
statement that only TieredQ 1-bit fits there is superseded. These comparisons
concern storage; they make no new performance claim.

## Evidence and validation

- `results.json`: all 33 per-seed measurements, summaries, tiers, adapter bytes,
  serialized index sizes, resident model/cache diagnostics, dataset manifests,
  source/harness/native-binary hashes, and compiler commands.
- `run.log`: fresh isolated native build and every storage measurement.
- `validation-x86.log`: fresh x86 Release C++ suite and all 16 native benchmark
  validation tests pass, including official RaBitQ reference parity. This is
  fresh Release validation; x86 ASan remains the historical pack-opt check.
- Fresh local ARM C++ suite and rebuilt Python API suite pass (44 tests).
  Both pruning tests and the 12 local benchmark/reporting unit tests also pass.
- `../audit.py` checks the 33-point matrix, original dataset manifests, accounting
  equations, maximum-over-seeds summaries and recovered scan-stage snapshot.
- `../scan-opt/as-run-sources.tar.gz` was recovered from the original workspace;
  its recorded SHA-256 and all 43 source hashes match the stage metadata.

No result archives were rewritten. Incidental Python bytecode in the historical
packed-stage snapshot remains part of its original hashed bytes.

## Reproduction

From `benchmark/` with the existing original dataset caches, choose a fresh output:

```sh
uv run --locked --python 3.12 python -c 'from native_build import build_native; build_native()'
taskset -c 0-7 uv run --locked --python 3.12 python measure_storage.py \
  --data data --output output-storage/results.json
```

The measurement used an isolated workspace at
`sean@hartmut:/home/sean/experiments/TieredQ-issue12-final-review-LibSHX6y`, the
original benchmark Python 3.12 environment, eight OpenMP threads, one BLAS thread,
and cached datasets from the earlier issue-12 workspace. No remote baseline
workspaces or historical outputs were modified.
