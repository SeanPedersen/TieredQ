# CPU deployment benchmark (issue #4)

Run on Linux x86-64 with AVX2/AVX-512, C++20, LAPACK/BLAS, OpenMP, Git, curl
and uv installed. Python 3.12 dependencies are isolated from the main package.
Both codecs compile with the same compiler and `-O3 -march=native -ffp-contract=off`.
The nanobind module uses `NOMINSIZE`: otherwise nanobind appends `-Os`, silently
penalizing header-only RaBitQ while TieredQ's separate library retains `-O3`.
The runner audits the effective options in `compile_commands.json` for both codecs
and refuses measurements unless the last optimization flag is `-O3`. The module
also rejects size optimization at compile time. Actual commands are saved in metadata.

```sh
cd benchmark
./validate.sh                                 # build + audit + tests; no vector download
./run.sh --seeds 0                            # fixed-seed full core matrix
./run.sh --max-base 1024 --queries 32 --seeds 0 # smoke across all datasets
./run.sh --datasets glove --max-base 1024 --queries 32 --seeds 0 --output output-smoke
uv run python -m unittest test_benchmark
PYTHONPATH=build-native uv run python -m unittest test_native test_reference
uv run python -m unittest test_build_policy
```

The first command downloads datasets, prepares fixed disjoint base/query splits,
caches exact FP32 top-10, builds both native adapters, runs the matrix and generates
`output/results.csv`, `meta.json`, `summary.json`, `preprocessing.csv` and exactly three PNG figures.
Use a new `--output` directory to retain previous runs. A failed matrix stops with
a nonzero exit code and leaves completed CSV rows; incomplete runs are not results.
`--data` allows reuse of a dataset cache. Downloads and ground-truth preparation
are excluded from `matrix_wall_s`; native compilation is also excluded, while index builds and the entire matrix are included.
The <1h acceptance bound requires a measured full run; smoke results cannot establish it.

## Data and metric

- SIFT1M: ANN Benchmarks `sift-128-euclidean.hdf5`, 1M base rows,
  first 1,000 of the separate 10K queries, L2.
- GloVe-200: ANN Benchmarks `glove-200-angular.hdf5`, sorted random 100K
  base subset (`default_rng(42)`), first 1,000 separate queries, normalized cosine.
- OpenAI3-1536: Qdrant's DBpedia embedding dataset on Hugging Face, sorted random
  101K sample (`default_rng(42)`), 1K randomly selected disjoint query positions,
  remaining 100K base rows, normalized cosine. Dataset revision
  `4a9731217921bc476a0f03544f11f22ae4903fa5` is pinned.

GloVe and OpenAI split logic follows the
[reference comparison](https://github.com/VectorDB-NTU/rabitq-turboquant-comparison/tree/main/ann).
Both codecs use identical original FP32 base/query rows, normalization and truth. Smoke truncation
happens after full split selection. Pinned source URLs/revisions and raw HDF5 SHA256 checksums live in `datasets.json`; downloads fail on checksum mismatch. Base/query hashes and selected row indices
are saved alongside each cache. Never interpret the reduced-base smoke as a full result.
Exact truth uses chunked FP32 matmul, bounded score buffers, and global top-k merging;
truth is recomputed for every base/query count and metric version.

**TieredQ L2 adapter:** TieredQ exposes IP scoring only. For SIFT it receives
`[x/s, -||x/s||²/2]` as base vectors and `[q/s, 1]` as queries, where
`s = max ||x||`. This inner-product ranking equals original L2 ranking exactly
before quantization. The extra dimension's storage and timed query transformation
are included. The target code budget remains `bits × 128`, shared across 129 input
dimensions. RaBitQ uses its native L2 scorer on the original 128-dimensional vectors;
forcing the lift on RaBitQ would unnecessarily pad it to 192 dimensions. Both
adapters are evaluated against exactly the same original L2 FP32 truth. CSV `d=128`
and effective bits/dimension refer to the original data. SIFT results include the
cost of TieredQ's metric adapter and do not establish future native TieredQ L2 speed.

## Native methods and fairness

TieredQ uses the checked-out source, fast-only storage, and its native batch search.
Training samples up to 32,768 base rows uniformly without replacement using the
run seed; training and all encoding/packing count toward build time. The centroid
count follows the Python API default (up to 256). The benchmark changes no TieredQ
kernels. `encode_vps` times native add/encode/pack separately from training. RaBitQ
encoding includes native batch packing; rotation setup and global mean computation precede this timer.

RaBitQ pins official RaBitQ-Library commit
`ced3763857e7e4e48553951bf66fabc8517bc4f0`, the reference repo's submodule revision.
The measured adapter constructs **no IVF/ANN container**. It owns native 32-vector
packed code blocks, the native correction/bound factor arrays, extra-bit codes,
the global mean in rotated coordinates, and the FHT/Kac rotation. Row positions
are implicit IDs: there is no stored ID array, posting list or cluster directory.
It calls official `quantize_split_batch`, `split_batch_estdist` and
`split_distance_boosting` kernels, using the default high-accuracy (HACC) lookup.
Encoding batches and query batches use OpenMP; query-level batching gives both codecs the same 1/8
worker counts without nested workers. This official revision targets x86;
this harness does not claim ARM portability.

An exact, checked replacement in a separate include overlay seeds the FHT rotation
with `BENCH_SEED` instead of `random_device`. Native bound-based extra-bit scoring
is preserved: all vectors' 1-bit codes are scanned and the official kernel decides
which extra-bit codes can affect top-k. There is no partition or probe selection. This is the default serving comparison.

`--full-score` additionally replaces the extra-bit candidate-bound condition with
an unconditional evaluation. It forces scoring of every complete code and can be
much slower; results must be kept separate from native serving results. For example:

```sh
./run.sh --full-score --output output-full-score
```

The upstream checkout is left untouched. Overlay SHA256 hashes and upstream commit
are saved in metadata. Missing patch context fails loudly. Both sides include query
preprocessing, top-k and binding overhead in timed calls. A fixed OpenMP count of
8 is used for builds; searches use the requested 1 or 8. BLAS is held at one thread
inside timed workers.

After the core matrix, `profile.py` builds a separate diagnostic extension from
copies of the same sources. It adds timers around actual rotation/projection/LUT
preparation and around the complete native query. It reruns seed 0 on the first
32 queries, three repeats plus three warmups, for each scheduling configuration.
`preprocessing.csv` reports amortized preprocessing milliseconds and its native
query-time fraction for every point, so cases above 10% are explicit. The original
native extension and headline timings have no diagnostic counters. Diagnostic
construction/build time is recorded separately as `profiling_wall_s`.
Single-query fractions are elapsed time; RaBitQ batch fractions aggregate worker
query elapsed times while TieredQ batch fractions use elapsed batch time. These fractions
identify preprocessing cost and are not cross-codec parallel-latency comparisons.
The TieredQ SIFT Python query lift is included in headline end-to-end latency but
excluded from these native-only stage counters.

## Storage and timing

Effective storage is **measured bytes**, not nominal bit width. RaBitQ's self-contained
saved flat representation includes padded code blocks, all native factor arrays,
extra-bit payload, the required global mean and rotation state. No explicit row
IDs or IVF metadata are stored. A regression test checks that adding vectors adds
exactly the native code/factor bytes, excluding a per-vector ID surcharge.
TieredQ uses the larger of saved index size and the serving footprint of the
reloaded index: vector storage and fast-cache allocations (including lookup
projection panels, groups and codebooks), plus `codec.memory_bytes()`. A reloaded model
holds basis, mean, sigma, centroids, codebooks and half norms; encoder buffers
(fused panels, FP64 centroid projections) are built on the first append and are
reported separately in diagnostics (`model_memory_bytes` minus
`model_serving_memory_bytes`). Storage excludes input
FP32 arrays and transient query/top-k scratch. This conservative TieredQ measure
and serialized RaBitQ measure are not process RSS; allocator bookkeeping is excluded.
Per-row native representations are fixed-stride for each built configuration, so
mean and p95 code bits per vector coincide; shared overhead is amortized over N.

Each timing point uses three warmup queries and three complete repeats of all
queries. All timings use seed 0; seeds 1 and 2 record recall with `NaN` timing fields.
Recall-only seeds search once at 8 threads/batch 32 and share that deterministic
recall across scheduling rows; native tests verify batch/serial result equality.
Recall plots show population mean ± std over the three seeds. QPS uses total queries
divided by total measured call time. Batch 1 p50/p95 are single-query end-to-end
latencies. Batch 32 p50/p95 are response latency per query in its batch, **not**
batch duration divided by 32; the tail batch is included. Only batch-1 latency and
batch-32 throughput are plotted. Keep thread counts separate; batch 1 has only one
query to schedule and should not gain query-level parallelism at eight threads.

`summary.json` reports the minimum **measured budget point** whose mean Recall@10
reaches 0.95, or null if unreachable. It does not interpolate across the four budgets,
and does not imply that matching nominal budgets match actual storage. Compare
points by effective bytes; at an RaBitQ anchor use TieredQ points at the same or
lower measured storage. The requested exact-equal-memory and exact-equal-recall
answers may have no matching measured point; report that instead of inventing one.
Optional reranking and TurboQuant are omitted.

## Hartmut validation

Workspace: `/home/sean/experiments/TieredQ-issue4-20261023/TieredQ` via `ssh sean@hartmut`.
CPU: Intel Core i9-7980XE (18 physical cores / 36 logical CPUs, AVX-512).
Compiler: GCC 16.2.1. Per-run OS, RAM, exact compiler version, source hashes,
commit and dirty state are recorded in `meta.json`.

The experiment's result artifacts remain in its `benchmark/output*` directories;
large datasets/builds are ignored by Git. Full-run validation and its acceptance
status must be read from the measured outputs, not inferred from the harness.

## Validation status

Earlier Hartmut runs were invalidated after finding the nanobind `-Os` override
and a disabled HACC setting. Their QPS comparisons must not be used. The full
matrix was stopped; focused validation runs after the correction.
The initial IVF-based adapter was also removed so it cannot add ID arrays or
container overhead to the comparison. An independent `codec_reference` executable
calls official one-list IVF only as a correctness reference, outside all measured
paths and storage accounting. Its IDs are checked against the direct flat adapter for all four budgets, L2/IP,
128/200/1536 dimensions with a partial final code block, serial direct search and eight-thread adapter batching.
The corrected fixed-seed full run is complete: see [Hartmut full results](results/hartmut-full-single-run-20261023/README.md). Its 96 rows cover the entire requested matrix; this one run does not establish statistical confidence.
`./validate.sh` reproduces focused validation from a clean Linux checkout using
the same build factory and compiler audit as the matrix runner.

The reproducible implementation, native adapters, independent reference executable,
build guards, tests, dependency lock and dataset catalogue all live in this directory.
Hartmut-specific paths appear only in the validation notes, not executable setup code.

## Fixed-seed full run (2026-10-23)

[Measured results and audit](results/hartmut-full-single-run-20261023/README.md): 24 codec configurations, 96 scheduling rows, seed 0 only. Core matrix time: 881.97 seconds; separate preprocessing diagnostics: 171.07 seconds. All 13 pre-run validation tests passed. Figures connect every measured 1/2/4/8-bit budget in order, label each budget, and separate thread counts into panels. The fixed-seed figures have no seed-variance error bars. Archived `summary.json` retains the generic `recall_mean`/`recall_std` field names; with one seed the latter is zero by construction and is not an uncertainty estimate.

## High-bit recall correction (#12)

[Investigation and verified results](results/issue12-recall/README.md): fast models
containing tiers wider than 4 bits preserve original code IDs and FP32 query weights. The
SIFT TieredQ adapter now fits a metric-preserving query-sensitive transform from
base vectors. Its fit, query transformation and stored maps are included in costs.
`--legacy-l2-lift` selects the original adapter; reproducing the original fast
approximation also requires its archived source snapshot. Native RaBitQ bounds
and HACC remain the primary comparison.

Use `--batches 32 --threads 8` for the primary serving point, or omit those flags
for the full scheduling grid. Single-query latencies are absent from a batch-only
run. `profile.py --stages` uses a separate instrumented build for projection,
centroid, LUT, fused scan/accumulation and selection worker-time counters.

To refresh the eleven reported TieredQ serving-memory points without running
searches or collecting new timings, build the benchmark extension and use:

```sh
uv run --locked --python 3.12 python -c 'from native_build import build_native; build_native()'
uv run --locked --python 3.12 python measure_storage.py --output output-storage/results.json
```

Run from `benchmark/` with the original full-dataset caches available. The report
records all three seeds, source and binary hashes, cached dataset manifests,
compiler commands, per-seed diagnostics, and maximum bytes/vector across seeds.
Historical performance measurements remain separate.

[Corrected full-dataset serving-memory results](results/issue12-recall/storage-refresh/README.md)
cover the eleven reported configurations at seeds 0/1/2. Historical figures and
equal-memory summaries retain the earlier accounting and are labeled accordingly.

[Compact narrow fast-query projection](results/serving-memory-20261026/README.md)
reports unchanged-budget RAM savings, separate disk/RAM allocation accounting for
both codecs, and a seed-0 OpenAI comparison constrained by both storage budgets.
The saved format and unchanged-budget disk sizes are preserved.

## Gemini/DBpedia native TurboQuant experiment

The focused [TurboQuant runner](turboquant/README.md) adds the pinned 100K, 768D
Gemini dataset with 1,000 held-out document queries and Qdrant’s native calibrated
1/2/4-bit implementation. The [single-seed measured bundle](results/gemini-qdrant-turboquant-single-seed-20261026/README.md)
contains 24 scheduling rows, storage-only integer TieredQ calibration, provenance,
raw predictions/timings, validation, audit and figures. The earlier datasets and
RaBitQ runner remain separate.

`run_rabitq_gemini.py` then measures the repository's normal flat RaBitQ adapter on
the same prepared Gemini arrays at the nominal 1/2/4-bit anchors. The
[RaBitQ bundle](results/gemini-rabitq-single-seed-20261026/README.md) contains its
12 scheduling rows and raw artifacts. `compare_gemini_three_codecs.py` merges both
bundles into the audited [three-codec comparison](results/gemini-three-codec-single-seed-20261026/README.md),
including a storage caveat: RaBitQ uses 3.0–5.9% more bytes/vector than the Qdrant
anchors, so its nominal-bit points are not strictly equal-storage comparisons.
