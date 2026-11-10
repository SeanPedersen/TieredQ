# Geometry-aware hierarchical ANN prototype

Standalone C++20/CMake research implementation of the first three variants. The production TieredQ library is unchanged. No required third-party C++ dependencies, per-vector graphs, quantization, or full covariance matrices. Optional dataset preparation and Faiss references use Python 3.12 via `uv` with script-local dependencies.

## Build and verify

```sh
cmake -S ann-prototypes/geometry_ann -B /tmp/tieredq-geometry-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/tieredq-geometry-build -j 4
ctest --test-dir /tmp/tieredq-geometry-build --output-on-failure
/tmp/tieredq-geometry-build/geometry_bench > /tmp/geometry-smoke.csv
```

The deterministic synthetic smoke dataset defaults to 10,000 clustered/elongated 32-dimensional vectors, 100 independently generated queries, and double-accumulated exhaustive ground truth. It validates the harness; it cannot establish performance on real embeddings. Tests cover complete-search agreement with exhaustive search, off-vector queries, ties, duplicates, singleton inputs, conservative bounds, projection metric normalization, and strict search caps. Tests do not depend on `assert`, so Release builds execute all checks.

## Variants and contracts

| Version | Partitioning | Search metadata |
| --- | --- | --- |
| V0 | Sampled coarse k-means, then 32 fine k-means children by default | Centroid ordering, no geometric pruning |
| V1 | Same coarse training; global best-score adaptive bisection | Centroid, maximum radius, total variance |
| V2 | Identical adaptive partitioning to V1 | V1 plus up to four approximate principal directions and full-population projection extrema |

Training uses four Lloyd iterations and at most 4,096 sampled vectors per clustering call by default; all members receive a final nearest-centroid assignment. Empty clusters disappear. Coarse and fine samples use a reproducible seeded generator. Packed leaf rows retain original base IDs. Training, metadata, and returned L2 distances use FP32 coordinates with double accumulation and independent accumulation lanes; no fast-math build flag is used. Leaf scans first use a SIMD FP32 distance filter (runtime AVX-512/AVX2 on GCC/Clang x86, NEON on AArch64, scalar elsewhere). A conservative error allowance rejects only candidates strictly above the current double threshold; competitive candidates are scored again in double, preserving ID ties and geometric pruning. Overflow or excessive error disables filtering. Build with `-DGEOMETRY_FAST_SCAN=OFF` to reproduce the original scan. Do not enable fast-math. See `results/simd-scan-20261107/` for a synthetic control; see also `results/hartmut-simd-20261107/` for the SIFT1M comparison on AVX-512.

Adaptive splitting prioritizes total scatter times `(1 + largest_axis_scatter / total_scatter)`. Total scatter combines population and dispersion. Axis anisotropy is a deliberately cheap proxy for dominant-eigenvalue anisotropy; it does not detect every rotated elongation. The default split is a median cut on the highest-variance coordinate, balanced in population. `--split kmeans` instead tries sampled two-means at each beneficial split, with median fallback if it degenerates. Splitting stops at the global leaf budget, when every leaf meets target size, or when remaining scatter is zero. Coarse groups remain the leaf owners; no point crosses its coarse parent during refinement. No speculative graph stage is implemented.

V2 estimates directions with eight covariance-vector power iterations on at most 256 sampled members and orthogonalizes against earlier directions. It never constructs a dimension-by-dimension covariance matrix. Projection extrema are computed over **every** member against the final stored directions. Directions stay FP32 and intervals/norms stay double in this first experiment. FP16 is intentionally deferred until its rounding and inflation costs can be isolated.

Radius distance is `max(0, ||q-c|| - radius)^2`. A slab contributes `distance(q·u, [lo, hi])^2 / ||u||²`. The maximum of these bounds remains valid even with imperfect orthogonality; summing projection bounds would require stronger guarantees. Radius and projection intervals include dimension-dependent floating-point padding, including absolute dot-product magnitudes to handle cancellation. These are conservative numerical envelopes, not exact convex hulls. Changing the stored direction format requires recomputing extrema or a proven additional inflation.

The priority queue contains both parents and leaves. V1/V2 prioritize parents by their cheap radius lower bound, independently of the leaf-ordering heuristic. Equal priorities expand parents first, then use centroid distance as a tie-breaker. This prevents a distant coarse centroid from hiding a nearby fine centroid, especially when many bounds are zero. V0 retains centroid ordering at both levels as the baseline. A parent expansion enqueues its children. Pruning always uses geometric bounds; centroid and variance ordering are separate heuristics. The optional variance heuristic is centroid distance divided by `(1 + total variance)`, **not** the proposed future M3 directional-variance model. V0 always uses centroid order. Queue size can reach the number of partitions; the prototype does not use a wide point-level beam.

The primary success gate is high recall while scanning a small fraction of the corpus. The Instagram follow-up tests hard 1% and 5% vector caps; reaching recall by scanning a quarter of the base fails this gate.

`vector_scans`, `leaf_scans`, and partition `metadata_evaluations` are hard caps. A leaf that cannot fit the remaining vector budget is skipped whole; later smaller leaves may still fit. These budget skips are reported separately from valid geometric pruning. One metadata evaluation means one partition, including its available projections, rather than one arithmetic operation. Metadata exhaustion can omit unexplored children; small caps can starve later parents because initial parent evaluation is deterministic. Queue entries already evaluated remain eligible to scan. Budgeted results are approximate. Complete traversal with sufficient metadata evaluates every unpruned candidate and agrees with exhaustive L2 search, including ID tie-breaking.

## Real data and matched recall

Use little-endian SIFT-style fvecs and ivecs files. Each record has an int32 width followed by FP32 coordinates or int32 IDs. Truth must reference **the exact supplied base file** and contain at least ten neighbors per query. Base vectors are never silently truncated.

For an existing `.npy` matrix, create a deterministic held-out split and exhaustive Faiss L2 truth:

```sh
uv run --python 3.12 ann-prototypes/geometry_ann/prepare_dataset.py \
  /Users/Studio-MacMini/projects/vector-db-benchmark/instagram_vectors.npy \
  /tmp/geometry-instagram-20261106 --nq 200 --seed 42
```

This preserves original row order for the base after removing query rows, converts all coordinates to FP32 without normalization, and hashes the exported benchmark files. Query row IDs and source shape/dtype/stat are recorded in `manifest.json`. Output files are created exclusively; existing artifacts are never overwritten. Base arrays and exported files are intentionally outside the repository. Preparation peak RAM and time are outside index build metrics. Source-vector duplicates are retained and may create zero-distance neighbors across the split.

Run each variant separately to isolate process RSS high-water measurements:

```sh
for variant in V0 V1 V2; do
  /tmp/tieredq-geometry-build/geometry_bench \
    --base /tmp/geometry-instagram-20261106/base.fvecs \
    --queries /tmp/geometry-instagram-20261106/queries.fvecs \
    --truth /tmp/geometry-instagram-20261106/truth.ivecs \
    --nq 200 --coarse 32 --children 32 --leaves 1024 --target-size 256 \
    --variant "$variant" > "/tmp/instagram-$variant.csv"
done
```

CSV includes recall@10, p50/p99, average scanned vectors, partition evaluations, pruned nodes, metadata exhaustion rate, build seconds, raw-vector bytes, and owned persistent overhead. Search budgets sweep powers of two through all leaves, and each ordering receives a warmup. Stderr selects the lowest observed p50 among settings attaining recall ≥0.95 and ≥0.99, or explicitly reports an unreached target. These are coarse sweep selections, not interpolated or exactly equal recall. Use the recall, scanned-work, and latency columns together. For more stable tail measurements, increase the query count and repeat runs/seeds on otherwise idle hardware. Two hundred queries provide only two observations in the upper 1% tail.

For a strict fraction-of-corpus experiment on the exported Instagram split:

```sh
/tmp/tieredq-geometry-build/geometry_bench \
  --base /tmp/geometry-instagram-20261106/base.fvecs \
  --queries /tmp/geometry-instagram-20261106/queries.fvecs \
  --truth /tmp/geometry-instagram-20261106/truth.ivecs \
  --nq 200 --coarse 32 --children 32 --leaves 1024 --target-size 256 \
  --variant V1 --split kmeans --routing both --leaf-scans 1024 \
  --vectors 4552,22763 > /tmp/instagram-low-scan.csv
```

`--vectors` accepts one or more comma-separated hard per-query vector caps. These are floors of 1% and 5% of 455,266 base vectors. `--ordering centroid|bound|variance|all` restricts the ordering ablation. `--leaf-scans` selects a single leaf budget, avoiding redundant sweeps once the vector cap dominates. CSV includes mean scan fraction and skipped-leaf counts. Check **recall at the scan cap**, rather than relaxing the cap to attain recall. Target selection explicitly reports `NOT_REACHED` when necessary.

`--routing hierarchy` is the default. `--routing flat` is a diagnostic that evaluates all leaf centroids directly, bypassing coarse routing while preserving the exact same partitions. `--routing both` compares both policies with one built index. This diagnostic spends more metadata work and is not evidence that hierarchical routing is good; it measures how much recall coarse routing loses. Its evaluations still obey the metadata cap. Default output preserves the hierarchical architecture.

Median bisection is cheap and population-balanced but sensitive to coordinate orientation. Local two-means costs more training/assignment work and can create imbalanced leaves, but can follow oblique structure. Compare both under the same hard vector cap.

`--metadata N` sets a fixed metadata cap across the leaf-budget sweep. By default the cap can cover all parents and leaves. `--sample`, `--iterations`, `--pca-sample`, `--directions`, `--seed`, `--split`, and partition options expose the main research controls. Use equal `coarse × children` and `leaves` budgets for matched partition caps; target size can still stop adaptation early. V1 versus V2 fixes the partitioning, separating geometry metadata from leaf allocation.

## Faiss references

```sh
for variant in ivf ivf-hnsw hnsw; do
  uv run --python 3.12 ann-prototypes/geometry_ann/compare_faiss.py \
    --base /tmp/geometry-instagram-20261106/base.fvecs \
    --queries /tmp/geometry-instagram-20261106/queries.fvecs \
    --truth /tmp/geometry-instagram-20261106/truth.ivecs \
    --nq 200 --nlist 1024 --sample 65536 --variant "$variant" \
    > "/tmp/instagram-faiss-$variant.csv"
done
```

References run single-threaded with squared L2; insertion uses batches of 8,192 rows with progress reporting. IVFFlat uses a flat quantizer or an HNSW quantizer (32 neighbors, construction ef=100, search ef=128); both use four coarse k-means iterations. HNSW uses 32 neighbors, efConstruction=100, and an efSearch sweep. IVF sweeps nprobe through nlist. The IVF/HNSW quantizer's approximate routing can prevent exhaustive coverage even at large nprobe. Sampling sizes differ because a single flat 1,024-centroid quantizer and a 32-by-32 hierarchy have different training problems; do not claim perfectly matched clustering work. Record parameters along with conclusions.

Faiss query latency includes Python API overhead, whereas the C++ prototype is timed directly inside C++. At microsecond-scale latency this biases comparisons. Distance-evaluation counters differ from scanned-vector counts: IVF counters exclude coarse quantization, and HNSW counts graph distance computations. Both scripts select measured settings at the same recall thresholds.

## Measurement limits and decision gates

- Prototype persistent overhead counts owned vector **capacities**, IDs, centroids, directions, interval arrays, parent links, and object storage. It excludes allocator bookkeeping, input vectors, and transient search queues.
- Prototype process peak RSS includes loaded input, build temporaries, packed index, runtime, and any earlier process high-water usage. It is not incremental index RAM or an allocation-traced build peak. Query measurements occur after its snapshot.
- Faiss overhead is serialized bytes minus raw-vector bytes, a **proxy**, not resident allocator RAM. Its RSS is sampled before serialization, which itself can allocate another index-sized buffer. Neither overhead metric should be presented as directly identical to the other.
- Build time includes clustering, geometry, assignment, and packing/add; it excludes dataset loading, truth computation, and index serialization.
- Default C++ is portable CPU code; Faiss can use architecture-specific SIMD and BLAS. Compiler, CPU, thread count, background load, parameters, and repetitions matter.
- Bounds may be weak in high dimensions. Retain adaptive allocation, radius, and PCA only if their measured work/latency improves at comparable recall. No performance win is presumed.
- Only consider partition shortcuts after examining V2 routing misses under practical budgets. No graph construction or compression is hidden in this baseline.

Source layout follows the research concerns: `training.cpp` partitions, `geometry.cpp` constructs and evaluates bounds, `index.cpp` assembles/searches packed storage, `dataset.cpp` handles benchmark I/O, `benchmark_options.cpp` validates configuration, and `benchmark.cpp` measures/selects settings. `index.h` is the small public API; `internal.h` exposes only internal training/geometry contracts.

Graph ablations now build only the PCA/shared projection metadata required by `--bounds` or the rows of `--settings`. Partitioning and graph topology stay the same; centroid-only/radius runs therefore avoid unused projection build time and persistent storage. Default all-mode ablations still build all metadata.

## Controlled routing and partition study

`graph_bench --variant V0 --bounds none --children 128 --coarse 64` now builds
fixed hierarchical k-means partitions with a graph over the same leaves. V0
requires centroid-only settings because it does not store geometric bounds.
V2 retains adaptive splitting. `--iterations` applies to coarse and fine/local
training, so increasing it changes both levels. Both flat and graph routing use
an exact SIMD double centroid kernel on AVX2 hosts; its four accumulation lanes
match the reference metric. Flat routing heapifies evaluated descriptors in bulk and pops only the candidates needed by the scan budget.

The held-out study and its calibration-first runner are in
`results/hartmut-routing-20261107/`. It uses the same partitions, packed vectors,
and scan kernel for routing comparisons. Flat and graph phase timers separate
navigation from scanning; hierarchy interleaves those operations, so its phase
columns are only initial-parent setup versus remaining search, not a pure split.

## Cell-ceiling study (hyperplane / convex bounds)

`cell_ceiling` builds fixed V0 leaves and, offline, measures how many leaves each lower
bound certifies against the 10th true-neighbour distance (`cell_bounds.{h,cpp}`,
`cell_diagnostics.cpp`). Slabs use member extremes along directions toward the K nearest
centroids. Bisectors of stored centroids are not valid boundaries, because the stored centroids are member means. The
joint bound is the exact distance to the intersection of K slabs (K×K Gram, safe at
every iterate). `hull` is the convex-hull ceiling for any convex metadata. `nearest-member`
is the oracle. `cell_tests` covers the closed-form polytope corner, hull convergence, and
bound monotonicity on FP32/SQ8 fixtures.

Result (`results/hartmut-cells-20261110/`): on SIFT1M the slab bounds keep about 4,300 of 8,192 leaves, while the
hull keeps 163. On Qwen 8.84M×1024 the slabs keep about 8,130, the hull 888, and the oracle 4.3. Cheap convex
metadata cannot approach the hull in high dimension, and the hull costs more than a scan.
Geometric pruning is therefore dropped. The remaining gap is ranking: 130K centroid-ordered
vectors versus 8.6K oracle vectors for 10/10 on Qwen.
