# Native integration and interpretation

Qdrant is pinned to `222b2e44ab1c3f17e8246396ead6558e45041df7`. The actual production segment wrapper selects `TQMode::Plus` for 1/2/4-bit storage (`lib/segment/src/vector_storage/quantized/quantized_vectors.rs`); this experiment follows that choice. It does not use the Python showcase, generic TurboQuant, Normal mode, HNSW, filtering or original-vector reranking. The integration is Python 3.12/uv, C++20/nanobind and a Rust release shared library with a small ctypes C ABI. Both adapters expose complete end-to-end native search calls.

## Production API constraints resolved

Production TQ+ calibration's streaming sampler calls `rand::rng()` without a public seed parameter. A checked overlay changes only that call to `StdRng::seed_from_u64(0)`, retaining the production uniform sample sizes, sorted sample order, P-square estimators and fitting formulas. The quantizer's immutable permutation seeds remain upstream's fixed constants; they are not replaced with experiment seed 0. `provenance/qdrant-overlay.patch` records every change. Other uncalled random helpers are untouched.

`TurboQuantizer::heap_size_bytes` is crate-private and omits the materialized rotation maps. The overlay exposes it and adds the six maps' actual vector capacities. The production encoder also receives a timing observation around its encoding loop. There are no encoding, scoring, SIMD, codebook or floating-point algorithm changes.

The adapter calls production `EncodedVectorsTQ::encode` with all corpus rows eligible for calibration, `input_already_rotated=false`, padded rotation, cosine distance and eight calibration threads. Its temporary in-memory storage uses upstream `TestEncodedStorageBuilder` only as a contiguous byte sink. That builder and the entire encoder wrapper are dropped before serving. Serving reconstructs the public `TurboQuantizer` from the production persisted metadata and retains one contiguous compressed code buffer. It calls `precompute_query` and `score_precomputed_batch`, scores every row and selects the largest ten scores, breaking equal scores by row ID. No production Rust scoring implementation was copied or substituted.

This choice avoids retaining the production wrapper's unused writer scratch and duplicate metadata arrays at serving time. It keeps all required serving arrays, including the quantizer's derived correction cache even though that cache is primarily useful for symmetric scoring. It is a benchmark adapter, not a measurement of Qdrant server/process overhead.

## Actual 1/2/4-bit paths

At 768 input dimensions these widths require no extra dimensional padding. Production preprocessing converts normalized FP32 inputs to FP64, applies the three fixed-permutation Hadamard rounds and rescales to the standard-normal Lloyd-Max grid. TQ+ estimates per-coordinate lower/upper empirical quantiles at the Gaussian CDF of each codebook's outer centroid. It uses 2,048 / 4,096 / 8,192 corpus samples at 1 / 2 / 4 bits. A minimum interval-width guard of `1e-3` remains unchanged. The resulting shift/scale calibration is applied before native quantization and packing.

| Anchor | Codebook | Code payload | Required per-row extras | Native x86 query precision |
|---|---|---:|---:|---|
| A1 | 2 Lloyd-Max centroids | 96 bytes | 8 bytes | `Query1bitWideSimd`, two i8 query planes, radix 256, max absolute integer 32639 (roughly 16-bit) |
| A2 | 4 Lloyd-Max centroids | 192 bytes | 8 bytes | `Query2bitSimd`, two 7-bit query planes, radix 128, max absolute integer 8127 (roughly 14-bit) |
| A4 | 16 Lloyd-Max centroids | 384 bytes | 8 bytes | `Query4bitSimd`, same radix-128 two-plane query representation |

The extras are the FP32 centroid norm and TQ+ symmetric-correction scalar. Query preprocessing rotates in FP64, applies inverse calibration scale and computes a scalar correction, narrows to FP32, then quantizes to the native integer query representation above. It is asymmetric compressed scoring, not FP32 query-by-reconstructed-vector dot products.

This i9 supports AVX-512BW but lacks AVX-512 VNNI. The upstream asymmetric dispatch therefore selects AVX2, unchanged. The adapter does not force another SIMD implementation. Qdrant schedules independent queries through a persistent eight-worker Rayon pool; batch 1 runs on one worker even when eight threads are requested. TieredQ uses its native OpenMP scheduling, which can parallelize a batch-1 scan. Both remain pinned to CPUs 0–7.

## Serving storage and timing

The measured serving total is `max(disk_bytes, resident_bytes)` for each codec. It includes the native adapter shell. `storage-components.csv`, `anchors.json`, `selections.json` and `calibration.json` retain the allocations rather than reusing historical totals.

Qdrant resident storage is the compressed buffer's actual capacity plus 18,432 bytes of forward/inverse rotation maps, 7,680 bytes of shift/scale/derived integer calibration arrays, and the 296-byte quantizer/adapter shell. Its metadata JSON is required on disk; rotation maps are reconstructed from fixed seeds rather than serialized. Runtime metadata parsing objects and the calibration builder are dropped. The resident total determines all three anchors.

TieredQ resident storage is the reloaded model's `memory_bytes()`, actual `storage_bytes()` for vector/fast-cache allocations, and the remaining index/adapter shell. The vector/cache aggregate includes padded code planes, factors/corrections/centroid IDs, lookup projection panels, groups, code offsets and cached codebooks as applicable. The model aggregate includes basis, mean/sigma, centroids, codebooks, half norms and serving scan geometry. Reloading excludes lazy encode-only panels. No separate original-vector payload is retained in fast mode. Disk is the actual saved index size, with hashes retained.

These are owned persistent allocation measurements, not RSS. Allocator bookkeeping, Python/ctypes objects and shared language/thread runtimes are excluded, consistently with the existing benchmark. Thread-local rotation scratch, query projections/LUTs, complete transient Qdrant scores/temporary top-k IDs and TieredQ top-k scratch are query-time scratch, excluded from serving storage but included in call timings. Input FP32 arrays and exact truth are evaluation inputs owned by the driver; neither adapter retains FP32 corpus backing storage.

Qdrant top-k is the adapter's transient full-score partition/sort. TieredQ retains its native scan/selection implementation; no kernels or serving schedules were tuned for this experiment. Thus these results measure the stated exhaustive adapters, not an end-to-end database or a guarantee for alternative top-k integrations.

Qdrant `build_s` includes production calibration, encoding, code compaction/copy and persistence; TieredQ `build_s` includes training and native add/pack but excludes subsequent persistence/reload. This explicit construction-boundary difference prevents treating the build-time ratio as a pure algorithm comparison. Encoding throughput uses only the native encode/pack loops, excludes calibration/training and persistence, and includes Qdrant's byte-sink appends. All selected construction times are separate from the 97.62-second storage-only calibration search.

## Compiler and validation audit

C++ uses GCC 16.2.1 with `-O3 -march=native -ffp-contract=off`, C++20, Release, and nanobind `NOMINSIZE`. Every effective compile command is archived and the final optimization flag is audited as `-O3`.

Rust uses the separately installed `1.99.1` toolchain (without changing the default toolchain), a release profile with `opt-level=3`, fat LTO and one codegen unit, and `RUSTFLAGS='-C target-cpu=native -C opt-level=3'`. Dependency lockfiles and the measured Cargo fingerprint retain the build identity. Rust's default strict floating-point operations are used: no fast-math or reassociation flags and no explicit FMA intrinsic or `mul_add` in the TurboQuant path. SIMD dots are integer operations; FP64 preprocessing and FP32 score postprocessing stay upstream. Qdrant's separate legacy C kernel build requests Haswell internally; TurboQuant's measured scoring path is Rust SIMD and does not call those legacy C kernels. The C build also receives `-O3 -march=native -ffp-contract=off` through CFLAGS. This preserves the respective native numerical contracts; it does not assert cross-language bitwise numerical equality.

All six adapters passed 65-query serial/batch/thread identity checks and save/reload checks before timing. TieredQ's selected rebuild also compares built versus reloaded predictions. Qdrant additionally compares production scalar-per-vector and contiguous-batch score bits for three queries across the full corpus at every width. Every timed repeat compares all 1,000 returned ID rows against its serial reference. Ground truth was independently checked with full-corpus FP32 scores and a full argsort for ten query rows. The final audit recomputes recall and timing statistics from raw artifacts for every scheduling row.

Recall rises across the measured anchors for both codecs and is identical across scheduling configurations. No recall variation was averaged away, and no profiling was needed. The lower TieredQ anchors use narrow lookup paths; its A4 learned layout includes 5/6-bit tiers and follows the existing FP32 code-ID fast path. This implementation distinction helps interpret the sharp throughput change between A2 and A4; no optimization experiment was added.

## Alignment limitations and scope

TieredQ A1's measured budgets 470 and 471 share the same serving allocation. Budget 472 exceeds its anchor; the selected 470 is the first of the equal-storage maximum feasible attempts. A4 budgets 2778, 2779, 2780 and 2784 share the largest measured feasible allocation; the adjacent observed budget 2785 exceeds the anchor. The selected 2778 is the first of these equal-storage attempts. Their 0.86% and 0.77% gaps cannot meet the preferred 0.5% in those observed brackets. A2 fits within 0.5%. No nominal-bit matching, interpolation, recall-based selection, other datasets, normal TieredQ mode, RaBitQ matrix, seed sweep or optimization experiments are included.

Other Hartmut services were observed and preserved. The host process snapshot is archived. These are single-seed measurements under that host state, with no statistical-confidence claims. The isolated workspace, compiled sources, saved indexes, calibration attempts and source datasets remain at `/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R`.
