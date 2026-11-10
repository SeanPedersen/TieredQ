# MSMARCO Qwen 8.84M scale study — completed

All seven full-corpus stages exited successfully on hartmut, finishing
2026-11-08 02:55 UTC. Refreshed SIFT controls also completed. Measurements
and frozen selections are in results/; results/summary.json aggregates query
repeats and checks hard caps. The full dataset has 8,840,823 vectors, 1,024D.

## Measured results

These are operating points selected for 95% calibration recall, evaluated on
750 held-out queries. Recall differs, so this is not an exact matched-recall
comparison. Latency is median p50 across three query repeats; one build/model.
GB means decimal GB. Serving RSS includes process/runtime allocations.

| Configuration | Held-out recall@10 | p50 ms | Build min | Serving RSS GB | Peak build RSS GB |
|---|---:|---:|---:|---:|---:|
| Adaptive4 graph SQ8 | 0.9449 | 30.00 | 12.21 | 9.22 | 18.34 |
| Fixed4 graph SQ8 | 0.9504 | 25.51 | 15.90 | 9.23 | 18.37 |
| Fixed4 flat SQ8 | 0.9557 | 26.90 | 15.90 | 9.23 | 18.37 |
| Fixed12 graph SQ8 | 0.9597 | 29.29 | 18.12 | 9.23 | 18.37 |
| Fixed12 flat SQ8 | 0.9597 | 26.96 | 18.12 | 9.23 | 18.37 |
| Faiss IVF-SQ8, nprobe64 | 0.9556 | 12.31 | 18.13 | 14.22 | 23.26 |
| Faiss IVFFlat FP32, nprobe96 | 0.9735 | 36.37 | 30.66 | 45.75 | 50.89 |

FP32 Faiss had approximately 4.89 GB swapped out and major faults during some
query repetitions. Its timing is affected by memory pressure and is not a clean
fully resident baseline. Compressed variants had zero swap and query major
faults. Lower prototype process RSS versus Faiss SQ8 is not a corresponding
index-size advantage: owned prototype index is ~9.124 GB, serialized Faiss SQ8
~9.157 GB. Faiss process RSS includes allocations/capacity retained by its runtime.

The current prototype does not beat Faiss IVF-SQ8: fixed4 graph takes ~2.07x
its p50 at slightly lower recall. Fixed12 graph is slower than its flat control
at identical recall. Four-iteration graph/flat selections differ in recall;
their latency ratio cannot establish a matched-recall graph advantage.

No prototype reaches 99% calibration recall within the tested maximum vector
cap of 442,041 (5% of the corpus). Diagnostic held-out recalls at that cap are
0.9751 adaptive4, 0.9765 fixed4, and 0.9788 fixed12. Their graph p50s are
65.02/63.22/63.17 ms; flat controls are 62.77/60.61/60.46 ms at identical recall.
Fixed SIFT scan settings fall to ~85–86% recall at this scale.
Faiss SQ8 reaches 0.9927 held-out recall at nprobe1024 / 160.56 ms; FP32 Faiss
reaches 0.9949 at nprobe512 / 180.37 ms, with the paging caveat above.
Calibration targets are not held-out guarantees: several selected points fall
below their nominal targets, explicitly recorded in summary.json.

Exhaustive search over the same 50 calibration queries gives FP32 recall1.000
and SQ8 recall0.996. Quantization alone therefore does not explain the larger
prototype routing loss on these queries. This is a diagnostic sample, not a
universal recall upper bound.

Next useful work is a controlled leaf-count sweep (e.g. 32K/64K leaves versus
8K), and profiling assignment/scanning kernels against Faiss. More Lloyd
iterations alone did not close the gap. Any future graph claim should use
paired budgets and recall; geometry metadata needs independent evidence before
adding it back. These are proposed next experiments, not completed results.

Source: https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/msmarco-qwen-1024-normalized.hdf5
Verified remote metadata: 8,840,823 x 1,024 FP32 train vectors, 1,000 queries,
100 supplied neighbors per query, normalized vectors, contiguous train payload
at byte offset 6144. Expected full-file SHA256:
efa21f8418e29137ad5be44171fd4657e1738caee86eb62645ad20605457cb2c.

## Protocol

The user requested INT8 storage for this scale evaluation. One full-population,
per-coordinate affine SQ8 codec is shared by all compressed configurations.
The conversion uses coordinate extrema from train vectors only, 256 signed-byte
levels, and per-coordinate FP32 scales/offsets. It does not normalize vectors
again. Codes decode as float32(code) * scale + offset. Both original source
hash and converted code/parameter hashes are recorded. Code payload is
9,053,002,752 bytes versus 36,212,011,008 bytes for the FP32 corpus; these are
payload sizes, not measurements of total serving RSS.

The INT8 corpus and parameter files are copied to the experiment directory on
NVMe after checking space and verifying SHA256. To make room, the older 1M-vector
MSMARCO NPZ cache was checksum-verified on /mnt/storage/tieredq-dataset-cache/
and its original path replaced with a symlink. cache-relocation.json records
that reversible move; no dataset contents were discarded.

All compressed configurations train partitions from decoded INT8 values. The
C++ prototype decodes coordinates into registers and keeps centroids FP32;
Faiss decodes only its bounded training sample and insertion batches. Neither
materializes a full decoded corpus. Geometry pruning remains disabled in this
routing comparison. Competitive scanned candidates are rescored in double
against decoded SQ8 vectors. Recall uses the original supplied FP32 truth.

The full corpus is retained; no 1M subset is used. Seed 42, 250 calibration
queries and 750 disjoint held-out queries. Budgets are calibrated on the former
and frozen before reading the latter. Targets: 0.95 / 0.99 / 0.995. Selected
query configurations have three repeats on one built index per model. Build
time is one observation per model, not three independent builds. Unreached
calibration targets are explicit; the highest-recall calibration configuration
is replayed as a labeled diagnostic rather than silently relaxing scan caps.

Configurations:

- Adaptive two-means, 4 iterations, graph and flat controls.
- Fixed hierarchical k-means, 4 iterations, graph and flat controls.
- Fixed hierarchical k-means, 12 iterations, graph and flat controls.
- Faiss IVF-SQ8, using the exact same converted code bytes and scale/offset
  parameters, with residual coding disabled.
- Faiss IVFFlat on the original FP32 vectors, as an uncompressed reference.

All prototype models keep 64 coarse groups / 8,192 leaves, graph degree 8,
4 graph search seeds, 4,096 training samples per clustering call, and target leaf
size 64. Fixed models use 128 fine children per coarse group. The grid includes
35K/50K absolute caps and 1%/2%/3.5%/5% caps of the full corpus. Graph metadata
budgets are 512/1024/2048/4096/8256; flat evaluates all leaves. Original SIFT
operating points are also explicitly replayed on held-out queries to distinguish
constant-work scaling from a new recall-calibrated operating point.

Separate exhaustive FP32 and SQ8 diagnostics on the same 50 calibration queries
validate the supplied ground truth and measure quantization loss without routing. Its recall is not a strict
upper bound on every approximate search policy's recall against FP32 truth.

Faiss trains 8,192 coarse centroids on 327,680 samples with four Lloyd iterations;
nprobe calibration covers 32/64/96/128/192/256/512/1024. Shared SQ8 bytes are
inserted using standalone codes and decoded-INT8 coarse assignments. Faiss
continues to use its native SIMD asymmetric SQ8 distance scanner. Floating-point
rounding in the two decoders may differ slightly despite common code levels.

## RAM, timing and validation

Input datasets are memory-mapped and released after index construction. Prototype
packing streams source rows in order, avoiding random disk reads after packed
storage displaces mapped source pages. Serving RSS, swap and major faults are
recorded separately from peak build RSS; file-backed input working sets and
anonymous index storage must not be conflated. Any paging during query timing
will be called out. Faiss serialization is counted with a bounded streaming
callback rather than allocating another whole-index copy.

Index build clocks include finite-coordinate validation. Dataset download,
conversion and NVMe copying are separate preparation phases; conversion time,
peak process RSS, quantization error and hashes are recorded in the manifest.
Single-thread queries/builds are pinned to CPU 2 on hartmut (AVX-512). Full
corpus runs do not overlap each other.

C++ tests and an end-to-end small fixture exercise mapping, native SQ8 scans,
shared Faiss decoding, frozen selection, input release and exhaustive SQ8 search.
Release and sanitizer checks are run before full-corpus timing. These preflights
are correctness checks, not MSMARCO performance evidence.

Reproduction: build source/ with CMake Release; run job.sh from the remote root.
It waits for download completion, verifies/prepares the corpus, copies SQ8 to
NVMe, then invokes study.py. Outputs are under results/; status.json and
job.exitcode distinguish progress, completion and failure. Source vectors and
large exports remain outside the repository. This workload changes corpus size,
dimension, distribution and precision relative to the earlier SIFT FP32 study;
latency ratios therefore do not isolate vector-count scaling alone.

The exact four-double-lane SIMD kernel is also reused in partition training and
graph construction; the SIFT SQ8 controls were refreshed with the same kernel.
See the sibling `hartmut-sift-sq8-20261107` report for those completed controls.
