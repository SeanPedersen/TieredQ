# SIFT1M SQ8 control for the MSMARCO scale study

Completed on hartmut, i9-7980XE (AVX-512), CPU 2, GCC 16.2.1, CMake Release.
Same full SIFT1M corpus and seed-42 query split as the earlier FP32 study:
250 calibration / 1,000 held-out queries. No extra normalization. One built
index per model, three query repeats. This control uses the updated exact SIMD
training kernel also used for MSMARCO; it is not a pure codec-only build-time
ablation against the earlier FP32 implementation. The corpus download continued
in the background; other benchmark processes did not overlap.

The shared learned per-dimension min/max affine SQ8 codec stores 128 MB of code
payload instead of 512 MB of FP32 vectors. Prototype fixed4 estimated owned
storage is 138.27 MB including metadata. Its post-input-release process RSS is
173.98 MB; build-process peak RSS is 819.50 MB. Faiss SQ8 process RSS is 299.96 MB,
including Python/NumPy/runtime and allocator overhead. These are process-memory
measurements, not allocator-isolated C++ index footprints. All query batches
record zero swap and zero major faults.

## Precision and query results

| Frozen SIFT FP32 operating point replayed with SQ8 scans | Held-out recall@10 | Median p50 ms | Build seconds |
| --- | ---: | ---: | ---: |
| Adaptive4 graph, metadata 2048 / 50K vectors | 0.9820 | 2.101 | 14.50 |
| Fixed4 graph, metadata 2048 / 35K vectors | 0.9815 | 1.656 | 15.75 |
| Fixed4 flat heap, 35K vectors | 0.9820 | 1.848 | 15.75 |
| Fixed12 graph, metadata 2048 / 35K vectors | 0.9813 | 1.648 | 28.28 |

No prototype or Faiss SQ8 calibration point meets recall 0.99 or 0.995 on the
specified grid. Those targets remain marked NOT_REACHED; the highest-recall
calibration settings are replayed as diagnostics. For fixed4, a highest-recall
50K-vector/full-centroid diagnostic reaches 0.9847 on held-out queries. The
fixed4 graph configuration selected for 0.95 calibration recall only reaches
0.9438 on held-out queries; it must not be presented as achieving that target.
See summary.json for all selected points, failures, work and memory counters.

Faiss SQ8 selects nprobe 96 for the 0.95 target: held-out recall 0.9589 at
0.515 ms. Its highest-recall calibration budget, nprobe 256, is also replayed as
a diagnostic: held-out recall 0.9828 at 1.025 ms. Faiss still wins latency.
These are not exactly matched-recall comparisons.

Exhaustive searches on the SAME first 50 calibration queries yield:

| Storage/metric | Recall against supplied FP32 ground truth |
| --- | ---: |
| FP32, complete scan | 1.000 |
| Learned affine SQ8, complete scan | 0.986 |

This identifies quantization loss separately from routing misses on this sample.
It is not proof of an unavoidable INT8 ceiling: SIFT's integer coordinates can
be encoded losslessly with a suitable byte grid. This control deliberately uses
the same general learned min/max SQ8 procedure as the continuous-valued Qwen
corpus. No codec parameters were tuned on held-out queries. Quantization MSE is
0.032008 per coordinate, maximum absolute error 0.42745; hashes and parameters
are in dataset-manifest.json.

## Reproduction

Remote root: /home/sean/experiments/TieredQ-issue44-sift-sq8-20261107.
Build/source link to the isolated MSMARCO study's implementation. Prepare with
prepare_hdf5_view.py, --int8 --calibration 250 --test 1000 and the source hash
from dataset-manifest.json; run the companion sift_sq8_study.py from the MSMARCO
artifact directory as study.py. The runner shares a benchmark lock with the
large-corpus study. The separate FP32 oracle command is:

```sh
taskset -c 2 build/sq8_oracle --base data/base.rawview \
  --queries data/calibration_queries.fvecs --truth data/calibration_truth.ivecs --nq 50
```

C++ correctness checks and the end-to-end SQ8/Faiss preflight pass on hartmut;
local ASan/UBSan checks also pass. Source vectors, quantized codes and large
exports are not committed. The production library is untouched.
