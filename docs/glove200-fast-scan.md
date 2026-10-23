# GloVe200: exact and fast single-core x86 scans

Measured on hartmut (Intel i9-7980XE), pinned to CPU 2, one query and one search
thread. Both modes perform a complete scan. There is no reranking or candidate
generation. ARM was not changed during this experiment.

“Exact” below means the existing TieredQ FP32 scorer over the compressed index.
It does **not** mean an uncompressed-vector search with perfect recall.

Two quality measurements answer different questions:

- **Overlap@10 versus exact:** how closely fast mode reproduces the old scorer.
- **Ground-truth recall@10:** how well each mode retrieves the uncompressed cosine neighbors.

The current design keeps exact as the default and fast as an explicit opt-in. The
user accepted an approximate fast kernel, including overlap losses of about 0.05,
and prioritized single-core latency and throughput over a strict 0.01 overlap-loss
constraint. All measured byte-lookup overlap losses below are under 0.021.

## Distance-kernel timing

A separate scan-only experiment prepared each query outside the timer and excluded
top-k. The exact path used `codec.score_planes(..., automatic, false)`; the fast
path used the integrated byte-lookup kernel in an isolated namespace. Both wrote
all 100k scores, including centroid/factor/correction terms. Queries and original
compressed rows were identical. These are single-core full-scan timings, not the
cost of one document distance and not batched-search measurements.

| Code bits | Bits/input dimension | Exact scan p50 | Fast scan p50 | Scan speedup | Overlap@10 |
|---:|---:|---:|---:|---:|---:|
| 400 | 2 | 0.761 ms | 0.349 ms | 2.18x | 0.9818 |
| 800 | 4 | 1.296 ms | 0.608 ms | 2.13x | 0.9818 |

## End-to-end query timing

Timings include query preparation, score-buffer allocation, full scan and top-k.
Each path was warmed independently with 64 queries. The 100k cases use three
repetitions of 1,000 queries; the full-base cases use one repetition of 1,000.

| Documents | Code bits | Exact p50 | Fast p50 | Speedup | Exact GT recall@10 | Fast GT recall@10 | Overlap@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 400 | 0.863 ms | 0.412 ms | 2.10x | 0.6674 | 0.6661 | 0.9818 |
| 100,000 | 600 | 1.413 ms | 0.648 ms | 2.18x | 0.8202 | 0.8202 | 0.9826 |
| 100,000 | 800 | 1.407 ms | 0.642 ms | 2.19x | 0.9009 | 0.9007 | 0.9818 |
| 1,183,514 | 400 | 12.058 ms | 6.511 ms | 1.85x | 0.6633 | 0.6640 | 0.9816 |
| 1,183,514 | 800 | 19.710 ms | 11.199 ms | 1.76x | 0.8999 | 0.8981 | 0.9795 |

The new lookup path gives about 2.1–2.2x on this 100k fixture and about 1.8x on
the full base. It does not establish a universal 2x win or sub-ms latency at a
million documents. These are warm separate-path measurements; competing exact/fast
cache traffic can change latency. No batched QPS claim follows from these results.

The models used the normal `max_bits=8` training default. Learned tiers were:

- 400 bits: one coordinate at three bits, 198 at two bits, one at one bit.
- 600 bits: all 200 coordinates at three bits.
- 800 bits: all 200 coordinates at four bits.

All therefore qualify for the new byte-lookup kernel without a special training
cap. An additional 800-bit `max_bits=4` run produced the same quality and comparable
timing (0.637 ms fast vs 1.406 ms exact).

## Wider-tier fallback

The 1,200-bit model assigned six bits to each coordinate. It uses the existing
expanded int8-level AVX-512BW fallback, not the new nibble lookup kernel.

| Documents | Exact p50 | Fast p50 | Speedup | Exact GT recall@10 | Fast GT recall@10 | Overlap@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 2.149 ms | 1.059 ms | 2.03x | 0.9709 | 0.9647 | 0.9782 |
| 1,183,514 | 28.921 ms | 17.539 ms | 1.65x | 0.9720 | 0.9634 | 0.9759 |

## Precision experiments

Tuning used isolated header copies; the working kernel and ARM path were retained.
Reducing the number of tables sharing a scale improved overlap but cost latency.
Coarsening byte-table quantization allowed replacing pair multiply-adds with byte
addition and widening, but did not produce a convincing speed gain.

| Variant | 400-bit fast p50 | 400-bit overlap | 800-bit fast p50 | 800-bit overlap |
|---|---:|---:|---:|---:|
| Signed 8-bit table, 16 groups/scale | 0.424 ms | 0.9846 | 0.648 ms | 0.9834 |
| Signed 8-bit table, 8 groups/scale | 0.455 ms | 0.9865 | 0.691 ms | 0.9842 |
| Signed 8-bit table, 4 groups/scale | 0.520 ms | 0.9888 | 0.827 ms | 0.9874 |
| Signed 7-bit table, 32 groups/scale | 0.425 ms | 0.9655 | 0.653 ms | 0.9608 |
| Signed 7-bit table, 64 groups/scale | 0.418 ms | 0.9599 | 0.637 ms | 0.9583 |
| Signed 7-bit table, 128 groups/scale | 0.415 ms | 0.9527 | 0.623 ms | 0.9578 |

The retained implementation uses signed 8-bit lookup scores and 32 groups per
scale. Its better recall at similar latency makes it preferable to these coarser
variants. These table-score precisions are separate from the model's code bits per
coordinate.

## Cache sizes and protocol

Approximate primary auxiliary-cache storage, excluding small level books, group
descriptors, vector/object overhead and query buffers:

| Documents | 400-bit cache | 800-bit cache | 1,200-bit fallback cache |
|---:|---:|---:|---:|
| 100,000 | 6.28 MiB | 10.95 MiB | 19.93 MiB |
| 1,183,514 | 68.28 MiB | 123.60 MiB | 235.90 MiB |

The original compressed payload remains allocated. Hartmut has 24.8 MB L3, so
cache-resident 100k results do not extrapolate directly to the full base.

Source data already existed on hartmut under
`~/experiments/diskovec-ntu-20261021/work/ntu-comparison/rabitq-turboquant-comparison/ann/data/`.
The source is ANN-Benchmarks `glove-200-angular.hdf5`. Base/query vectors are
L2-normalized. The 100k subset uses the existing seed-42 sample and recomputed
cosine neighbors; the full case uses all 1,183,514 base vectors and the source
angular neighbors. Four query rows per base were independently checked with FP32
brute-force dot products. No query is used as a training row.

Training used 20,000 evenly spaced base rows, 256 centroids, tier rotation enabled,
seed 42. Every exact/fast comparison uses the same trained index. Search always
uses one pinned core; training/encoding used 16 OpenMP workers.

Raw inputs, trained indexes, manifest, preparation script and experiment sources are
preserved on hartmut under
`~/experiments/TieredQ-int8-x86-20261023/glove200/`. Structured end-to-end results
are in [glove200-fast-scan-results.json](benchmarks/glove200-fast-scan-results.json).

The checked-in benchmark now accepts an optional raw ground-truth file: ten
little-endian uint32 neighbor ids per selected query. Reproduce the 100k result:

```sh
cd ~/experiments/TieredQ-int8-x86-20261023/src
cmake --build build --target tieredq_fast_scan_benchmark -j36
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 \
  build/tieredq_fast_scan_benchmark ../glove200/glove_100k_800_cap8.index \
  ../glove200/queries.f32 1000 separate 0 ../glove200/truth_100k.u32
```

An independent run through this public benchmark reproduced 0.619 ms fast versus
1.399 ms exact, overlap 0.9818, exact GT recall 0.9009 and fast GT recall 0.9007.
Timing variation does not change the quality measurements.
