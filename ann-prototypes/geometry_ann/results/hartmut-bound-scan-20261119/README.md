# #58: faster bounded restricted assignment (hartmut, 2026-11-19)

Follow-up to #45 (`7761b5f`, FLOP baseline in `../hartmut-assign-flops-20261119/`). All runs: Qwen/MS MARCO
8.84M x 1024, 64K lists, 256 supers x 16 probes, bound 224, 32 threads on hartmut (i9-7980XE, AVX-512). FLOP harness
runs use the saved seed-0 64K centroids (`assign_flops_benchmark.cpp` plus an instrumented copy of the assigner
per variant, `tieredq_ivf_assign.*.profiled.h`). Each comparison ran in one window in ABBA order.

## Kept

1. **Vectorized bound selection** (`bound_survivors`, `tieredq_ivf_bound.h`): |Pc|^2 and |r_c| stored per super in
   member order, 16 candidates per AVX-512 compare with the scalar bound's operations in the same order (builds use
   `-ffp-contract=off`), so survivors are identical (55.77 per row in every run).
2. **Division-free slot grouping and a parallel per-row reduction** (`group_slots` by probe range, `assign`). A
   `perf` profile of step 1 (`perf-report.txt`) put 25% of all thread cycles in libgomp spin-wait: `group_slots` ran
   on one thread with a runtime `slot % probes` per slot (2 calls x 2 passes x 512K slots per 32K-row chunk) while
   31 threads waited. The harness had charged that time to the first probe and the bound scan.

| Final assignment (FLOP harness) | `7761b5f` | step 1 | step 2 |
|---|---:|---:|---:|
| probe supers | 3.87 s | 3.82 s | 3.95 s |
| first probe in full | 5.80 s (1,091 GFLOP/s) | 5.62 s | **3.54 s (1,786 GFLOP/s, 77% of peak)** |
| bound GEMM | 9.42 s | 10.29 s | 9.32 s (2,050 GFLOP/s) |
| bound scan + rescoring | 10.92 s (92 GFLOP/s) | 5.21 s | 5.10 s |
| reduce + fallback | 0.28 s | 0.29 s | 0.03 s |
| **total** | **34.54 s** | **29.37-29.68 s** | **25.90 s (-25%)** |
| peak RSS (harness) | 9.858 GB | 9.875 GB | 9.875 GB |

(`flops.log`: `7761b5f` vs step 1; `flops2.log`: step 1 vs step 2.)

Full builds (`bench.sh`, `full-builds/`), `TieredQNestedIVFIndex` #52 defaults, tier on NVMe, 750 test queries on
pinned CPU 2:

| | `7761b5f` seed 0 / 7 | steps 1+2 seed 0 / 7 | mean change |
|---|---:|---:|---:|
| build | 160.1 / 160.0 s | 152.4 / 152.1 s | **-7.8 s (-4.9%)** |
| final assignment | 37.18 / 36.62 s | 29.01 / 28.22 s | **-8.3 s (-22%)** |
| train / encode | 62.6, 49.6 / 62.3, 50.5 s | 62.7, 50.0 / 62.6, 50.8 s | - |
| recall@10, nprobe 96-1024 | | identical on both seeds | |
| peak build RSS | 15.418 / 15.452 GB | 15.514 / 15.548 GB | +96 MB (+0.6%) |

The RSS increase is not held by the assigner: sampled anon RSS (`rss.sh`, `rss/`) is identical through training (1.80
GB) and assignment (1.23 GB). After encode, whose code is unchanged, the baseline drops from 2.33 to 2.08 GB and steps
1+2 from 2.24 to 2.17 GB. The final save spike sits on that plateau. Most likely glibc arena retention after the
assigner's different allocation pattern (not proven). The baseline's own peak RSS ranged 15.06-15.42 GB across runs.
Accepted as allocator state outside the changed code.

## Tried and dropped

- **Smaller bound widths** (`dims.log`, on step 2): d' 128 / 160 / 192 / 224 / 256 -> 41.8 / 30.3 / 27.3 / 26.0 /
  26.4 s (survivors 544 / 246 / 113 / 56 / 30 per row). 224 stays.
- **Bounded phase over L3-sized row ranges** (`probe.log`, `tieredq_ivf_assign.subrange.profiled.h`): 32K / 8K / 4K
  / 2K rows -> 29.6 / 31.5 / 34.0 / 36.5 s. Re-reading x was not the cost.
- **Several GEMM blocks of one super per work item** (`items.log`, `tieredq_ivf_assign.itemblocks.profiled.h`): 1 /
  4 / 8 / 16 / 32 blocks -> 29.5 / 30.4 / 32.6 / 34.3 / 37.1 s (the first probe balances worse).
- **Per-thread scratch buffers** for the blocks and the super ranking (`flops3.log`,
  `tieredq_ivf_assign.scratch.profiled.h`): 25.92 -> 25.59 s (-1.3%) for +15 MB harness peak RSS. Not worth it.

## Remaining profile (step 2, harness)

Bound GEMM 9.3 s at 91% of the FP32 panel kernel's peak, probe supers 3.9 s (58%), first probe 3.5 s (77%), projection
2.7 s (sgemm), rescoring about 3.5 s (0.06 s per survivor per row), bound selection about 1.1 s. The GEMMs are near
peak, so further gains need fewer FLOPs (fewer candidates per row) rather than faster kernels. E-means training (62 s)
and encode (50 s) are now the largest build stages.
