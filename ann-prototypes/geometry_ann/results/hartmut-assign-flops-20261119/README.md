# #45: FLOP rates of the final restricted assignment (hartmut, 2026-11-19)

`assign_flops_benchmark.cpp` with the instrumented `tieredq_ivf_assign.profiled.h` (source `7761b5f` plus timers and
FLOP counters; padded panel columns counted as executed): all 8.84M Qwen/MS MARCO rows to the saved seed-0 64K
centroids, 256 supers x 16 probes, 32 threads (`flops.sh`, `flops.log`). Peaks on the same threads: OpenBLAS sgemm
4096 x 4096 x 1024 1.70-1.74 TFLOP/s, binary16 panel kernel (depth 1024, in-cache 64-row blocks) 2.33-2.37 TFLOP/s,
FP32 panel kernel (depth 224) 2.27 TFLOP/s.

| Stage | bound 224: time | GFLOP | GFLOP/s (% of panel peak) | bound off: time | GFLOP/s |
|---|---:|---:|---:|---:|---:|
| probe supers (256, padded 288) | 3.84 s | 5,215 | 1,358 (58%) | 4.36 s | 1,197 |
| project rows (sgemm, d' 224) | 2.71 s | 4,056 | 1,497 (88% of sgemm) | - | - |
| first probe in full | 5.75 s | 6,322 | 1,100 (47%) | - | - |
| limits (row norms) | 0.64 s | - | - | - | - |
| bound GEMM (15 probes) | 9.28 s | 19,106 | 2,060 (91%) | - | - |
| bound scan + rescoring | **10.60 s** | 1,010 | **95** | - | - |
| all probes in full | - | - | - | 47.40 s | 1,976 (84%) |
| total | 33.95 s | 35,707 | 1,052 | 52.55 s | 1,882 |

Candidates 4,479 per row, survivors 55.8 (1.25%). The bound removed 64% of the FLOPs and 35% of the time: the
GEMMs run at 47-91% of peak, but the per-candidate scan (indirect per-list loads of |Pc|^2 and |r_c|, scalar
compare, 40 G candidate checks) and the rescoring of survivors take 31% of the stage at 95 GFLOP/s.
