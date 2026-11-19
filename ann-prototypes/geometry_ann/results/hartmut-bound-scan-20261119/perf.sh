#!/bin/sh
# #58 step 2 diagnostic: where the cycles go in the step-1 assigner (user-space cycles per symbol, bound 224).
E=/home/sean/experiments/TieredQ-issue58-scan-20261119
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
cd $E && OMP_NUM_THREADS=32 perf record -F 499 -e cycles:u -o perf.data $E/flops/build/tieredq_assign_flops_benchmark $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 224 > perf-run.log 2>&1
perf report -i perf.data --no-children --sort symbol --stdio 2>/dev/null | grep -v '^#' | grep -v '^$' | head -25 > perf-report.txt
echo DONE >> perf-report.txt
