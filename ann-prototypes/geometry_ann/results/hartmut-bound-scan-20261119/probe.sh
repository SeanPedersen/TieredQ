#!/bin/sh
# #58 step 2 diagnostic: bounded probes over row sub-ranges (TQ_BOUND_ROWS), bound 224; 32768 is step 1 itself.
B=/home/sean/experiments/TieredQ-issue58-scan-20261119/probe/build/tieredq_assign_flops_benchmark
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
L=/home/sean/experiments/TieredQ-issue58-scan-20261119/probe.log
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for rows in 32768 8192 4096 2048; do
  echo "== TQ_BOUND_ROWS=$rows" >> $L
  TQ_BOUND_ROWS=$rows OMP_NUM_THREADS=32 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 224 2>&1 | grep -v '^peaks' >> $L
done
echo DONE >> $L
