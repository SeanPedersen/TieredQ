#!/bin/sh
# #45: FLOP rates of the final restricted assignment stages (bound 224, then off) on 8.84M x 1024, 64K saved seed-0
# centroids, 32 threads. Takes the benchmark lock.
E=/home/sean/experiments/TieredQ-issue45-flops-20261119
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for bound in 224 0; do
  OMP_NUM_THREADS=32 $E/source/build/tieredq_assign_flops_benchmark $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 $bound >> $E/flops.log 2>&1
done
echo DONE >> $E/flops.log
