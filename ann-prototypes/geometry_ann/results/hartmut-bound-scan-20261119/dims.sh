#!/bin/sh
# #58: bound dimensions sweep on step 2 (rescoring is cheaper now, so the best d' may be below 224).
B=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops2/build/tieredq_assign_flops_benchmark
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
L=/home/sean/experiments/TieredQ-issue58-scan-20261119/dims.log
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for dp in 224 128 160 192 256 224; do
  echo "== d'=$dp" >> $L
  OMP_NUM_THREADS=32 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 $dp 2>&1 | grep -v '^peaks' >> $L
done
echo DONE >> $L
