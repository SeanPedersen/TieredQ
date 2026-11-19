#!/bin/sh
# #58 step 2 diagnostic: consecutive GEMM blocks of one super per work item (TQ_ITEM_BLOCKS), bound 224; 1 is step 1.
B=/home/sean/experiments/TieredQ-issue58-scan-20261119/probe/build/tieredq_assign_flops_benchmark
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
L=/home/sean/experiments/TieredQ-issue58-scan-20261119/items.log
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for blocks in 1 4 8 16 32; do
  echo "== TQ_ITEM_BLOCKS=$blocks" >> $L
  TQ_ITEM_BLOCKS=$blocks OMP_NUM_THREADS=32 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 224 2>&1 | grep -v '^peaks' >> $L
done
echo DONE >> $L
