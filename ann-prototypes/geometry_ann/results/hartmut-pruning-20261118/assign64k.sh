#!/bin/sh
# #45 final-assignment pruning prototype: all 8.84M rows to the saved seed-0 64K centroids, 256 supers x 16 probes,
# 32 threads; TQ_PRUNE_DIMS 0 / 192 / 128 / 0. Labels per run in labels/. Takes the benchmark lock.
E=/home/sean/experiments/TieredQ-issue45-pruning-20261118
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
B=$E/source/build/tieredq_restricted_assign_benchmark
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
i=0
for dims in 0 192 128 0; do
  i=$((i+1))
  TQ_PRUNE_DIMS=$dims OMP_NUM_THREADS=32 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $E/results/centroids-64k-seed0.f32 \
    65536 256 16 0 $E/labels/run$i-d$dims.u32 >> $E/results/assign64k.jsonl 2>> $E/results/assign64k.err || echo FAILED >> $E/results/assign64k.jsonl
done
echo DONE >> $E/results/assign64k.jsonl
