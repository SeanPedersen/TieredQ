#!/bin/sh
# #45 pruning prototype timing: 8.84M x 1024, 64K lists, e-means all rows x 2 epochs, 256 supers x 4 probes, 32 threads;
# TQ_PRUNE_DIMS 0 (current) vs 128 / 192, alternating for host noise. Takes the benchmark lock.
E=/home/sean/experiments/TieredQ-issue45-pruning-20261118
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
B=$E/source/build/tieredq_coarse_training_benchmark
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for dims in 0 128 192 128 0; do
  echo "dims $dims" >> $E/results/prune64k.profile
  TQ_PRUNE_DIMS=$dims OMP_NUM_THREADS=32 TQ_PROFILE=1 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 65536 0 emeans 2 256 4 0 \
    2>> $E/results/prune64k.profile | sed "s/^{/{\"prune_dims\":$dims,/" >> $E/results/prune64k.jsonl || echo FAILED >> $E/results/prune64k.jsonl
done
echo DONE >> $E/results/prune64k.jsonl
