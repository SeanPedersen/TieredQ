#!/bin/sh
# #45 check 1: e-means revival threshold (0.1 default vs 0.2 / 0.3, the SuperEMeans post) at 8.84M x 1024, 64K lists,
# all rows x 2 epochs, 256 supers x 4 probes, 32 threads; held-out objective + imbalance. Saves seed-0 default centroids.
E=/home/sean/experiments/TieredQ-issue45-pruning-20261118
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
B=$E/source/build/tieredq_coarse_training_benchmark
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for spec in 0.1:0 0.3:0 0.2:0 0.1:7 0.3:7; do
  t=${spec%:*}; seed=${spec#*:}
  save=""; [ "$t" = 0.1 ] && [ "$seed" = 0 ] && save=$E/results/centroids-64k-seed0.f32
  echo "threshold $t seed $seed" >> $E/results/revival.profile
  TQ_EMEANS_UNUSED_THRESHOLD=$t TQ_SAVE_CENTROIDS=$save OMP_NUM_THREADS=32 TQ_PROFILE=1 $B $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 65536 0 emeans 2 256 4 $seed \
    | sed "s/^{/{\"unused_threshold\":$t,/" >> $E/results/revival.jsonl 2>> $E/results/revival.profile || echo "FAILED $spec" >> $E/results/revival.jsonl
done
echo DONE >> $E/results/revival.jsonl
