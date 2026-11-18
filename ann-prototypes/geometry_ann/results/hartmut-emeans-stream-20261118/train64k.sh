#!/bin/sh
# Train-only 64K-list runs: Lloyd fast vs streamed e-means (args: SEED, then trainer:rows_per_list:budget triples).
B=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/source/build/tieredq_coarse_training_benchmark
A='/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/base.i8 /home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/scales.f32 /home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/offsets.f32 1024 65536'
seed=$1; shift
for spec in "$@"; do
  IFS=: read trainer per budget probes <<X
$spec
X
  OMP_NUM_THREADS=32 TQ_PROFILE=1 $B $A $per $trainer $budget 256 ${probes:-16} $seed >> results/train64k.jsonl 2>> results/train64k.profile
done
