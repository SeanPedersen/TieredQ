#!/bin/sh
# Issue #45, streamed e-means at 1M: TieredQIVFIndex (MS MARCO 1M, its 1,000 queries, 8 threads) with the automatic
# TieredQCoarseTraining.lloyd recipe vs the new automatic e-means one (256 rows per list x 2.5 epochs exact below 8,192
# lists, from there 2 epochs with 4 probes in training and 16 in the final pass). Seeds 0 and 1, same window.
set -e
cd "$(dirname "$0")/knn"
uv sync --reinstall-package tieredq > sync.log 2>&1
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/data
O=/home/sean/experiments/TieredQ-issue45-emeans-ivfflat-20261118
for seed in 0 1; do
  for spec in 8192:16,32,64,128,256 2048:8,16,32,64,128; do
    lists=${spec%%:*}; probes=${spec#*:}
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 uv run python ivf_lists_check.py $O/data/msmarco1m.npy \
      $O/results/msmarco1m-queries/query_truth.npz ../results/ivf1m.jsonl \
      --queries $Q/calibration_queries.fvecs,$Q/test_queries.fvecs --lists $lists --probes $probes --seed $seed \
      --only lloyd-auto,auto
  done
done
echo DONE
