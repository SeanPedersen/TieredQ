#!/bin/sh
# #45 check 2 after check 1: PCA lower-bound pruning rate (pruning_rate.py) on the saved seed-0 64K centroids.
E=/home/sean/experiments/TieredQ-issue45-pruning-20261118
until grep -q DONE $E/results/revival.jsonl; do sleep 10; done
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
cd $E && UV_CACHE_DIR=/home/sean/experiments/uv-cache OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
  uv run -q --no-project --with numpy python -I pruning_rate.py $E/results/centroids-64k-seed0.f32 4000 \
  > $E/results/pruning.jsonl 2> $E/results/pruning.err; echo "exit $?" >> $E/results/pruning.err
