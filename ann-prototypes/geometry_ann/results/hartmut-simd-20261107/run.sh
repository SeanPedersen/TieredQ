#!/bin/sh
# Run from the remote experiment root after building fast/ and reference/.
set -eu
mkdir -p results
for run in 0 1 2; do
  for mode in reference fast; do
    taskset -c 2 "$mode/graph_bench" --base data/base.fvecs \
      --queries data/test_queries.fvecs --truth data/test_truth.ivecs --nq 1000 \
      --coarse 64 --leaves 8192 --target-size 64 --split kmeans \
      --directions 4 --graph-degree 8 --shared-directions 8 \
      --settings settings.csv >"results/$mode-$run.csv" 2>"results/$mode-$run.log"
  done
  taskset -c 2 uv run --python 3.12 source/compare_faiss.py \
    --base data/base.fvecs --queries data/test_queries.fvecs \
    --truth data/test_truth.ivecs --nq 1000 --nlist 8192 --sample 327680 \
    --variant ivf --budgets 128,256 >"results/faiss-ivf-$run.csv" 2>"results/faiss-ivf-$run.log"
done
