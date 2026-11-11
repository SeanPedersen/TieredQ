#!/usr/bin/env bash
# Issue #22 task 3.1: train-question sample, the eval's PCA dump and tiers at 512/768/1024, then query energy per rank.
# Runs in ~/experiments/TieredQ-issue22-msmarco-20261031 with the eval binary of TieredQ-issue6-ivf-20261030.
set -euo pipefail
cd "$(dirname "$0")"
W=$PWD
(cd source/benchmark && OPENBLAS_NUM_THREADS=8 taskset -c 8-15 uv run --no-project --with numpy --with pyarrow --with h5py \
  python msmarco1m_data.py train-queries "$W/data" "$W/abdata/msmarco-train")
mkdir -p diagnostic
E=$HOME/experiments/TieredQ-issue6-ivf-20261030
TQ_TRAIN_DUMP=$W/diagnostic/pca taskset -c 8-15 "$E/bin/ivf_eval_refinement5" tieredq-nesting 512,768,1024 \
  "$W/abdata/msmarco1m-real" 1024 256 0 1 > diagnostic/nesting.txt 2> diagnostic/nesting.err
(cd source/benchmark && uv run --no-project --with numpy python experiments/issue22_query_alloc/query_energy.py \
  "$W/diagnostic/pca" "$W/diagnostic/nesting.txt" 1024 train real="$W/abdata/msmarco1m-real/queries.f32" \
  doc="$W/abdata/msmarco1m-doc/queries.f32" train="$W/abdata/msmarco-train/train_queries.f32") > diagnostic/energy.md
echo done > diagnostic/finished
