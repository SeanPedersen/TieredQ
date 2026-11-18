# Issue #45: native e-means coarse training (TieredQCoarseTraining.emeans_epochs) in TieredQIVFIndex vs its automatic
# Lloyd training, MS MARCO 1M with its 1,000 queries, seed 0, 8 threads. 2,048 lists (exact assignment) and 8,192
# (the fast recipe's 256 x 16 restricted assignment, now inside e-means' batches too).
set -e
cd "$(dirname "$0")"
uv sync --reinstall-package tieredq > sync-native.log 2>&1
uv run --with pytest python -m pytest -q tieredq/tests/python/test_ivf_flat.py
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/data
for spec in 8192:16,32,64,128,256 2048:8,16,32,64,128; do
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 uv run python ivf_lists_check.py ../data/msmarco1m.npy \
    ../results/msmarco1m-queries/query_truth.npz ../results/ivf-lists-native.jsonl \
    --queries $Q/calibration_queries.fvecs,$Q/test_queries.fvecs --lists ${spec%%:*} --probes ${spec#*:} \
    --only lloyd-auto,emeans-s64-e2.5,emeans-s256-e2.5
done
echo DONE
