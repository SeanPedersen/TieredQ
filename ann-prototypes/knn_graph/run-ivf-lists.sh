# Issue #45: native e-means lists vs TieredQIVFIndex's own Lloyd lists, MS MARCO 1M with its 1,000 queries, seed 0.
# 2,048 lists (automatic count at 1M, exact Lloyd recipe) and 8,192 (fast recipe). 8 threads.
set -e
cd "$(dirname "$0")"
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/data
for spec in 2048:8,16,32,64,128 8192:16,32,64,128,256; do
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 uv run python ivf_lists_check.py ../data/msmarco1m.npy \
    ../results/msmarco1m-queries/query_truth.npz ../results/ivf-lists.jsonl \
    --queries $Q/calibration_queries.fvecs,$Q/test_queries.fvecs --lists ${spec%%:*} --probes ${spec#*:} \
    --only lloyd-auto,emeans-s64-e2.5,emeans-s256-e2.5
done
echo DONE
