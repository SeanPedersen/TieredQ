# TieredQ IVF-Flat SQ8 on PCA-128 + exact rerank, MS MARCO 1M, seed 0.
set -e
cd "$(dirname "$0")"
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-pca --suite 1m --truth-rows 10000 --seeds 0 \
  --only pca128-tq-sq8-L1024-p16-c30,pca128-tq-sq8-L1024-p32-c30
echo DONE
