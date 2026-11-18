# rinnd on PCA-128 + exact rerank, MS MARCO 1M, seed 0 (pynndescent's PCA-128 runs are from 2026-11-17).
set -e
cd "$(dirname "$0")"
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-pca --suite 1m --truth-rows 10000 --seeds 0 \
  --only pca128-rinnd-c30,pca128-rinnd-c45
echo DONE
