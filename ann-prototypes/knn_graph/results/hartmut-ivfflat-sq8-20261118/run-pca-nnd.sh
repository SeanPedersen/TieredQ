# PCA-256 + NN-descent + exact rerank: pynndescent and rinnd only, MS MARCO 1M, seed 1 (seed 0 in run-pca.sh).
set -e
cd "$(dirname "$0")"
while pgrep -f "worker pca256-rinnd-c45" > /dev/null; do sleep 5; done
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-pca --suite 1m --truth-rows 10000 --seeds 1 \
  --only pca256-nnd-c30,pca256-rinnd-c30,pca256-nnd-c45,pca256-rinnd-c45
echo DONE
