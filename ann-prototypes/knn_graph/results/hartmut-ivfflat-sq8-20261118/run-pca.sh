# Fourth pass: PCA-256 + candidate graph + exact rerank around each builder, MS MARCO 1M, seeds 0 and 1.
set -e
cd "$(dirname "$0")"
C=pca256-nnd-c30,pca256-rinnd-c30,pca256-mfrs-nnd-c30,pca256-mfrs-ivf-p31-c30,pca256-mfrs-ivf-p62-c30
C=$C,pca256-tq-sq8-L1024-p16-c30,pca256-tq-sq8-L1024-p32-c30,pca256-tq-L1024-p16-c30,pca256-nnd-c45,pca256-rinnd-c45
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-pca --suite 1m --truth-rows 10000 --seeds 0,1 \
  --only $C
echo DONE
