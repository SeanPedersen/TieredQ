# Native e-means port (cpp-emeans-*) against Lloyd, all trained inside the index. First MS MARCO 1M with its own
# queries (250 calibration + 750 test): search recall@10 at equal rows scanned per query, seeds 0 and 1. Then the
# knn_graph study at 100K and 1M, seeds 0 and 1. (The C++ unit tests passed locally on this source.)
set -e
cd "$(dirname "$0")"
uv sync --reinstall-package tieredq > sync-queries.log 2>&1
uv run --with pytest python -m pytest -q tieredq/tests/python/test_ivf_flat.py
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107/data
NATIVE=cpp-emeans-s256-e2.5,cpp-emeans-s640-e1
uv run python partition_study.py ../data/msmarco1m.npy ../results/msmarco1m-queries --lists 1024 --k 10 \
  --probes 4,8,16,32,64 --seeds 0,1 --emeans-path vendor \
  --queries $Q/calibration_queries.fvecs,$Q/test_queries.fvecs --only $NATIVE,lloyd-s64-i10,lloyd-s256-i10
uv run python partition_study.py ../data/dbpedia100k.npy ../results/dbpedia100k --lists 256 --probes 8,16,32 \
  --seeds 0,1 --emeans-path vendor --only $NATIVE
uv run python partition_study.py ../data/msmarco1m.npy ../results/msmarco1m --lists 1024 --probes 8,16,32 \
  --seeds 0,1 --truth-rows 10000 --emeans-path vendor --only $NATIVE
echo DONE
