# MS MARCO 1M seed 1 for the comparisons that decide the default: Lloyd 64 x 10 (default) and 256 x 10 against
# e-means at the default budget (256 x 2.5, 640 x 1).
set -e
cd "$(dirname "$0")"
uv run --with pytest python -m pytest -q tieredq/tests/python/test_ivf_flat.py
uv run python partition_study.py ../data/msmarco1m.npy ../results/msmarco1m --lists 1024 --probes 8,16,32 \
  --seeds 1 --truth-rows 10000 --emeans-path vendor \
  --only lloyd-s64-i10,emeans-s256-e2.5,lloyd-s256-i10,emeans-s640-e1
echo DONE
