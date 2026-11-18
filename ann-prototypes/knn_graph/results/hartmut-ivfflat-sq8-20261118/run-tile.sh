# Second pass: SQ8 list tiles capped at 512 KiB (L2-resident) instead of 1,024 rows; SQ8 first (B-A after run.sh's A-B).
set -e
cd "$(dirname "$0")"
cmake --build cbuild -j 16 --target test_tieredq > cbuild-tile.log
OMP_NUM_THREADS=8 cbuild/test_tieredq > ctest-tile.log 2>&1
grep "flat IVF" ctest-tile.log
uv sync --reinstall-package tieredq > sync-tile.log 2>&1
for seed in 0 1; do
  uv run python compare.py ../data/dbpedia100k.npy ../results/dbpedia100k-tile --suite 100k --seeds $seed \
    --only ivfflat-sq8-L128-p32,ivfflat-L128-p32
done
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-tile --suite 1m --truth-rows 10000 --seeds 0 \
  --only ivfflat-sq8-L512-p8,ivfflat-L512-p8
echo DONE
