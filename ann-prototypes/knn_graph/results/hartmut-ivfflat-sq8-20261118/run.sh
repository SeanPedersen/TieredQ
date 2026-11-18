# SQ8 vs FP32 row storage for TieredQIVFFlatIndex.knn_graph; each SQ8 run directly after its FP32 twin.
set -e
cd "$(dirname "$0")"
cmake -S tieredq -B cbuild -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_TESTS=ON -DTIEREDQ_EXAMPLES=OFF > cmake.log
cmake --build cbuild -j 16 --target test_tieredq > cbuild.log
OMP_NUM_THREADS=8 cbuild/test_tieredq > ctest.log 2>&1
grep "flat IVF" ctest.log
uv sync > sync.log 2>&1
B=ivfflat-L128-p32,ivfflat-sq8-L128-p32,ivfflat-L256-p16,ivfflat-sq8-L256-p16,ivfflat-L512-p32,ivfflat-sq8-L512-p32
uv run python compare.py ../data/dbpedia100k.npy ../results/dbpedia100k --suite 100k --only $B
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m --suite 1m --truth-rows 10000 --seeds 0 \
  --only ivfflat-L512-p8,ivfflat-sq8-L512-p8
echo DONE
