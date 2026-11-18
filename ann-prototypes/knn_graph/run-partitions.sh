# Issue #45 follow-up: Lloyd vs e-means (Faiss contrib, vendored) vs s-means lists for TieredQIVFFlatIndex.knn_graph.
# DBpedia 100K (256 lists) on 2 seeds, then MS MARCO 1M (1,024 lists) on seed 0; 8 threads, k = 15.
set -e
cd "$(dirname "$0")"
cmake -S tieredq -B cbuild -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_TESTS=ON -DTIEREDQ_EXAMPLES=OFF > cmake.log
cmake --build cbuild -j 16 --target test_tieredq > cbuild.log
OMP_NUM_THREADS=8 cbuild/test_tieredq > ctest.log 2>&1
grep "flat IVF" ctest.log
(cd vendor && sha256sum -c --quiet ../vendor-sha256.txt)  # Faiss contrib e_means at commit fc1095a
uv run python partition_study.py ../data/dbpedia100k.npy ../results/dbpedia100k --lists 256 --probes 8,16,32 \
  --seeds 0,1 --emeans-path vendor
uv run python partition_study.py ../data/msmarco1m.npy ../results/msmarco1m --lists 1024 --probes 8,16,32 \
  --seeds 0 --truth-rows 10000 --emeans-path vendor
echo DONE
