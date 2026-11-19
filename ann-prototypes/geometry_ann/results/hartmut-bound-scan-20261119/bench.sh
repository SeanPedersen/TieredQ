#!/bin/sh
# #58 steps 1+2: TieredQNestedIVFIndex #52 defaults (bound 224) on Qwen/MS MARCO 8.84M x 1024, 64K lists, 32 build
# threads, tier on NVMe, 750 queries on pinned CPU 2. base = 7761b5f, new = vectorized bound scan + division-free slot
# grouping and parallel reduction. Order ABBA over seeds 0 / 7. Takes the benchmark lock; resumable per step.
set -e
E=/home/sean/experiments/TieredQ-issue58-scan-20261119
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
SSD=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/ssd
export UV_CACHE_DIR=/home/sean/experiments/uv-cache
BASE=/home/sean/experiments/TieredQ-issue45-bound-20261119/source
NEW=$E/source
NPROBES=96,128,192,256,512,1024
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for spec in base:0 new:0 new:7 base:7; do
  mode=${spec%:*}; seed=${spec#*:}; name=$mode-seed$seed; out=$E/results/$name; index=$SSD/issue58-$name.tq; tier=$SSD/issue58-$name.rows
  [ $mode = base ] && src=$BASE || src=$NEW
  PY="uv run --project $src --no-dev --with threadpoolctl python"
  mkdir -p $out/build $out/serve
  [ -f $out/build/build.json ] || OMP_NUM_THREADS=32 $PY $E/build.py $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024     $index $tier $out/build/build.json $seed > $out/build/run.log 2>&1
  [ -f $out/serve/run.csv ] || OMP_NUM_THREADS=1 taskset -c 2 $PY $E/serve.py $index $tier $Q/data/test_queries.fvecs     $Q/data/test_truth.ivecs 750 $NPROBES $out/serve/run.csv > $out/serve/run.log 2>&1
  rm -f $index $tier
  echo "done $name"
done
echo ALLDONE
