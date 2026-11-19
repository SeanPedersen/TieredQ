#!/bin/sh
# #45: full builds with final-assignment pruning (TQ_PRUNE_ASSIGN_DIMS 0 vs 224), otherwise as the revival study.
# emeans-stream study's bench.sh: build on 32 threads from the SQ8 memmap, tier on NVMe, 750 held-out queries on pinned
# CPU 2. Order ABBA over (threshold, seed); resumable per step. Takes the benchmark lock.
set -e
E=/home/sean/experiments/TieredQ-issue45-revival-build-20261118
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
SSD=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/ssd
NPROBES=96,128,192,256,512,1024
PY="uv run --project $E/source --no-dev python"
export UV_CACHE_DIR=/home/sean/experiments/uv-cache
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for spec in 0:0 224:0 224:7 0:7; do
  t=${spec%:*}; seed=${spec#*:}; name=prune$t-seed$seed; out=$E/results/$name; index=$SSD/rev-$name.tq; tier=$SSD/rev-$name.rows
  mkdir -p $out/build $out/serve
  [ -f $out/build/build.json ] || TQ_PRUNE_ASSIGN_DIMS=$t OMP_NUM_THREADS=32 $PY $E/build.py $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024     $index $tier $out/build/build.json $seed > $out/build/run.log 2>&1
  [ -f $out/serve/run.csv ] || OMP_NUM_THREADS=1 taskset -c 2 $PY $E/serve.py $index $tier $Q/data/test_queries.fvecs     $Q/data/test_truth.ivecs 750 $NPROBES $out/serve/run.csv > $out/serve/run.log 2>&1
  rm -f $index $tier
  echo "done $name"
done
echo ALLDONE
