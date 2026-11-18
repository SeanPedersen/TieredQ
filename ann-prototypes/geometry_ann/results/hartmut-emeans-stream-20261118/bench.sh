#!/bin/sh
# Issue #45, streamed e-means: one 8.84M x 1024 TieredQNestedIVFIndex build with the #52 defaults (64K lists, router,
# 8-bit suffix tier on the NVMe SSD) from the memory-mapped SQ8 store on 32 threads, then the 750 held-out queries on
# pinned CPU 2 with one OpenMP thread; the index and tier files are removed once measured. Resumable per step.
# Usage (under the benchmark lock): bench.sh NAME SEED [COARSE_TRAINING_JSON]   (empty JSON: automatic fast Lloyd)
set -e
E=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
NPROBES=96,128,192,256,512,1024
name=$1-seed$2; out=$E/results/$name; index=$E/ssd/$name.tq; tier=$E/ssd/$name.rows
PY="uv run --project $E/source --no-dev python"
mkdir -p $out/build $out/serve $E/ssd
[ -f $out/build/build.json ] || OMP_NUM_THREADS=32 $PY $E/build.py $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 \
  $index $tier $out/build/build.json $2 ${3:+"$3"} > $out/build/run.log 2>&1
[ -f $out/serve/run.csv ] || OMP_NUM_THREADS=1 taskset -c 2 $PY $E/serve.py $index $tier $Q/data/test_queries.fvecs \
  $Q/data/test_truth.ivecs 750 $NPROBES $out/serve/run.csv > $out/serve/run.log 2>&1
rm -f $index $tier
