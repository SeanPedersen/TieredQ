#!/bin/bash
# Router re-rank depth 256/128/64: recall and p50 of full queries (router + 2-bit scan + SQ8 top-40 re-rank).
cd "$(dirname "$0")"; E=/home/sean/experiments; Q=$E/TieredQ-issue44-msmarco8m-20261107
exec 9>>$E/.tieredq-issue44-benchmark.lock; flock 9
cmake --build build -j 16 --target profile_router > build4.log 2>&1 || { echo build-failed > depths.status; exit 1; }
build/profile_router index=$E/TieredQ-issue44-router-20261110/indexes/scratch-64k.tq queries=$Q/data/test_queries.fvecs count=750 cpu=2 \
  nprobes=32,64,96,128,192,256 rerank_depths=1 rows=8840823 codes=$Q/base.i8 scales=$Q/scales.f32 offsets=$Q/offsets.f32 \
  truth=$Q/data/test_truth.ivecs > depths.csv 2> depths.log
echo $? > depths.status
