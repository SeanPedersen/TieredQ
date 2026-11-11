#!/bin/bash
# Router fixed-cost profile (#47 follow-up). Times probe_lists / routed / exact search at a few nprobes on the 64K
# from-scratch index, then perf-records 60 s of routed search at nprobe 128. Waits for the shared benchmark lock.
set -e
cd "$(dirname "$0")"
E=/home/sean/experiments; Q=$E/TieredQ-issue44-msmarco8m-20261107
exec 9>>$E/.tieredq-issue44-benchmark.lock; flock 9
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=-g -fno-omit-frame-pointer" -DTIEREDQ_SOURCE_DIR=$PWD/source/tieredq > configure.log 2>&1
cmake --build build -j 16 --target profile_router > build.log 2>&1
args="index=$E/TieredQ-issue44-router-20261110/indexes/scratch-64k.tq queries=$Q/data/test_queries.fvecs count=750 cpu=2"
build/profile_router $args nprobes=32,64,128,256 > timings.csv 2> timings.log
perf record --call-graph dwarf,16384 -F 499 -o perf.data build/profile_router $args nprobes=128 loop=60 > /dev/null 2> perf-run.log
perf report -i perf.data --no-children --percent-limit 0.5 --stdio 2>/dev/null | grep -v '^$' | head -120 > perf-flat.txt
perf report -i perf.data --children --percent-limit 2 --stdio -g none 2>/dev/null | grep -v '^$' | head -80 > perf-children.txt
echo complete > status
