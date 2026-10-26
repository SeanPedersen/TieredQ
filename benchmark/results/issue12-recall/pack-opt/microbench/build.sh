#!/bin/bash
# Builds baseline/new static libraries and links the synthetic A/B benchmark against each.
set -euo pipefail
W=/home/sean/experiments/TieredQ-issue12-recall; P=$W/pack-opt
for v in baseline new; do
  src=$W; [ $v = baseline ] && src=$P/baseline-ws
  cmake -S $src -B $P/build-$v -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_TESTS=OFF -DTIEREDQ_EXAMPLES=OFF >/dev/null
  cmake --build $P/build-$v -j16 --target dv_tieredq >/dev/null
  c++ -O3 -DNDEBUG -fopenmp -I$src/include $P/ebench.cpp $P/build-$v/libdv_tieredq.a -llapack -lblas -o $P/ebench-$v
done
