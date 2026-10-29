#!/bin/bash
# Issue #20 round 2: shared nibble groups + no trimming (source-shared) vs 177db5f (source-base) on CPUs 0-7, at the
# budgets where round 1 models differed. AVX-512 tests first; one process per (dataset, budget, tree), order alternating.
set -u
W=/home/sean/experiments/TieredQ-issue20-ship-20261029
E=$W/source-shared/benchmark/experiments/issue20_additive
cd $W
date -u > shared-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > shared-host-processes.txt
cmake -S $E -B build-shared -DTIEREDQ_SOURCE=$W/source-shared -DCMAKE_BUILD_TYPE=Release > build-shared.log 2>&1 &&
  taskset -c 0-7 cmake --build build-shared -j8 >> build-shared.log 2>&1 || { echo build > shared-exit-code; exit 1; }
cmake -S source-shared -B tests-shared -DCMAKE_BUILD_TYPE=Release > tests-shared.log 2>&1 && taskset -c 0-7 cmake --build tests-shared -j8 >> tests-shared.log 2>&1 &&
  OMP_NUM_THREADS=2 taskset -c 0-7 ctest --test-dir tests-shared --output-on-failure >> tests-shared.log 2>&1
echo $? > tests-shared-exit-code
run() {  # $1 dataset, $2 dimensions, $3 timed, $4.. budgets
  ds=$1 d=$2 timed=$3; shift 3
  i=0
  for budget in "$@"; do
    trees="base shared"; [ $((i % 2)) -eq 1 ] && trees="shared base"; i=$((i+1))
    for tree in $trees; do
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-7 build-$tree/ab_eval $tree data/$ds $d $timed $budget >> shared-$ds.tsv 2>> shared-errors.log || return 1
    done
  done
}
run glove 200 1 400 450 500 550 600 650 700 750 766 780 794 &&
  run sift 128 1 256 288 320 352 384 &&
  run gemini 768 0 768 &&
  run openai 1536 0 1536
echo $? > shared-exit-code
date -u > shared-finished-utc.txt
