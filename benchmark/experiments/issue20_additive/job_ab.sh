#!/bin/bash
# Issue #20 shipped vs 177db5f A/B on CPUs 0-7: build both trees, run the shipped C++ tests (AVX-512), export data,
# then interleave one process per (dataset, budget, tree) with the tree order alternating per budget.
set -u
W=/home/sean/experiments/TieredQ-issue20-ship-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
D=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
E=$W/source-ship/benchmark/experiments/issue20_additive
cd $W
date -u > ab-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > ab-host-processes.txt
for tree in base ship; do
  cmake -S $E -B build-$tree -DTIEREDQ_SOURCE=$W/source-$tree -DCMAKE_BUILD_TYPE=Release > build-$tree.log 2>&1 &&
    taskset -c 0-7 cmake --build build-$tree -j8 >> build-$tree.log 2>&1 || { echo build-$tree > ab-exit-code; exit 1; }
done
cmake -S source-ship -B tests-ship -DCMAKE_BUILD_TYPE=Release > tests-ship.log 2>&1 && taskset -c 0-7 cmake --build tests-ship -j8 >> tests-ship.log 2>&1 &&
  OMP_NUM_THREADS=2 taskset -c 0-7 ctest --test-dir tests-ship --output-on-failure >> tests-ship.log 2>&1
echo $? > tests-ship-exit-code
[ -f data/gemini/truth.u32 ] || taskset -c 0-7 $P $E/export_data.py $D data > export.log 2>&1 || { echo export > ab-exit-code; exit 1; }
run() {  # $1 dataset, $2 dimensions, $3 timed, $4.. budgets
  ds=$1 d=$2 timed=$3; shift 3
  i=0
  for budget in "$@"; do
    trees="base ship"; [ $((i % 2)) -eq 1 ] && trees="ship base"; i=$((i+1))
    for tree in $trees; do
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-7 build-$tree/ab_eval $tree data/$ds $d $timed $budget >> ab-$ds.tsv 2>> ab-errors.log || return 1
    done
  done
}
run glove 200 1 400 450 500 550 600 650 700 750 766 780 794 800 &&
  run sift 128 1 256 288 320 352 384 448 480 512 &&
  run gemini 768 0 768 1536 2304 3072 &&
  run openai 1536 0 1536 3072 4608 6144
echo $? > ab-exit-code
date -u > ab-finished-utc.txt
