#!/bin/bash
# Issue #9 recall oracle: the exact padded allocator on variance^power, power 0.5-1.5 (1 is MSE-optimal), against
# the base (c85b0f8) rows of job_ab.sh. Recall only; five 7-CPU lanes in parallel.
# Pinned: source-new must be 8762da6. Main removed these switches when exact allocation on variance^1.5 became
# the fast default.
set -u
W=/home/sean/experiments/TieredQ-issue9-empirical-20261029
E=$W/source-new/benchmark/experiments/issue20_additive
cd $W
mkdir -p oracle
date -u > oracle/started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > oracle/host-processes.txt
cmake -S $E -B build-oracle -DTIEREDQ_SOURCE=$W/source-new -DCMAKE_BUILD_TYPE=Release > oracle/build.log 2>&1 &&
  taskset -c 0-15 cmake --build build-oracle -j16 >> oracle/build.log 2>&1 || { echo build > oracle/exit-code; exit 1; }
for ds in glove sift gemini openai; do grep -P '^base\t' ab-$ds.tsv > oracle/ab-$ds.tsv; done
run() {  # $1 power, $2 cpus
  power=$1 cpus=$2 arm=exact-p$1
  for ds in glove sift gemini openai; do
    case $ds in
      glove) d=200 budgets="200 300 400 450 500 550 600 650 700 750 800" ;;
      sift) d=128 budgets="128 192 256 288 320 352 384 416 448 480 512" ;;
      gemini) d=768 budgets="768 1152 1536 1920 2304 2688 3072" ;;
      openai) d=1536 budgets="1536 2304 3072 3840 4608 5376 6144" ;;
    esac
    DV_TIEREDQ_ALLOCATOR=exact DV_TIEREDQ_ALLOCATION_POWER=$power OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=1 \
      taskset -c $cpus build-oracle/ab_eval $arm data/$ds $d 0 $budgets >> oracle/ab-$ds.tsv 2>> oracle/errors-$arm.log ||
      { echo 1 > oracle/exit-$arm; return 1; }
  done
  echo 0 > oracle/exit-$arm
}
run 1 0-6 & run 0.5 7-13 & run 0.75 14-20 & run 1.25 21-27 & run 1.5 28-34 &
wait
date -u > oracle/finished-utc.txt
