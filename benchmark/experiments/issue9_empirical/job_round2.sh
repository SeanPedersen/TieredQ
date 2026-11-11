#!/bin/bash
# Issue #9 round 2. Recall: layout-preserving exact allocator (nibble path and padded limit from the unweighted
# model) on variance^power and on query-energy weights, five 7-CPU lanes. Then build and search cost on quiet
# CPUs 0-7: one process per (round, dataset, arm), arm order rotating per round, against c85b0f8 (base).
# Pinned: source-new must be 8762da6. Main removed these switches when exact allocation on variance^1.5 became
# the fast default.
set -u
W=/home/sean/experiments/TieredQ-issue9-empirical-20261029
X=$W/source-new/benchmark/experiments
O=$W/round2
cd $W
mkdir -p $O
date -u > $O/started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
build() {  # $1 name, $2 source dir, $3 extra cmake args
  cmake -S $X/$4 -B build-$1 -DTIEREDQ_SOURCE=$W/$2 -DCMAKE_BUILD_TYPE=Release $3 > $O/build-$1.log 2>&1 &&
    taskset -c 0-15 cmake --build build-$1 -j16 >> $O/build-$1.log 2>&1
}
build r2 source-new "" issue20_additive && build r2-time source-new "" issue9_empirical &&
  build base-time source-base -DISSUE9_GAP=OFF issue9_empirical || { echo build > $O/exit-code; exit 1; }
budgets() {  # $1 dataset -> "d budgets"
  case $1 in
    glove) echo "200 200 300 400 450 500 550 600 650 700 750 800" ;;
    sift) echo "128 128 192 256 288 320 352 384 416 448 480 512" ;;
    gemini) echo "768 768 1152 1536 1920 2304 2688 3072" ;;
    openai) echo "1536 1536 2304 3072 3840 4608 5376 6144" ;;
  esac
}
arm_env() {  # $1 arm -> environment for that arm
  case $1 in
    base) echo "" ;;
    layout-query) echo "TQ_ALLOCATOR=exact TQ_ALLOCATION_WEIGHT=query" ;;
    layout-p*) echo "TQ_ALLOCATOR=exact TQ_ALLOCATION_POWER=${1#layout-p}" ;;
  esac
}
for ds in glove sift gemini openai; do grep -P '^base\t' ab-$ds.tsv > $O/ab-$ds.tsv; done
recall() {  # $1 arm, $2 cpus
  for ds in glove sift gemini openai; do
    read d b <<< "$(budgets $ds)"
    env $(arm_env $1) OPENBLAS_NUM_THREADS=1 taskset -c $2 build-r2/ab_eval $1 data/$ds $d 0 $b >> $O/ab-$ds.tsv 2>> $O/errors-$1.log ||
      { echo 1 > $O/exit-$1; return 1; }
  done
  echo 0 > $O/exit-$1
}
recall layout-p1 0-6 & recall layout-p1.5 7-13 & recall layout-p2 14-20 & recall layout-p2.5 21-27 & recall layout-query 28-34 &
wait
date -u > $O/recall-finished-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/timing-host-processes.txt
arms=(base layout-p1 layout-p1.5 layout-p2 layout-query)
timing_budgets() {
  case $1 in glove) echo "200 200 400 600 800" ;; sift) echo "128 128 256 384 512" ;;
    gemini) echo "768 768 1536 2304 3072" ;; openai) echo "1536 1536 3072 4608 6144" ;; esac
}
for round in 0 1 2; do
  for ds in glove sift gemini openai; do
    read d b <<< "$(timing_budgets $ds)"
    for i in 0 1 2 3 4; do
      arm=${arms[$(( (i + round) % 5 ))]} binary=build-r2-time/issue9_build_time
      [ $arm = base ] && binary=build-base-time/issue9_build_time
      env $(arm_env $arm) OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 $binary $arm data/$ds $d 1 $b \
        >> $O/time-$ds.tsv 2>> $O/errors-time.log || { echo time > $O/exit-code; exit 1; }
    done
  done
done
echo 0 > $O/exit-code
date -u > $O/finished-utc.txt
