#!/bin/bash
# Issue #9 empirical-distortion A/B: c85b0f8 (base) vs the new tree under DV_TIEREDQ_DISTORTION=gaussian|raw|shape.
# Recall is deterministic, so arms run untimed in parallel, one 8-CPU lane each; reuses the issue #20 ab_eval
# and exported data. The gaussian arm must reproduce base model_ids and runs on GloVe/SIFT only.
# Pinned: source-new must be 8762da6. Main removed these switches when exact allocation on variance^1.5 became
# the fast default.
set -u
W=/home/sean/experiments/TieredQ-issue9-empirical-20261029
E=$W/source-new/benchmark/experiments/issue20_additive
cd $W
date -u > started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > host-processes.txt
for tree in base new; do
  cmake -S $E -B build-$tree -DTIEREDQ_SOURCE=$W/source-$tree -DCMAKE_BUILD_TYPE=Release > build-$tree.log 2>&1 &&
    taskset -c 0-15 cmake --build build-$tree -j16 >> build-$tree.log 2>&1 || { echo build-$tree > exit-code; exit 1; }
done
run() {  # $1 arm, $2 cpus, $3.. datasets
  arm=$1 cpus=$2; shift 2
  tree=new mode=$arm; [ $arm = base ] && { tree=base; mode=gaussian; }
  for ds in "$@"; do
    case $ds in
      glove) d=200 budgets="200 300 400 450 500 550 600 650 700 750 800" ;;
      sift) d=128 budgets="128 192 256 288 320 352 384 416 448 480 512" ;;
      gemini) d=768 budgets="768 1152 1536 1920 2304 2688 3072" ;;
      openai) d=1536 budgets="1536 2304 3072 3840 4608 5376 6144" ;;
    esac
    DV_TIEREDQ_DISTORTION=$mode OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c $cpus \
      build-$tree/ab_eval $arm data/$ds $d 0 $budgets >> ab-$ds.tsv 2>> errors-$arm.log || { echo $arm > exit-$arm; return 1; }
  done
  echo 0 > exit-$arm
}
run base 0-7 glove sift gemini openai &
run raw 8-15 glove sift gemini openai &
run shape 16-23 glove sift gemini openai &
run gaussian 24-31 glove sift &
wait
date -u > finished-utc.txt
