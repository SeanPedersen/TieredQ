#!/bin/bash
# Issue #9 held-out selection: recall with even/odd query halves for base (c85b0f8), the layout-preserving exact
# allocator on variance^power, and the parallel (ScaNN-style) weight. Waits for round 2's timing so it does not
# disturb it; eight 4-CPU lanes (LANES="arm:cpus ..." reruns selected arms). heldout.py picks a setting on even
# queries and reports odd ones.
set -u
W=/home/sean/experiments/TieredQ-issue9-empirical-20261029
X=$W/source-new/benchmark/experiments/issue20_additive
O=$W/heldout
until [ -f $W/round2/finished-utc.txt ] || [ -f $W/round2/exit-code ]; do sleep 30; done
cd $W
mkdir -p $O
date -u > $O/started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
for tree in base new; do
  cmake -S $X -B build-heldout-$tree -DTIEREDQ_SOURCE=$W/source-$tree -DCMAKE_BUILD_TYPE=Release > $O/build-$tree.log 2>&1 &&
    taskset -c 0-31 cmake --build build-heldout-$tree -j32 >> $O/build-$tree.log 2>&1 || { echo build > $O/exit-code; exit 1; }
done
arm_env() {
  case $1 in
    base) echo "" ;;
    layout-par*) echo "DV_TIEREDQ_ALLOCATOR=exact DV_TIEREDQ_ALLOCATION_WEIGHT=parallel DV_TIEREDQ_PARALLEL_WEIGHT=${1#layout-par}" ;;
    layout-p*) echo "DV_TIEREDQ_ALLOCATOR=exact DV_TIEREDQ_ALLOCATION_POWER=${1#layout-p}" ;;
  esac
}
recall() {  # $1 arm, $2 cpus
  tree=new; [ $1 = base ] && tree=base
  for ds in glove sift gemini openai; do
    case $ds in
      glove) d=200 b="200 300 400 450 500 550 600 650 700 750 800" ;;
      sift) d=128 b="128 192 256 288 320 352 384 416 448 480 512" ;;
      gemini) d=768 b="768 1152 1536 1920 2304 2688 3072" ;;
      openai) d=1536 b="1536 2304 3072 3840 4608 5376 6144" ;;
    esac
    env $(arm_env $1) OPENBLAS_NUM_THREADS=1 taskset -c $2 build-heldout-$tree/ab_eval $1 data/$ds $d 0 $b >> $O/ab-$ds.tsv 2>> $O/errors-$1.log ||
      { echo 1 > $O/exit-$1; return 1; }
  done
  echo 0 > $O/exit-$1
}
LANES=${LANES:-"base:0-3 layout-p1:4-7 layout-p1.25:8-11 layout-p1.5:12-15 layout-p1.75:16-19 layout-p2:20-23 layout-par0.1:24-27 layout-par1:28-31"}
for lane in $LANES; do recall ${lane%%:*} ${lane#*:} & done
wait
echo 0 > $O/exit-code
date -u > $O/finished-utc.txt
