#!/bin/bash
# Issue #9 greedy-vs-exact allocation gap on every dataset (4 CPUs; cheap, runs beside the A/B lanes).
# Pinned: build-gap must come from 8762da6 (-DISSUE9_GAP=ON); main no longer has fast_lookup_allocation.
set -u
W=/home/sean/experiments/TieredQ-issue9-empirical-20261029
cd $W
gap() {  # $1 dataset, $2 dimensions, $3.. budgets
  ds=$1; shift
  OMP_NUM_THREADS=4 taskset -c 32-35 build-gap/allocation_gap data/$ds "$@" > gap-$ds.tsv
}
gap glove 200 200 300 400 450 500 550 600 650 700 750 800 &&
  gap sift 128 128 192 256 288 320 352 384 416 448 480 512 &&
  gap gemini 768 768 1152 1536 1920 2304 2688 3072 &&
  gap openai 1536 1536 2304 3072 3840 4608 5376 6144
echo $? > gap-exit-code
