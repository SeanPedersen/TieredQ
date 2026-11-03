#!/bin/bash
# Issue #13 step 3: approximate IVF pruning on the exact path against nibble scoring, interleaved, 2 rounds, alone on CPUs 0-7.
cd ~/experiments/TieredQ-issue6-ivf-20261030; O=~/experiments/TieredQ-issue13-additive-20261031/prune.tsv; : > $O
run() { # label data lists nprobes budget env...
  local label=$1 data=$2 lists=$3 np=$4 b=$5; shift 5
  env "$@" BENCH_SEED=0 RERANK=10 LISTS=$lists OPENBLAS_NUM_THREADS=1 taskset -c 0-7 bin/ivf_eval_backend tieredq $b $data 1536 $lists 1 ${np//,/ } 2>/dev/null | grep -P '^tieredq\t' | sed "s/^/$label\t/" >> $O; }
set_() { for r in 1 2; do for b in 3072 6144; do
  run nibble $1 $2 $3 $b X=1
  run exact-z0 $1 $2 $3 $b TQ_EXACT=1
  run exact-z8p4 $1 $2 $3 $b TQ_EXACT=1 PRUNE_Z=8 PRUNE_PREFIX=4
  run exact-z8p3 $1 $2 $3 $b TQ_EXACT=1 PRUNE_Z=8 PRUNE_PREFIX=3
done; done; }
set_ ~/experiments/TieredQ-issue20-ship-20261029/data/openai 64 16,64
set_ ~/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m 256 32,128
echo done > ~/experiments/TieredQ-issue13-additive-20261031/prune.status
