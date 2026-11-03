#!/bin/bash
# Issue #13 nibble pruning port: arms interleaved, 2 rounds, alone on CPUs 0-7, seed 0. PRUNE_PREFIX=64 leaves no suffix (unpruned).
cd ~/experiments/TieredQ-issue6-ivf-20261030; O=~/experiments/TieredQ-issue13-additive-20261031/port.tsv; : > $O
run() { local set=$1 label=$2 data=$3 dim=$4 lists=$5 np=$6 b=$7; shift 7
  env "$@" BENCH_SEED=0 RERANK=10 LISTS=$lists OPENBLAS_NUM_THREADS=1 taskset -c 0-7 bin/ivf_eval_prune tieredq $b $data $dim $lists 1 ${np//,/ } 2>/dev/null | grep -P '^tieredq\t' | sed "s/^/$set\t$label\t/" >> $O; }
dataset() { local set=$1 data=$2 dim=$3 lists=$4 np=$5; for r in 1 2; do for b in $((2*dim)) $((4*dim)); do
  run $set nibble-z0 $data $dim $lists $np $b X=1
  run $set nibble-off $data $dim $lists $np $b PRUNE_PREFIX=64
  run $set nibble-z8 $data $dim $lists $np $b PRUNE_Z=8
  run $set exact-z0 $data $dim $lists $np $b TQ_EXACT=1
  run $set exact-z8 $data $dim $lists $np $b TQ_EXACT=1 PRUNE_Z=8
done; done; }
dataset openai100k ~/experiments/TieredQ-issue20-ship-20261029/data/openai 1536 64 16,64
dataset gemini ~/experiments/TieredQ-issue20-ship-20261029/data/gemini 768 64 16,64
dataset msmarco ~/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-real 1024 256 32,128
dataset openai1m ~/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m 1536 256 32,128
echo done > ~/experiments/TieredQ-issue13-additive-20261031/port.status
