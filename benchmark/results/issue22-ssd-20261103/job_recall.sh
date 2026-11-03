#!/bin/bash
# Issue #22 SSD comparison, phase A: recall only (untimed, seed 0) on OpenAI-1M, 256 lists (TieredQ's k-means, shared
# with RaBitQ). RaBitQ native IVF at 3-6 bits; TieredQ nested IVF (RAM tier: recall equals the SSD tier's). 3 lanes, CPUs 8-31.
S=$HOME/experiments/TieredQ-issue22-ssd-20261103; D=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
cd $S; O=recall.tsv; : > $O
run() { local cpus=$1 label=$2 mode=$3 budget=$4 probes=$5; shift 5
  env "$@" BENCH_SEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c $cpus build/issue6_ivf_eval $mode $budget $D 1536 256 0 ${probes//,/ } \
    2>/dev/null | grep -P '^(tieredq-nested|rabitq)\t' | sed "s/^/$label\t/" >> $O; }
( run 8-15 rq3 rabitq 3 16,32,64,128,256 X=1; run 8-15 rq4 rabitq 4 16,32,64,128,256 X=1 ) &
( run 16-23 rq5 rabitq 5 16,32,64,128,256 X=1; run 16-23 rq6 rabitq 6 16,32,64,128,256 X=1 ) &
( for p in 512 768 1024; do run 24-31 nq$p tieredq-nested $p 16,32,64,128 NESTED_BITS=4 RERANK=50,100,200; done ) &
wait; echo done > recall.status
