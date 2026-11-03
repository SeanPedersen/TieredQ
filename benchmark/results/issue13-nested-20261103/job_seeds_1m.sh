#!/bin/bash
# Issue #13 option 2: OpenAI-1M recall over seeds 1-2 for RaBitQ 4-bit and t3p512 k'=100 (flat, one list), CPUs 8-15.
N=$HOME/experiments/TieredQ-issue13-nested-20261103; D=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
cd $N; O=seeds-1m.tsv; : > $O
for s in 1 2; do
  env BENCH_SEED=$s TIEREDQ_SEED=$s LISTS=1 RERANK=50,100 NESTED_BITS=3 NESTED_ROWS=suffix OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    taskset -c 8-15 build/issue6_ivf_eval tieredq-nested 512 $D 1536 256 0 1 2>/dev/null | grep -P '^tieredq-nested\t' | sed "s/^/s$s\t/" >> $O
  env BENCH_SEED=$s TIEREDQ_SEED=$s LISTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
    taskset -c 8-15 build/issue6_ivf_eval rabitq 4 $D 1536 256 0 1 2>/dev/null | grep -P '^rabitq\t' | sed "s/^/s$s\t/" >> $O
done
echo done > seeds-1m.status
