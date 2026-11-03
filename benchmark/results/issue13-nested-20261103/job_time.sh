#!/bin/bash
# Issue #13 option 2: flat nested search (TieredQNestedIVFIndex, one list, nprobe 1, suffix-only RAM tier) against
# flat RaBitQ in the same eval, OpenAI-100K. 2 interleaved rounds, alone on CPUs 0-7, seed 0.
N=$HOME/experiments/TieredQ-issue13-nested-20261103; D=$HOME/experiments/TieredQ-issue20-ship-20261029/data/openai
cd $N; O=time.tsv; : > $O
run() { local label=$1; shift; env "$@" BENCH_SEED=0 LISTS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 build/issue6_ivf_eval $MODE $BUDGET $D 1536 256 1 1 \
  2>/dev/null | grep -P '^(tieredq-nested|rabitq)\t' | sed "s/^/$label\t/" >> $O; }
for r in 1 2; do
  MODE=rabitq BUDGET=4 run rabitq4 X=1
  MODE=rabitq BUDGET=2 run rabitq2 X=1
  MODE=tieredq-nested BUDGET=1536 run t3p1536 NESTED_BITS=3 NESTED_ROWS=suffix RERANK=20,30
  MODE=tieredq-nested BUDGET=1024 run t3p1024 NESTED_BITS=3 NESTED_ROWS=suffix RERANK=20,30
  MODE=tieredq-nested BUDGET=768 run t3p768 NESTED_BITS=3 NESTED_ROWS=suffix RERANK=30,50
  MODE=tieredq-nested BUDGET=512 run t3p512 NESTED_BITS=3 NESTED_ROWS=suffix RERANK=50
  MODE=tieredq-nested BUDGET=1536 run t2p1536 NESTED_BITS=2 NESTED_ROWS=suffix RERANK=20,30
done
echo done > time.status
