#!/bin/bash
# Forced backends on OpenAI-100K, nibble vs exact, 2 rounds alone on CPUs 0-7 (scalar 1 round, it is slow).
cd ~/experiments/TieredQ-issue6-ivf-20261030; D=~/experiments/TieredQ-issue20-ship-20261029/data/openai; O=~/experiments/TieredQ-issue13-additive-20261031/backend2.tsv; : > $O
run() { local env=(); [ $2 = exact ] && env=(TQ_EXACT=1)
  env "${env[@]}" TQ_BACKEND=$1 BENCH_SEED=0 RERANK=10 LISTS=64 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 timeout 1500 bin/ivf_eval_backend tieredq $3 $D 1536 64 1 16 2>/dev/null | grep -P '^tieredq\t' | sed "s/^/$1\t$2\t/" >> $O; }
for r in 1 2; do for be in avx512 avx2; do for arm in nibble exact; do for b in 3072 6144; do run $be $arm $b; done; done; done; done
for arm in nibble exact; do for b in 3072 6144; do run scalar $arm $b; done; done
echo done > ~/experiments/TieredQ-issue13-additive-20261031/backend.status
