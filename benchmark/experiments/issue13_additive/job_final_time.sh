#!/usr/bin/env bash
# Issue #13: exact (i13, without hooks) vs nibble default, interleaved, 6 rounds alone on CPUs 0-7, seed 0.
here=$(cd "$(dirname "$0")" && pwd); cd ~/experiments/TieredQ-issue6-ivf-20261030
run() { # name data dim lists nprobes budgets...
  local name=$1 data=$2 dim=$3 lists=$4 np=$5; shift 5
  for r in 1 2 3 4 5 6; do for bin in i13 final; do for b; do
    BENCH_SEED=0 RERANK=10 LISTS=$lists OPENBLAS_NUM_THREADS=1 taskset -c 0-7 bin/ivf_eval_$bin tieredq $b $data $dim $lists 1 ${np//,/ } 2>/dev/null | grep -P '^tieredq\t' | sed "s/^/$bin\t/" >> $here/ft-$name.tsv
  done; done; done
}
run openai100k ~/experiments/TieredQ-issue20-ship-20261029/data/openai 1536 64 16,64 3072 6144
run openai1m ~/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m 1536 256 32,128 3072 6144
echo time-done
