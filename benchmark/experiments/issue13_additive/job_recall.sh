#!/usr/bin/env bash
# Issue #13 recall arms at equal code bits via the #6 job_rerank.sh runner. Usage: job_recall.sh NAME DATA DIM LISTS CODEC_LISTS NPROBES BUDGETS...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=$1 data=$2 dim=$3 lists=$4 codec=$5 nprobes=$6; shift 6
configs=""
for budget; do for seed in 0 1 2; do
  configs+="base$seed $lists tieredq $budget BENCH_SEED=0 TIEREDQ_SEED=$seed"$'\n'
  configs+="additive$seed $lists tieredq $budget BENCH_SEED=0 TIEREDQ_SEED=$seed LEVELS_FILE=$here/best.levels"$'\n'
  configs+="mb5s$seed $lists tieredq $budget BENCH_SEED=0 TIEREDQ_SEED=$seed MAX_BITS=5"$'\n'
  configs+="mb6s$seed $lists tieredq $budget BENCH_SEED=0 TIEREDQ_SEED=$seed MAX_BITS=6"$'\n'
done; done
cd ~/experiments/TieredQ-issue6-ivf-20261030
CONFIGS="$configs" BIN=bin/ivf_eval_i13 DATA=$data DIM=$dim OUT=$here/recall2-$name NPROBES=$nprobes RERANK=10,100 CODEC_LISTS=$codec \
  OPENBLAS_NUM_THREADS=1 bash job_rerank.sh
