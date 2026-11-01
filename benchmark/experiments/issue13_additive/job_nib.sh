#!/usr/bin/env bash
# Issue #13: nibble default on a dataset; rows compare with recall2-NAME base/additive. Usage: job_nib.sh NAME DATA DIM LISTS CODEC_LISTS NPROBES BUDGETS...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=$1 data=$2 dim=$3 lists=$4 codec=$5 nprobes=$6; shift 6
configs=""
for budget; do for seed in 0 1 2; do configs+="nib$seed $lists tieredq $budget BENCH_SEED=0 TIEREDQ_SEED=$seed"$'\n'; done; done
cd ~/experiments/TieredQ-issue6-ivf-20261030
CONFIGS="$configs" BIN=${BIN:-bin/ivf_eval_nib} DATA=$data DIM=$dim OUT=$here/${PREFIX:-nib}-$name NPROBES=$nprobes RERANK=10,100 CODEC_LISTS=$codec \
  OPENBLAS_NUM_THREADS=1 bash job_rerank.sh
