#!/usr/bin/env bash
# Issue #13 step 2: interleaved timing of the MAX_BITS caps, alone on CPUs 0-7 (job_rerank.sh TIMED=1), seed 0.
here=$(cd "$(dirname "$0")" && pwd)
ROUNDS=6
configs() { # lists budget_2 budget_4
  for ((r=0;r<ROUNDS;r++)); do
    echo "base $1 tieredq $3"; echo "mb6 $1 tieredq $3 MAX_BITS=6"; echo "mb5 $1 tieredq $3 MAX_BITS=5"
    echo "base $1 tieredq $2"; echo "mb5 $1 tieredq $2 MAX_BITS=5"
  done
}
cd ~/experiments/TieredQ-issue6-ivf-20261030
while pgrep -f "^bin/ivf_eval_" >/dev/null || pgrep -f "^bash job_rerank.sh" >/dev/null; do sleep 20; done
export OPENBLAS_NUM_THREADS=1 BIN=bin/ivf_eval_i13 TIMED=1 RERANK=10
CONFIGS="$(configs 64 3072 6144)" CODEC_LISTS=64 DATA=$HOME/experiments/TieredQ-issue20-ship-20261029/data/openai OUT=$here/time-openai100k NPROBES=16,64 bash job_rerank.sh
CONFIGS="$(configs 256 3072 6144)" CODEC_LISTS=256 DATA=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m OUT=$here/time-openai1m NPROBES=32,128 bash job_rerank.sh
echo done
