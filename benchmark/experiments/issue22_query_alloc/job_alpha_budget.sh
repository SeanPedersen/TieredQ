#!/usr/bin/env bash
# Budget-dependent allocation power: alpha 1 / 1.25 / 1.5 at 0.25-1 bits/dim (prefix code budgets), IVF candidate
# recall on OpenAI-1M and MS MARCO-1M real/doc, chosen on even query ids and reported on odd (QUERY_PARITY).
# Untimed, 3 lanes on CPUs 8-31 via the issue #6 workspace's job_rerank.sh. Waits for the drift job.
set -u
cd "$(dirname "$0")"
W=$PWD
E=$HOME/experiments/TieredQ-issue6-ivf-20261030
while pgrep -f "^bash job_drift.sh" >/dev/null; do sleep 30; done
declare -A DATA=([openai1m]=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
                 [msmarco-real]=$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-real
                 [msmarco-doc]=$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-doc)
declare -A DIM=([openai1m]=1536 [msmarco-real]=1024 [msmarco-doc]=1024)
date -u > alpha-started-utc.txt
for set in openai1m msmarco-real msmarco-doc; do
  d=${DIM[$set]}
  for parity in even odd; do
    list=$W/alpha-$set-$parity.list; : > $list
    for quarter in 1 2 3 4; do for power in 1 1.25 1.5; do
      echo "a$power 256 tieredq $((d*quarter/4)) ALLOCATION_POWER=$power QUERY_PARITY=$parity" >> $list
    done; done
    mkdir -p $W/alpha-$set-$parity && cp $list $W/alpha-$set-$parity/configs
    (cd $E && CONFIGS="$(cat $list)" BIN=$E/bin/ivf_eval_alpha DATA=${DATA[$set]} DIM=$d OUT=$W/alpha-$set-$parity \
      NPROBES=32,128 RERANK=20,100 CODEC_LISTS=256 OPENBLAS_NUM_THREADS=1 bash job_rerank.sh)
  done
done
date -u > alpha-finished-utc.txt
