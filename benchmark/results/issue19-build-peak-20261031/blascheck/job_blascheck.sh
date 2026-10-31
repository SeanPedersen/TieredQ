#!/usr/bin/env bash
# Recall of default models with OpenBLAS (chunked syrk covariance, OpenBLAS syevd) vs the reference-LAPACK sweep
# cells (issue19-drift sweep, TRAIN_ROWS 32768). Waits for the alpha job.
set -u
cd "$(dirname "$0")"; W=$PWD; E=$HOME/experiments/TieredQ-issue6-ivf-20261030
while pgrep -f "^bash job_alpha_budget.sh" >/dev/null; do sleep 30; done
for set in openai1m:1536:$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m msmarco-real:1024:$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-real; do
  IFS=: read name dim data <<< "$set"
  : > $W/$name.list
  for b in 768 1024; do echo "fast-p$b 256 tieredq $b PREFIX_BITS=$b RESCORE=int8,tieredq-refinement:4" >> $W/$name.list; done
  mkdir -p $W/$name && cp $W/$name.list $W/$name/configs
  (cd $E && CONFIGS="$(cat $W/$name.list)" BIN=$E/bin/ivf_eval_fastblas DATA=$data DIM=$dim OUT=$W/$name NPROBES=32,128 \
    RERANK=20,50,100 CODEC_LISTS=256 OPENBLAS_NUM_THREADS=1 bash job_rerank.sh)
done
date -u > finished-utc.txt
