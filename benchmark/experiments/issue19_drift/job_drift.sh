#!/usr/bin/env bash
# Issue #19.2: training sample size and shifted-distribution drift on OpenAI-1M and MS MARCO-1M (real, doc), IVF +
# rerank recall (untimed, 3 lanes on CPUs 8-31 via job_rerank.sh). Runs in ~/experiments/TieredQ-issue19-drift-20261031
# with the issue #6 workspace's job_rerank.sh and BIN.
#  1. shift_data.py per data set: 64 k-means clusters, 25% held out, queries split by where their truth lies.
#  2. sweep: codec (prefix 768 / 1024, INT8 and refinement:4 rerank) trained on TRAIN_ROWS 1k-100k random rows, the
#     partition from all rows; RaBitQ 1 + INT8 once per set (no codec training).
#  3. shift: per query group, the codec trained without the held-out clusters (ex) or on all rows (all), the partition
#     from all rows (full) or from the training rows (sub); RaBitQ with each partition.
set -u
cd "$(dirname "$0")"
W=$PWD
E=$HOME/experiments/TieredQ-issue6-ivf-20261030
BIN=$E/bin/ivf_eval_drift
PY="uv run --no-project --with numpy python"
declare -A DATA=([openai1m]=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
                 [msmarco-real]=$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-real
                 [msmarco-doc]=$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata/msmarco1m-doc)
declare -A DIM=([openai1m]=1536 [msmarco-real]=1024 [msmarco-doc]=1024)
SETS="openai1m msmarco-real msmarco-doc"
rerank() {  # set out configs-file data
  cp "$3" "$2.configs"
  (cd $E && CONFIGS="$(cat $3)" BIN=$BIN DATA=$4 DIM=${DIM[$1]} OUT=$2 NPROBES=32,128 RERANK=20,50,100 CODEC_LISTS=256 \
    OPENBLAS_NUM_THREADS=1 bash job_rerank.sh)
}
for set in $SETS; do
  [ -f shift-$set/manifest.json ] || taskset -c 8-31 $PY source/benchmark/experiments/issue19_drift/shift_data.py \
    ${DATA[$set]} ${DIM[$set]} $W/shift-$set >> shift-data.log 2>&1
done
date -u > sweep-started-utc.txt
for set in $SETS; do
  : > sweep-$set.list
  for rows in 1000 2000 4000 8000 16000 32768 100000; do for b in 768 1024; do
    echo "r$rows-p$b 256 tieredq $b PREFIX_BITS=$b TRAIN_ROWS=$rows RESCORE=int8,tieredq-refinement:4" >> sweep-$set.list
  done; done
  echo "rabitq1 256 rabitq 1 RESCORE=int8" >> sweep-$set.list
  rerank $set $W/sweep-$set sweep-$set.list ${DATA[$set]}
done
date -u > shift-started-utc.txt
for set in $SETS; do
  X=$W/shift-$set/excluded_rows.u32
  : > shift-$set.list
  for b in 768 1024; do
    common="tieredq $b PREFIX_BITS=$b RESCORE=int8,tieredq-refinement:4"
    echo "ex-full-p$b 256 $common TRAIN_EXCLUDE_ROWS=$X" >> shift-$set.list
    echo "ex-sub-p$b 256 $common TRAIN_EXCLUDE_ROWS=$X PARTITION_FROM=subset" >> shift-$set.list
    echo "all-full-p$b 256 $common" >> shift-$set.list
    echo "all-sub-p$b 256 $common PARTITION_FROM=subset" >> shift-$set.list
  done
  echo "rabitq1-all-full 256 rabitq 1 RESCORE=int8" >> shift-$set.list
  echo "rabitq1-all-sub 256 rabitq 1 RESCORE=int8 PARTITION_FROM=subset" >> shift-$set.list
  echo "rabitq1-ex-sub 256 rabitq 1 RESCORE=int8 TRAIN_EXCLUDE_ROWS=$X PARTITION_FROM=subset" >> shift-$set.list
  for group in heldout seen; do rerank $set $W/shift-$set-$group shift-$set.list $W/shift-$set/$group; done
done
date -u > finished-utc.txt
