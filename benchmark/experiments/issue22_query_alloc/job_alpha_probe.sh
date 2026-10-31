#!/usr/bin/env bash
# Issue #22 task 3.1 check: allocation power on MS MARCO-1M real and doc queries, prefixes 512 / 768, fp32 rerank.
# Runs in ~/experiments/TieredQ-issue6-ivf-20261030 (job_rerank.sh, bin/ivf_eval_power = ivf_eval with ALLOCATION_POWER).
set -u
cd "$(dirname "$0")"
D=$HOME/experiments/TieredQ-issue22-msmarco-20261031/abdata
for set in real doc; do
  mkdir -p alpha-probe-$set && cp alpha-probe.configs alpha-probe-$set/configs
  CONFIGS="$(cat alpha-probe.configs)" BIN=bin/ivf_eval_power DATA=$D/msmarco1m-$set DIM=1024 OUT=$PWD/alpha-probe-$set \
    NPROBES=32,128 RERANK=20,100 CODEC_LISTS=256 bash job_rerank.sh
done
echo done > alpha-probe.finished
