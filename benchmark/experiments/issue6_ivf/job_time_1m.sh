#!/usr/bin/env bash
# Issue #19.1 / #22: 1M rerank timing cells, alone on CPUs 0-7 once every untimed lane has finished.
set -euo pipefail
cd "$(dirname "$0")"
while pgrep -f "^bin/ivf_eval_" >/dev/null || pgrep -f "^bash job_rerank.sh" >/dev/null; do sleep 20; done
date -u > rerank-1m-time.started
OPENBLAS_NUM_THREADS=1 CONFIGS="$(cat rerank-1m-time.configs)" CODEC_LISTS=256 BIN=bin/ivf_eval_rerank1m TIMED=1 \
  DATA=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m OUT=$PWD/rerank-1m-time NPROBES=1 bash job_rerank.sh
date -u > rerank-1m-time.finished
