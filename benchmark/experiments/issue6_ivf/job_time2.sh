#!/usr/bin/env bash
# Shared-level-table refinement scorer: rebuild, test, re-time the 100k refinement cells alone on CPUs 0-7.
set -euo pipefail
cd "$(dirname "$0")"
cmake --build build-eval -j24 > build-eval-refinement2.log 2>&1
cp build-eval/issue6_ivf_eval bin/ivf_eval_refinement2
cmake --build build-tests -j24 --target test_tieredq > tests-refinement2.log 2>&1
(cd build-tests && ./test_tieredq) >> tests-refinement2.log 2>&1 && echo TESTS-OK >> tests-refinement2.log || echo TESTS-FAILED >> tests-refinement2.log
OPENBLAS_NUM_THREADS=1 CONFIGS="$(cat rescore-100k-time2.configs)" BIN=bin/ivf_eval_refinement2 TIMED=1 \
  DATA=$HOME/experiments/TieredQ-issue20-ship-20261029/data/openai OUT=$PWD/rescore-100k-time2 NPROBES=16 bash job_rerank.sh
date -u > time2.finished
