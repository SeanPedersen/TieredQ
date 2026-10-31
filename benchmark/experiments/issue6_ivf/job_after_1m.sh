#!/usr/bin/env bash
# After the 1M timing: build the refinement library + eval, run the tests (AVX-512), then time the 100k rescore cells.
set -euo pipefail
cd "$(dirname "$0")"
while pgrep -f "^bash job_time_1m.sh" >/dev/null || pgrep -f "^bin/ivf_eval_" >/dev/null; do sleep 20; done
date -u > after-1m.started
cmake --build build-eval -j24 > build-eval-refinement.log 2>&1
cp build-eval/issue6_ivf_eval bin/ivf_eval_refinement
cmake -S source -B build-tests -DCMAKE_BUILD_TYPE=Release > tests-refinement.log 2>&1
cmake --build build-tests -j24 --target test_tieredq >> tests-refinement.log 2>&1
(cd build-tests && ./test_tieredq) >> tests-refinement.log 2>&1 && echo TESTS-OK >> tests-refinement.log || echo TESTS-FAILED >> tests-refinement.log
OPENBLAS_NUM_THREADS=1 CONFIGS="$(cat rescore-100k-time.configs)" BIN=bin/ivf_eval_refinement TIMED=1 \
  DATA=$HOME/experiments/TieredQ-issue20-ship-20261029/data/openai OUT=$PWD/rescore-100k-time NPROBES=16 bash job_rerank.sh
date -u > after-1m.finished
