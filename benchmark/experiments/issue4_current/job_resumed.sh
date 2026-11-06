#!/usr/bin/env bash
set -euo pipefail
W=${WORKSPACE:-/home/sean/experiments/TieredQ-issue4-current-20261105}
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
B=$W/source/benchmark
export WORKSPACE="$W" PYTHONPATH="$B/build-native:$B" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
trap 'status=$?; echo "$status" > "$W/pipeline-exit-code"; date -u +%FT%TZ > "$W/pipeline-finished-utc.txt"' EXIT
test "$(cat "$W/adapter-validation-exit-code")" = 0
bash "$B/experiments/issue4_current/job_glove1m.sh"
taskset -c 8-17 "$P" -u "$B/experiments/issue4_current/timing.py" "$W" > "$W/timing-progress.log" 2>&1
taskset -c 8-17 "$P" -u "$B/experiments/issue4_current/report.py" "$W" > "$W/report.log" 2>&1
taskset -c 8-17 "$P" -u "$B/experiments/issue4_current/audit.py" "$W" > "$W/review-audit.log" 2>&1
