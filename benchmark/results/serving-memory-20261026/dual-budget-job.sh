#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU
trap 'status=$?; echo "$status" > "$root/dual-budget-exit-code"; echo "Dual-budget job finished: $status"' EXIT
while ! test -f "$root/quality-baseline-exit-code"; do sleep 5; done
cd "$root/sequential/benchmark"
PYTHONPATH="$PWD/build-native" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_dual_budget.py"
