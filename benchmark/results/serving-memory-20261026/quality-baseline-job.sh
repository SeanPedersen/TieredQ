#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU
while ! test -f "$root/sequential-exit-code"; do sleep 5; done
test "$(cat "$root/sequential-exit-code")" = 0
cd "$root/before/benchmark"
PYTHONPATH="$PWD/build-native" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c 0-7 uv run --locked --python 3.12 python "$root/check_quality_baseline.py"
echo 0 > "$root/quality-baseline-exit-code"
