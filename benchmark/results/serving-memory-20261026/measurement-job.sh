#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU
trap 'status=$?; echo "$status" > "$root/measurement-exit-code"; echo "Measurement job finished: $status"' EXIT
while ! test -f "$root/build-exit-code"; do sleep 5; done
test "$(cat "$root/build-exit-code")" = 0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
for variant in before after; do
  echo "Measuring $variant"
  cd "$root/$variant/benchmark"
  PYTHONPATH="$PWD/build-native" taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_memory.py" "$variant" 2>&1 | tee "$root/$variant-measurement.log"
done
cd "$root/after/benchmark"
PYTHONPATH="$PWD/build-native" taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_memory.py" after --quality 2>&1 | tee "$root/quality-measurement.log"
