#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-openai-aligned-20261026-akXzMJ
trap 'code=$?; echo "$code" > "$root/exit-code"; echo "Job completed: exit $code"' EXIT
cd "$root/benchmark"
taskset -c 0-7 ./validate.sh 2>&1 | tee "$root/validation.log"
PYTHONPATH="$root/benchmark/build-native" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c 0-7 uv run --locked --python 3.12 python ../run_aligned.py 2>&1 | tee "$root/run.log"
