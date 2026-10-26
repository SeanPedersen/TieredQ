#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze
trap 'code=$?; echo "$code" > "$root/exit-code"; echo "Job completed: exit $code"' EXIT
cd "$root/benchmark"
echo "Validation started $(date -u +%FT%TZ)"
taskset -c 0-7 ./validate.sh 2>&1 | tee "$root/validation.log"
echo "Dataset verification started $(date -u +%FT%TZ)"
taskset -c 0-7 uv run --locked --python 3.12 python "$root/verify-data.py" 2>&1 | tee "$root/data-verification.log"
echo "Core matrix started $(date -u +%FT%TZ)"
taskset -c 0-7 ./run.sh --bits 1 2 4 8 --seeds 0 --threads 1 8 --batches 1 32 --output output-current-20261026-tpj3ze 2>&1 | tee "$root/matrix.log"
echo "Core matrix and figures completed $(date -u +%FT%TZ)"
