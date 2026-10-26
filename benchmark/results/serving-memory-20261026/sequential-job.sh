#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU
trap 'status=$?; echo "$status" > "$root/sequential-exit-code"; echo "Sequential job finished: $status"' EXIT
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
cd "$root/sequential/benchmark"
taskset -c 0-7 ./validate.sh 2>&1 | tee "$root/sequential-validation.log"
cmake -S .. -B ../build-tests -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_EXAMPLES=OFF
cmake --build ../build-tests -j 8
taskset -c 0-7 ctest --test-dir ../build-tests --output-on-failure
PYTHONPATH="$PWD/build-native" taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_memory.py" after
PYTHONPATH="$PWD/build-native" taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_memory.py" after --quality
PYTHONPATH="$PWD/build-native" taskset -c 0-7 uv run --locked --python 3.12 python "$root/run_rabitq_memory.py"
