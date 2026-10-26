#!/bin/bash
set -euo pipefail
root=/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU
trap 'status=$?; echo "$status" > "$root/build-exit-code"; echo "Build job finished: $status"' EXIT
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
for variant in before after; do
  echo "Validating $variant"
  cd "$root/$variant/benchmark"
  taskset -c 0-7 ./validate.sh 2>&1 | tee "$root/$variant-validation.log"
  cmake -S .. -B ../build-tests -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_EXAMPLES=OFF
  cmake --build ../build-tests -j 8
  taskset -c 0-7 ctest --test-dir ../build-tests --output-on-failure
 done
