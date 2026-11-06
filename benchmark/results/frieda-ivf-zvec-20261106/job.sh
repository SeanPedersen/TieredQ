#!/usr/bin/env bash
# Build-independent, sequential single-core sweep; run from the experiment directory.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="$(pwd)"
CPU=3
run() {
  taskset -c "$CPU" "$ROOT/env/bin/python" -u "$ROOT/run.py" \
    --harness "$ROOT/source/benchmark/experiments/issue6_zvec" \
    --data "$ROOT/data" --output "$ROOT/results" --workdir "$ROOT/work" "$@"
}
date -u
for bits in 1 2 3 4 5 6 7; do
  run --codec zvec --bits "$bits"
done
for bits in 1.16 2.16 3.16 4.16 5.16 6.16 7.16; do
  run --codec tieredq --bits "$bits"
done
for candidates in 20 50; do
  for bits in 3 4 5 6 7; do
    run --codec nested --bits "$bits" --prefix 1 --candidates "$candidates"
  done
done
for bits in 3 4 5 6 7; do
  run --codec nested --bits "$bits" --prefix 2 --candidates 20
done
date -u
echo __DONE__
