#!/bin/bash
set -euo pipefail
cd /home/sean/experiments/TieredQ-fast-aligned-20261026-0GBhDm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmark:$PWD/benchmark/build-native
cmake -S . -B build-validation -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -ffp-contract=off" -DTIEREDQ_EXAMPLES=OFF > cpp-validation.log 2>&1
cmake --build build-validation -j8 >> cpp-validation.log 2>&1
taskset -c 0-7 ctest --test-dir build-validation --output-on-failure >> cpp-validation.log 2>&1
taskset -c 0-7 benchmark/validate.sh > validation-final.log 2>&1
taskset -c 0-7 benchmark/.venv/bin/python run_calibrated.py calibrate > calibration.log 2>&1
ps -eo pid,etime,pcpu,args --sort=-pcpu | head -15 > host-before-matrix.txt
taskset -c 0-7 benchmark/.venv/bin/python run_calibrated.py matrix > matrix.log 2>&1
