#!/bin/bash
set -euo pipefail
cd /home/sean/experiments/TieredQ-fast-aligned-20261026-0GBhDm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/benchmark:$PWD/benchmark/build-native
taskset -c 0-7 benchmark/.venv/bin/python run_calibrated.py calibrate > calibration-resume.log 2>&1
taskset -c 0-7 benchmark/.venv/bin/python run_calibrated.py matrix > matrix.log 2>&1
