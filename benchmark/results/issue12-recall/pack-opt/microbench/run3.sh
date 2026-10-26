#!/bin/bash
# Resumes run.sh after the baseline profile step (run separately) failed on a copied CMake cache.
set -euo pipefail
W=/home/sean/experiments/TieredQ-issue12-recall; B=$W/pack-opt/baseline-ws; DATA=$W/benchmark/data
stamp() { echo "=== $(date -Is) $*"; }
cd $W/benchmark
stamp new single8; taskset -c 0-7 ./run.sh --bits 8 --threads 1 8 --batches 1 --seeds 0 --output output-pack-single
stamp new single4; taskset -c 0-7 ./run.sh --bits 4 --threads 1 8 --batches 1 --seeds 0 --output output-pack-single4
cd $B/benchmark
stamp baseline single4; taskset -c 0-7 ./run.sh --bits 4 --threads 1 8 --batches 1 --seeds 0 --data $DATA --output output-pack-baseline-single4
stamp baseline single8; taskset -c 0-7 ./run.sh --bits 8 --threads 1 8 --batches 1 --seeds 0 --data $DATA --output output-pack-baseline-single
cd $W/benchmark
stamp new curve; taskset -c 0-7 ./run.sh --bits 1 2 3 4 5 6 7 8 --threads 8 --batches 32 --seeds 0 1 2 --output output-pack-curve
stamp stages; taskset -c 0-7 uv run --locked --python 3.12 python profile.py output-pack --stages
stamp tiers; taskset -c 0-7 uv run --locked --python 3.12 python $W/pack-opt/tiers.py $W/pack-opt/tiers.json
stamp done
