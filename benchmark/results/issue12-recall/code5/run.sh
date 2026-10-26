#!/bin/bash
# A/B: packed code lookup for models whose widest tier has 5-6 bits (new) vs x86 int8 levels (baseline-ws).
set -euo pipefail
W=/home/sean/experiments/TieredQ-issue12-recall; B=$W/code5/baseline-ws; DATA=$W/benchmark/data
stamp() { echo "=== $(date -Is) $*"; }
points() { # $1 workspace benchmark dir, $2 output prefix, rest: run.sh options
  local dir=$1 prefix=$2; shift 2
  cd $dir
  taskset -c 0-7 ./run.sh --datasets sift --bits 2 --data $DATA --output $prefix-sift "$@"
  taskset -c 0-7 ./run.sh --datasets glove --bits 5 6 --data $DATA --output $prefix-glove "$@"
  taskset -c 0-7 ./run.sh --datasets openai --bits 2 3 --data $DATA --output $prefix-openai "$@"
}
stamp new validate; cd $W/benchmark && ./validate.sh
stamp baseline validate; cd $B/benchmark && ./validate.sh
stamp new b32; points $W/benchmark output-code5 --threads 8 --batches 32 --seeds 0 1 2
stamp baseline b32; points $B/benchmark output-code5-baseline --threads 8 --batches 32 --seeds 0 1 2
stamp new b1; points $W/benchmark output-code5-single --threads 1 8 --batches 1 --seeds 0
stamp baseline b1; points $B/benchmark output-code5-baseline-single --threads 1 8 --batches 1 --seeds 0
stamp done
