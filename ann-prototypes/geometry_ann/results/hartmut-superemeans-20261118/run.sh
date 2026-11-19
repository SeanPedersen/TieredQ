#!/bin/sh
# Issue #45: SuperEMeans / e-means (SuperKMeans library) vs TieredQ's C++ e-means in the SuperEMeans reproducibility
# harness, train-only, 32 threads, same rows/k/epochs. Resumable: skips logs that hold a RESULT line.
# Usage (takes the benchmark lock): run.sh DATASET K [EPOCHS]
S=/home/sean/experiments/TieredQ-issue45-superemeans-20261118
dataset=$1; k=$2; epochs=${3:-2}
B=$S/repro/build; L=$S/logs; mkdir -p $L
export THREADS=32 OMP_NUM_THREADS=32
run() {
  log=$L/$1-$dataset-k$k-e$epochs.log; shift
  grep -q '^RESULT' $log 2>/dev/null && { echo "[skip] $log"; return; }
  echo "[run]  $log"; "$@" > $log 2>&1 || echo "FAILED exit $? $log"
}
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
run superemeans   $B/emeans    $S/data $dataset $k true $epochs
run tq_exact      $B/tq_emeans $S/data $dataset $k 0 0 $epochs
run tq_s256p4     $B/tq_emeans $S/data $dataset $k 256 4 $epochs
run emeans        $B/emeans    $S/data $dataset $k false $epochs
