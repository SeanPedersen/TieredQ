#!/bin/bash
# Issue #13: TieredQNestedIndex through run.py (codec tieredq-nested), OpenAI-100K and 1M at RaBitQ's 4-bit recall
# configs; 2 interleaved rounds alone on CPUs 0-7, seed 0. Compare with the run.py RaBitQ reference cells.
set -u
R=$HOME/experiments/TieredQ-issue13-nested-20261103/runpy; B=$R/source/benchmark
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
F=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data; O=$HOME/experiments/TieredQ-issue4-1m-20261030/data
(cd $B && taskset -c 16-31 $P -c "from native_build import build_native; build_native()") > $R/build.log 2>&1 || { echo build-failed > $R/status; exit 1; }
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
cell() { local ds=$1 data=$2 k=$3 r=$4; local out=$R/timing/r$r; mkdir -p $out
  (cd $B && taskset -c 0-7 $P -u run.py --codec tieredq-nested --datasets $ds --bits 3 --code-bits 4608 --prefix-bits 512 \
    --candidates $k --seeds 0 --data $data --output $out --cell $out/cell-$ds-nested-p512-k$k.json >> $R/run.log 2>&1) \
    || echo "$ds r$r" >> $R/failed.txt; }
for r in 1 2; do cell openai $F 50 $r; cell openai1m $O 100 $r; done
echo done > $R/status
