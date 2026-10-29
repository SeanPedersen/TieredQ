#!/bin/bash
# Interleaved speed A/B of two trees: fresh run.py process per (index, tree), order alternating per round, CPUs 0-7.
W=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
D=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
A=${A:-source-build}; B=${B:-source-perf}; OUT=$W/${OUT:-speed-ab}
mkdir -p $OUT; ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $OUT/host-processes.txt
for round in $(seq 1 ${ROUNDS:-3}); do
  for ix in ${INDEXES:-openai:1 openai:2 openai:4 gemini:1 gemini:2 glove:4 sift:1}; do
    ds=${ix%:*}; bits=${ix#*:}
    trees="$A $B"; [ $((round % 2)) -eq 0 ] && trees="$B $A"
    for tree in $trees; do
      cell=$OUT/$tree-$ds-$bits-r$round.json
      (cd $W/$tree/benchmark && PYTHONPATH=$W/$tree/benchmark/build-native OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0 \
        taskset -c 0-7 $P run.py --codec tieredq --datasets $ds --bits $bits --seeds 0 --data $D --output $OUT --cell $cell > /dev/null 2>>$OUT/errors.log) || exit 1
      echo "$tree $ds $bits r$round done"
    done
  done
done
echo 0 > $OUT/exit-code
