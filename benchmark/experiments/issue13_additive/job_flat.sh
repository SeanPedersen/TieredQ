#!/usr/bin/env bash
# Flat p50 after the batch timing, alone on CPUs 0-7 (search itself is single-query, OPENBLAS single-threaded).
cd "$(dirname "$0")"
while ! grep -q '^time-done' final.log; do sleep 30; done
D=~/experiments/TieredQ-issue20-ship-20261029/data/openai
for b in 3072 6144; do OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 ./flat_p50 $D 1536 $b 3 | sed "s/^/$b\t/" >> flat-p50.tsv; done
echo flat-done >> flat-p50.log
