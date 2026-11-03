#!/bin/bash
# Usage: prof.sh TAG CODE_BITS ROUNDS: perf the flat single-query loop (noinline snapshot source-prof), alone on CPUs 0-7.
W=$HOME/experiments/TieredQ-issue13-gap4-20261103; cd $W; mkdir -p prof
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
S=$W/source-prof/benchmark; F=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data
env PYTHONPATH=$S/build-native:$S OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 $P -u prof_driver.py openai $F $2 $3 > prof/$1.out 2> prof/$1.err &
pid=$!
until grep -q "profile ready" prof/$1.err 2>/dev/null; do sleep 0.2; done
perf record -p $pid -o prof/$1.data 2>/dev/null >/dev/null &
wait $pid; sleep 1
cat prof/$1.out
perf report -i prof/$1.data --no-children --percent-limit 0.5 --stdio 2>/dev/null | grep -v "^#" | grep -v "^$" | head -25 | cut -c1-200
