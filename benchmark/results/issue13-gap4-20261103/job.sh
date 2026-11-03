#!/bin/bash
# Issue #13 4-bit gap: variants that avoid fine (4-group) scale sections, at RaBitQ's 4 bits/dim recall on OpenAI-100K/1M.
# base = main; mb6/mb5 = TieredQConfig::max_bits cap (TQ_MAX_BITS workspace hook); f8 = 8-group fine sections (source-f8).
# Bisects in 3 lanes on CPUs 8-31, then times every variant at its own budget through run.py, 3 interleaved rounds,
# alone on CPUs 0-7, seed 0.
set -u
W=$HOME/experiments/TieredQ-issue13-gap4-20261103; cd $W
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
F=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data; O=$HOME/experiments/TieredQ-issue4-1m-20261030/data
M100=$HOME/experiments/TieredQ-issue9-ship-20261029/match-b/matches-rabitq.json
M1M=$HOME/experiments/TieredQ-issue4-1m-20261030/match-b4/matches-rabitq.json
VARIANTS="base mb6 mb5 f8"
for s in source source-f8; do
  (cd $W/$s/benchmark && taskset -c 16-31 $P -c "from native_build import build_native; build_native()") > $s-build.log 2>&1 \
    || { echo build-failed-$s > status; exit 1; }
done
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
src() { [ $1 = f8 ] && echo $W/source-f8/benchmark || echo $W/source/benchmark; }
venv() {  # prints the env assignments for a variant
  local cap=8; case $1 in mb6) cap=6 ;; mb5) cap=5 ;; esac
  echo PYTHONPATH=$(src $1)/build-native:$(src $1) TQ_MAX_BITS=$cap; }
bisect() { local v=$1 cpus=$2; mkdir -p equal-recall/$v
  for pair in "openai $F $M100" "openai1m $O $M1M"; do read ds data match <<< "$pair"
    env $(venv $v) taskset -c $cpus $P -u $(src $v)/experiments/issue4_1m/equal_recall.py $ds $data equal-recall/$v \
      --bits 4 --matches $match >> equal-recall/$v/run.log 2>&1 || echo "$v $ds" >> equal-recall/failed.txt
  done; }
echo bisect > status
( bisect base 8-15; bisect f8 8-15 ) & ( bisect mb6 16-23 ) & ( bisect mb5 24-31 ) & wait
echo timing > status
code() { cut -d, -f5 equal-recall/$1/equal-recall-$2.csv 2>/dev/null | grep -v code_bits; }
cell() { local v=$1 ds=$2 data=$3 r=$4 c; c=$(code $v $ds); [ -n "$c" ] || return
  local out=$W/timing/r$r/$v; mkdir -p $out
  (cd $(src $v) && env $(venv $v) taskset -c 0-7 $P -u run.py --codec tieredq --datasets $ds --bits $(((c+1535)/1536)) \
    --code-bits $c --seeds 0 --data $data --output $out --cell $out/cell-$ds-c$c.json >> $W/timing/run.log 2>&1) \
    || echo "$v $ds r$r" >> $W/timing/failed.txt; }
for r in 1 2 3; do for v in $VARIANTS; do cell $v openai $F $r; cell $v openai1m $O $r; done; done
echo done > status
