#!/bin/bash
# Issue #9: DBpedia OpenAI/Gemini speed and build time of TieredQ (final tree, RAM-matched budgets of match-b) next to
# RaBitQ and Qdrant TQ+ at 1/2/4 bits/dim, all through run.py (seed 0, 1/8 threads x batch 1/32) in one session,
# codecs alternating, alone on CPUs 0-7.
set -u
W=/home/sean/experiments/TieredQ-issue9-ship-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
O=$W/dbpedia-speed
B=$W/source-metric/benchmark
mkdir -p $O
cd $B
export PYTHONPATH=$B/build-native OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
date -u > $O/started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
cell() {  # $1 codec, $2 dataset, $3 bits, $4 code bits (TieredQ only)
  name=$O/cell-$2-$1-$3${4:+-c$4}.json
  [ -f $name ] && return
  extra=(); [ -n "${4:-}" ] && extra=(--code-bits $4)
  taskset -c 0-7 $P -u run.py --codec $1 --datasets $2 --bits $3 "${extra[@]}" --seeds 0 --data $DATA --output $O --cell $name \
    >> $O/run.log 2>&1 || { echo "$1 $2 $3 ${4:-}" >> $O/failed.txt; }
}
budgets() {  # $1 dataset, $2 bits -> "rabitq-matched qdrant-matched" TieredQ code bits (RAM targets)
  $P -c "import json,sys
for ref in ('rabitq','qdrant'):
    print(next(r['code_bits'] for r in json.load(open('$W/match-b/matches-'+ref+'.json')) if r['dataset']=='$1' and r['reference_bits']==$2 and r['metric']=='ram'))" | tr '\n' ' '
}
for ds in openai gemini; do
  for bits in 1 2 4; do
    read r q <<< "$(budgets $ds $bits)"
    cell rabitq $ds $bits; cell tieredq $ds $bits $r; cell qdrant $ds $bits; [ $q != $r ] && cell tieredq $ds $bits $q
  done
done
date -u > $O/finished-utc.txt
