#!/bin/bash
# Interleaved A/B: every round runs each tree once per index in a fresh process; tree order rotates per round.
B=~/experiments/TieredQ-issue4-build-profile-20261028
P=~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
TREES=(${TREES:-source regen regen-tile-2048 regen-tile-1024})
OUT=${OUT:-$B/ab.jsonl}
date -u > $OUT.started; ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $OUT.host-processes
for round in $(seq 1 ${ROUNDS:-6}); do
  for ix in ${INDEXES:-openai-1 gemini-1 glove-1 openai-2 gemini-2}; do
    for offset in $(seq 0 $((${#TREES[@]}-1))); do
      t=${TREES[$(( (offset+round) % ${#TREES[@]} ))]}
      (cd $B/$t/benchmark && PYTHONPATH=build-native:. OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 $P $B/timing_probe.py $B/data $B/indexes/$ix.bin $t ${SCHEDULES:-1x1,1x32,8x1,8x32} 2>&1 | grep -v Rotator) >> $OUT
    done
  done
done
echo $? > $OUT.exit-code
