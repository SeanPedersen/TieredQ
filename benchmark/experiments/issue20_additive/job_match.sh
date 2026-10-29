#!/bin/bash
# Issue #20 SIFT/GloVe equal-memory re-match on the shipped tree (CPUs 0-7), reusing only the Qdrant TQ+ and RaBitQ
# reference cells of the issue #4 final-match-a; TieredQ cells are re-measured on the shared-group tree. Then time every matched budget.
set -u
W=/home/sean/experiments/TieredQ-issue20-ship-20261029
F=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
until [ -f $W/ab-finished-utc.txt ]; do sleep 20; done
cd $W
mkdir -p ship-match-a
[ -f ship-match-a/cells.csv ] || awk -F, 'NR==1 || $4!="tieredq"' $F/final-match-a/cells.csv > ship-match-a/cells.csv
cp $F/time_matched.py .
date -u > match-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > match-host-processes.txt
cd source-shared/benchmark
for ref in qdrant rabitq; do
  taskset -c 0-7 $P -u match_memory.py --quick --reference $ref --reference-bits 1 2 4 --datasets sift glove \
    --data $FINAL_DATA --output $W/ship-match-a >> $W/ship-match-a.log 2>&1 || { echo $ref > $W/match-exit-code; exit 1; }
done
echo 0 > $W/match-exit-code
cd $W
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > matched-timing-host-processes.txt
FINAL_SOURCE=source-shared MATCH_GLOB='ship-match-a' MATCHED_OUTPUT=output-ship-matched-timing taskset -c 0-7 $P -u time_matched.py > matched-timing.log 2>&1
echo $? > matched-timing-exit-code
date -u > finished-utc.txt
