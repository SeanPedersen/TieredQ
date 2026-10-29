#!/bin/bash
# Resume the final re-match on the committed library (source-final; part b's earlier cells came from source-perf,
# which differs only by a bit-identical vectorized binary16 decode), then time every matched budget alone on CPUs 0-7.
W=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
command rm -f $W/final-match-a-exit-code $W/final-match-b-exit-code
date -u > $W/final-match-started-utc.txt
cd $W/source-final/benchmark
part() {  # $1 cpus, $2 output, $3 datasets
  for ref in qdrant rabitq; do
    taskset -c $1 $P -u match_memory.py --quick --reference $ref --reference-bits 1 2 4 --datasets $3 --data $FINAL_DATA --output $W/$2 >> $W/$2.log 2>&1 || return 1
  done
}
part 0-7 final-match-a "sift glove" & A=$!
sleep 120  # shared native build and data loading settle before part b starts
part 8-15 final-match-b "openai gemini" & B=$!
wait $A; echo $? > $W/final-match-a-exit-code
wait $B; echo $? > $W/final-match-b-exit-code
date -u > $W/final-match-finished-utc.txt
cd $W
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > final-matched-timing-host-processes.txt
FINAL_SOURCE=source-final MATCH_GLOB='final-match-*' MATCHED_OUTPUT=output-final-matched-timing taskset -c 0-7 $P -u time_matched.py > final-matched-timing.log 2>&1
echo $? > final-matched-timing-exit-code
date -u > final-finished-utc.txt
