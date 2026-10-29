#!/bin/bash
# Re-match with the step-down fix (match_memory.py steps below the smallest cached budget when every seed-noise
# confirmation fails), reusing all cached cells, then time only budgets not yet timed, alone on CPUs 0-7.
W=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
cd $W
for part in final-match-a final-match-b; do
  mkdir -p first-pass/$part && cp $part/matches-* $part/meta-* first-pass/$part/
done
command rm -f rematch-a-exit-code rematch-b-exit-code
date -u > rematch-started-utc.txt
cd source-final/benchmark
part() {  # $1 cpus, $2 output, $3 datasets
  for ref in qdrant rabitq; do
    taskset -c $1 $P -u match_memory.py --quick --reference $ref --reference-bits 1 2 4 --datasets $3 --data $FINAL_DATA --output $W/$2 >> $W/rematch-$2.log 2>&1 || return 1
  done
}
part 0-7 final-match-a "sift glove" & A=$!
sleep 60  # cached runs start fast; stagger anyway
part 8-15 final-match-b "openai gemini" & B=$!
wait $A; echo $? > $W/rematch-a-exit-code
wait $B; echo $? > $W/rematch-b-exit-code
cd $W
[ "$(cat rematch-a-exit-code)$(cat rematch-b-exit-code)" = 00 ] || exit 1
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > rematch-timing-host-processes.txt
FINAL_SOURCE=source-final MATCH_GLOB='final-match-*' MATCHED_OUTPUT=output-final-matched-timing taskset -c 0-7 $P -u time_matched.py > rematch-timing.log 2>&1
echo $? > rematch-timing-exit-code
date -u > rematch-finished-utc.txt
