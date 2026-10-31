#!/bin/bash
# Bits-2 lane of job_match_time.sh, restarted from its own tree copy after a RaBitQ overlay race at lane start.
W=/home/sean/experiments/TieredQ-issue4-1m-20261030
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
B=$W/source-b2/benchmark
status=0
for ref in rabitq qdrant; do
  (cd $B && taskset -c 16-23 $P -u match_memory.py --quick --reference $ref --reference-bits 2 --datasets openai1m \
    --data $W/data --output $W/match-b2 >> $W/match-b2.log 2>&1) || { status=$?; break; }
done
echo $status > $W/match-b2-exit-code
