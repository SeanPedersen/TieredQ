#!/bin/bash
# TQ+ memory now on CPUs 26-35 (memory is contention-insensitive); TQ+ timing alone on 0-7 after every memory job.
W=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
cd $W/source-tq/benchmark
date -u > $W/tq-started-utc.txt
taskset -c 26-35 $P -u measure_memory.py --quick --codecs qdrant --seeds 0 1 2 --data $FINAL_DATA --output $W/output-memory-qdrant > $W/memory-qdrant.log 2>&1
echo $? > $W/memory-qdrant-exit-code
until [ -f $W/finished-utc.txt ] && [ -f $W/memory-a2-exit-code ]; do sleep 20; done
cd $W
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > tq-host-processes.txt
FINAL_CODEC=qdrant FINAL_SOURCE=source-tq FINAL_OUTPUT=output-qdrant taskset -c 0-7 $P -u run_final.py > timing-qdrant.log 2>&1
echo $? > timing-qdrant-exit-code
date -u > tq-finished-utc.txt
