#!/bin/bash
# TQ+ timing relaunch (first attempt reused the TieredQ tests-build dir), after the OpenAI re-time.
W=/home/sean/experiments/TieredQ-issue4-final-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
until [ -f $W/retime/exit-code ]; do sleep 15; done
cd $W
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > tq-host-processes.txt
FINAL_CODEC=qdrant FINAL_SOURCE=source-tq FINAL_OUTPUT=output-qdrant taskset -c 0-7 $P -u run_final.py > timing-qdrant.log 2>&1
echo $? > timing-qdrant-exit-code
date -u > tq-finished-utc.txt
