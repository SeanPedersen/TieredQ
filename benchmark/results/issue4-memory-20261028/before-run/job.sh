#!/bin/bash
cd ~/experiments/TieredQ-issue4-memory-20261028/source/benchmark
OLD=~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark
date -u > ../../started-utc.txt
lscpu > ../../host-cpu.txt; free -b > ../../host-memory.txt; ps -eo pid,pcpu,rss,args --sort=-pcpu | head -20 > ../../host-processes.txt
taskset -c 0-7 $OLD/.venv/bin/python -u measure_memory.py --data ../../data --output ../../output
echo $? > ../../exit-code
