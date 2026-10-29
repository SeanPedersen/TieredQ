#!/bin/bash
cd /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/source/benchmark
date -u > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/host-processes.txt
lscpu > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/host-cpu.txt; free -b > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/host-memory.txt
taskset -c 0-7 ~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u match_memory.py --data /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/data --output /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/output
echo $? > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/exit-code
date -u > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/finished-utc.txt
