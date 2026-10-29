#!/bin/bash
cd /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/source/benchmark
date -u > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/quick-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/quick-host-processes.txt
taskset -c 8-15 /home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u match_memory.py --quick --data /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/data --output /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/output-quick
echo $? > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/quick-exit-code
date -u > /home/sean/experiments/TieredQ-issue4-equal-memory-20261028/quick-finished-utc.txt
