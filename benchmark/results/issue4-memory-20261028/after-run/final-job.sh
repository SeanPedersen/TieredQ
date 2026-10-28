#!/bin/bash
cd ~/experiments/TieredQ-issue4-streamed-load-20261028/source/benchmark
date -u > ../../final-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > ../../final-host-processes.txt
taskset -c 0-7 ~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u measure_memory.py --data ../../data --output ../../final
echo $? > ../../final-exit-code
