#!/bin/bash
cd ~/experiments/TieredQ-issue4-streamed-load-20261028
date -u > started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > host-processes.txt
taskset -c 0-7 ~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u ab_streamed_load.py
echo $? > exit-code
