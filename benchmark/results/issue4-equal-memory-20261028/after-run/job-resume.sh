#!/bin/bash
cd /home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/source/benchmark
taskset -c 0-7 /home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u match_memory.py --quick --data /home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data --output /home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/output
echo $? > /home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/exit-code
date -u > /home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/finished-utc.txt
