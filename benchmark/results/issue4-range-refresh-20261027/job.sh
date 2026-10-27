#!/bin/bash
cd /home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542
taskset -c 0-7 source/benchmark/.venv/bin/python -u run_refresh.py
code=$?
echo "$code" > exit-code
echo __EXIT_CODE__=$code
