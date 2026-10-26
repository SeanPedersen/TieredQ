#!/bin/bash
# Interleaved baseline/new synthetic A/B over mixed (4, 6 bits/dim) and all-8-bit models.
cd /home/sean/experiments/TieredQ-issue12-recall/pack-opt
for d in "128 1000000 4" "1536 100000 4" "200 1000000 6" "128 1000000 8" "1536 100000 8"; do
  for v in baseline new; do echo "== $v $d"; taskset -c 0-7 ./ebench-$v $d; echo "exit $?"; done
done
echo "=== done"
