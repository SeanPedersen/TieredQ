#!/bin/bash
# Interleaved single-core kernel A/B (baseline byte layout vs packed) on core 17 (sibling 35 idle); 65,536 documents.
cd /home/sean/experiments/TieredQ-issue12-recall/pack-opt
for shape in "8:10,7:12,5:30,4:40,3:28,2:9" "5:128" "7:128" "8:128" "7:13,6:129,5:316,4:536,3:477,2:61,1:2"; do
  for i in 1 2 3; do for v in baseline new new2 new3; do echo "== $v"; taskset -c 17 ./kbench-$v "$shape" 65536 | grep -v sink; done; done
done
