#!/bin/bash
# Usage: query_peak.sh <variants...>; prints one JSON line per (variant, index, schedule).
B=~/experiments/TieredQ-issue4-build-profile-20261028
P=~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
cd ${TREE:-$B/source}/benchmark
for variant in "$@"; do
  case $variant in default) E=();; arena1) E=(MALLOC_ARENA_MAX=1);; mmap128k) E=(MALLOC_MMAP_THRESHOLD_=131072);; esac
  for ix in ${INDEXES:-openai-1 openai-2 gemini-2}; do for s in "1 1" "1 32" "8 1" "8 32"; do
    echo -n "$variant "; env "${E[@]}" PYTHONPATH=build-native:. OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 taskset -c 8-15 $P $B/query_peak.py $B/data $B/indexes/$ix.bin $s 2>&1 | grep -v Rotator
  done; done
done
