#!/usr/bin/env bash
set -euo pipefail
W=${WORKSPACE:-/home/sean/experiments/TieredQ-issue4-current-20261105}
G=$W/glove1m
B=$W/source/benchmark
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export PYTHONPATH="$B/build-native:$B" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
cd "$B"
date -u +%FT%TZ > "$G/match-started-utc.txt"
ps -eo pid,pcpu,rss,args --sort=-pcpu > "$G/match-host-processes.txt"
lane() {
  local bits=$1 cpus=$2 status=0
  for ref in rabitq qdrant; do
    taskset -c "$cpus" "$P" -u match_memory.py --quick --reference "$ref" --reference-bits "$bits" --datasets glove1m --data "$W/data" --output "$G/match-b$bits" >> "$G/match-b$bits.log" 2>&1 || { status=$?; break; }
  done
  if [ "$status" = 0 ]; then
    taskset -c "$cpus" "$P" -u experiments/issue4_current/equal_recall.py "$G" "$bits" glove1m >> "$G/equal-recall-b$bits.log" 2>&1 || status=$?
  fi
  echo "$status" > "$G/match-b$bits-exit-code"
  return "$status"
}
lane 1 8-15 & p1=$!
lane 2 16-23 & p2=$!
lane 4 24-31 & p4=$!
status=0
for pid in "$p1" "$p2" "$p4"; do wait "$pid" || status=1; done
date -u +%FT%TZ > "$G/match-finished-utc.txt"
echo "$status" > "$G/match-exit-code"
exit "$status"
