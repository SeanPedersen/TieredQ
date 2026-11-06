#!/usr/bin/env bash
# Run only named datasets with verified cached inputs. Timing is a separate, serialized phase.
set -euo pipefail
W=${WORKSPACE:-/home/sean/experiments/TieredQ-issue4-current-20261105}
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
B=$W/source/benchmark
DATASETS=("${@:?Pass datasets to measure}")
cd "$B"
export PYTHONPATH="$B/build-native:$B" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
date -u +%FT%TZ > "$W/match-started-utc.txt"
ps -eo pid,pcpu,rss,args --sort=-pcpu > "$W/match-host-processes.txt"
lane() {
  local bits=$1 cpus=$2 status=0
  taskset -c "$cpus" "$P" -u experiments/issue4_current/reference_builds.py "$W" "$bits" "${DATASETS[@]}" \
    >> "$W/reference-build-b$bits.log" 2>&1 || { echo "$?" > "$W/match-b$bits-exit-code"; return 1; }
  for ref in rabitq qdrant; do
    taskset -c "$cpus" "$P" -u match_memory.py --quick --reference "$ref" --reference-bits "$bits" \
      --datasets "${DATASETS[@]}" --data "$W/data" --output "$W/match-b$bits" >> "$W/match-b$bits.log" 2>&1 || { status=$?; break; }
  done
  if [ "$status" = 0 ]; then
    taskset -c "$cpus" "$P" -u experiments/issue4_current/equal_recall.py "$W" "$bits" "${DATASETS[@]}" \
      >> "$W/equal-recall-b$bits.log" 2>&1 || status=$?
  fi
  echo "$status" > "$W/match-b$bits-exit-code"
  return "$status"
}
lane 1 8-15 & p1=$!
lane 2 16-23 & p2=$!
lane 4 24-31 & p4=$!
status=0
for pid in "$p1" "$p2" "$p4"; do wait "$pid" || status=1; done
date -u +%FT%TZ > "$W/match-finished-utc.txt"
echo "$status" > "$W/match-exit-code"
exit "$status"
