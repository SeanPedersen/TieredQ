#!/usr/bin/env bash
# Equal steady-RAM check (#4 precheck follow-up): measure_memory probes of one TieredQ variant around another's steady
# RSS after reload, on the recall lanes (CPUs 8-31). Probes: lines "dataset variant bits code_bits" in $W/eqram/probes.txt.
# Usage: W=<workspace> bash job_equal_ram.sh
set -euo pipefail
W=${W:?set W to the workspace}
P=${P:-/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python}
B=$W/source/benchmark
O=$W/eqram
LANES=(8-15 16-23 24-31)
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$O"
for i in "${!LANES[@]}"; do
  (awk -v n=${#LANES[@]} -v i="$i" 'NR % n == i' "$O/probes.txt" | while read -r ds variant bits code; do
    (cd "$B" && BENCH_SEED=0 taskset -c "${LANES[$i]}" "$P" -u "$B/experiments/issue4_precheck/memory.py" "$O" \
      "$W/cache" "$ds" tieredq "$variant" "$bits" 0 "$code" >> "$O/lane-$i.log" 2>&1) ||
      echo "$ds $variant $bits $code" >> "$O/failed.txt"
  done) &
done
wait
[ -f "$O/failed.txt" ] && { echo 1 > "$O/exit-code"; exit 1; }
echo 0 > "$O/exit-code"
