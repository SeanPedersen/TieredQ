#!/usr/bin/env bash
# Lean timing A/B after job_ab.sh's OpenAI 4-bit ABBA: only the configs whose code path the change touches (fine-tier
# and exact models with fine ranks), k=10, one baseline/candidate pair each, alone on CPUs 0-7.
# Usage: W=<workspace> BASE=<baseline source> CAND=<candidate source> DATA=<cache> bash job_timing_quick.sh
set -euo pipefail
W=${W:?}; BASE=${BASE:?}; CAND=${CAND:?}; DATA=${DATA:?}
P=${P:-/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python}
CONFIGS=("openai fine 8" "openai exact 8" "gemini fine 4" "gemini fine 8" "gemini exact 8")
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$W/cells-quick" "$W/tmp"
echo "$(date -u +%FT%TZ) quick timing" >> "$W/phases.log"
for config in "${CONFIGS[@]}"; do
  read -r ds variant bits <<< "$config"
  for tree in base cand; do
    src=$BASE; [ "$tree" = cand ] && src=$CAND
    name=$W/cells-quick/$ds-$variant-b$bits-$tree-r1.json
    [ -f "$name" ] && continue
    (cd "$src/benchmark" && PYTHONPATH="$src/benchmark/build-native:$src/benchmark" BENCH_SEED=0 taskset -c 0-7 "$P" -u \
      run.py --codec tieredq --variant "$variant" --datasets "$ds" --bits "$bits" --seeds 0 --k 10 \
      --schedules 8x32 1x1 8x1 --data "$DATA" --output "$W/tmp" --cell "$name" >> "$W/timing-quick.log" 2>&1) ||
      echo "quick $config $tree" >> "$W/failed.txt"
  done
done
echo "$(date -u +%FT%TZ) quick done" >> "$W/phases.log"
[ -f "$W/failed.txt" ] && { echo 1 > "$W/exit-code-quick"; exit 1; }
echo 0 > "$W/exit-code-quick"
