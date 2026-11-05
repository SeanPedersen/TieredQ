#!/usr/bin/env bash
# Fine-tier serving RAM A/B: baseline tree vs candidate tree (shared prefix-view model arrays, one-shot fine-row load).
#  1. candidate: native module, C++ suite and benchmark tests (CPUs 8-31);
#  2. memory cells (measure_memory quick) of fine/nofine/exact at 1/2/4/8 bits/dim, both trees (3 lanes on CPUs 8-31);
#  3. timing alone on CPUs 0-7 (SMT siblings 18-25 idle): seed 0 run.py cells at 4 and 8 bits/dim, ABBA per config.
# Usage: W=<workspace> BASE=<baseline source> CAND=<candidate source> DATA=<cache> bash job_ab.sh
set -euo pipefail
W=${W:?}; BASE=${BASE:?}; CAND=${CAND:?}; DATA=${DATA:?}
P=${P:-/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python}
DATASETS=(openai gemini)
VARIANTS=(fine nofine exact)
LANES=(8-15 16-23 24-31)
SCHEDULES=(8x32 8x1000 1x1 8x1)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$W/cells" "$W/memory-base" "$W/memory-cand" "$W/tmp"
cd "$W"
phase() { echo "$(date -u +%FT%TZ) $*" | tee -a "$W/phases.log"; }

phase "start base=$(cat "$BASE/SOURCE_REVISION") cand=$(cat "$CAND/SOURCE_REVISION")"
phase "1 candidate build and tests"
for tree in "$BASE" "$CAND"; do
  (cd "$tree/benchmark" && taskset -c 8-31 "$P" -c "from native_build import build_native; build_native()") >> "$W/native-build.log" 2>&1
done
(cmake -S "$CAND" -B "$W/cand-build" -DCMAKE_BUILD_TYPE=Release > "$W/cand-cmake.log" 2>&1 &&
  taskset -c 8-31 cmake --build "$W/cand-build" -j 16 --target test_tieredq >> "$W/cand-cmake.log" 2>&1 &&
  taskset -c 8-31 "$W/cand-build/test_tieredq" > "$W/cand-ctest.log" 2>&1) || { echo "cpp tests" >> "$W/failed.txt"; exit 1; }
(cd "$CAND/benchmark" && PYTHONPATH="$CAND/benchmark/build-native" taskset -c 8-31 "$P" -m unittest test_native \
  test_benchmark test_match_memory > "$W/cand-pytest.log" 2>&1) || { echo "benchmark tests" >> "$W/failed.txt"; exit 1; }

phase "2 memory cells"
for ds in "${DATASETS[@]}"; do for bits in 1 2 4 8; do for variant in "${VARIANTS[@]}"; do for tree in base cand; do
  echo "$tree $ds $variant $bits"; done; done; done; done > "$W/tasks-memory.txt"
for i in "${!LANES[@]}"; do
  (awk -v n=${#LANES[@]} -v i="$i" 'NR % n == i' "$W/tasks-memory.txt" | while read -r tree ds variant bits; do
    src=$BASE; [ "$tree" = cand ] && src=$CAND
    (cd "$src/benchmark" && PYTHONPATH="$src/benchmark/build-native:$src/benchmark" BENCH_SEED=0 \
      taskset -c "${LANES[$i]}" "$P" -u "$CAND/benchmark/experiments/issue4_precheck/memory.py" "$W/memory-$tree" "$DATA" \
      "$ds" tieredq "$variant" "$bits" 0 >> "$W/lane-$i.log" 2>&1) || echo "memory $tree $ds $variant $bits" >> "$W/failed.txt"
  done) &
done
wait

phase "3 timing on CPUs 0-7 (ABBA)"
ps -eo pid,pcpu,rss,psr,args --sort=-pcpu | head -25 > "$W/host-processes-timing.txt"
cell() {  # tree round dataset variant bits
  local src=$BASE; [ "$1" = cand ] && src=$CAND
  local name=$W/cells/$3-$4-b$5-$1-r$2.json
  [ -f "$name" ] && return 0
  (cd "$src/benchmark" && PYTHONPATH="$src/benchmark/build-native:$src/benchmark" BENCH_SEED=0 taskset -c 0-7 "$P" -u run.py \
    --codec tieredq --variant "$4" --datasets "$3" --bits "$5" --seeds 0 --k 10 100 --schedules "${SCHEDULES[@]}" \
    --data "$DATA" --output "$W/tmp" --cell "$name" >> "$W/timing.log" 2>&1) || echo "cell $*" >> "$W/failed.txt"
}
for ds in "${DATASETS[@]}"; do for bits in 4 8; do for variant in "${VARIANTS[@]}"; do
  cell base 1 "$ds" "$variant" "$bits"; cell cand 1 "$ds" "$variant" "$bits"
  cell cand 2 "$ds" "$variant" "$bits"; cell base 2 "$ds" "$variant" "$bits"
done; done; done

phase "done"
[ -f "$W/failed.txt" ] && { echo 1 > "$W/exit-code"; exit 1; }
echo 0 > "$W/exit-code"
