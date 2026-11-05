#!/usr/bin/env bash
# #4 precheck after #31: TieredQ fast index with fine tiers on (default), off, and exact fast scores, against official
# RaBitQ (flat) and Qdrant TQ+ at 1/2/4/8 bits/dim, on the cached #4 OpenAI-1536 100K and Gemini-768 99K splits.
#  1. recall lanes (3 lanes on CPUs 8-31): seeds 1-2 of every nominal cell; seed-0 memory cells (steady RAM);
#  2. equal-disk TieredQ budgets per reference cell and variant (match_disk.py, seed-1 reference bytes/vector);
#  3. recall lanes: seeds 1-2 of every matched cell;
#  4. timing alone on CPUs 0-7 (their SMT siblings 18-25 idle): seed 0 of every cell, codecs interleaved per budget.
# Every phase resumes: finished cells are skipped. Usage: W=<workspace> bash job.sh
set -euo pipefail
W=${W:?set W to the workspace}
P=${P:-/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python}
B=$W/source/benchmark
E=$B/experiments/issue4_precheck
DATA=$W/cache
O=$W/out
DATASETS=(openai gemini)
BITS=(1 2 4 8)
QDRANT_BITS=" 1 2 4 "
VARIANTS=(fine nofine exact)
LANES=(8-15 16-23 24-31)
SCHEDULES=(8x32 8x1000 1x1 8x1)
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$O/cells" "$O/memory" "$O/tmp"
cd "$O"

cell() {  # cpus log dataset codec variant bits seed [code_bits]
  local cpus=$1 log=$2 ds=$3 codec=$4 variant=$5 bits=$6 seed=$7 code=${8:-}
  [ "$codec" = tieredq ] || variant=""
  local name=$O/cells/$ds-$codec${variant:+-$variant}-b$bits${code:+-c$code}-s$seed.json
  [ -f "$name" ] && return 0
  local extra=() k=(10 100)
  [ -n "$variant" ] && extra+=(--variant "$variant")
  [ -n "$code" ] && extra+=(--code-bits "$code")
  [ "$codec" = qdrant ] && k=(10)
  (cd "$B" && BENCH_SEED=$seed taskset -c "$cpus" "$P" -u run.py --codec "$codec" --datasets "$ds" --bits "$bits" \
    --seeds "$seed" "${extra[@]}" --k "${k[@]}" --schedules "${SCHEDULES[@]}" --data "$DATA" --output "$O/tmp" \
    --cell "$name" >> "$log" 2>&1) || echo "cell $ds $codec $variant $bits $seed $code" >> "$O/failed.txt"
}

memory() {  # cpus log dataset codec variant bits
  local done=$O/memory/done-$3-$4-$5-$6
  [ -f "$done" ] && return 0
  (cd "$B" && BENCH_SEED=0 taskset -c "$1" "$P" -u "$E/memory.py" "$O/memory" "$DATA" "$3" "$4" "$5" "$6" 0 \
    >> "$2" 2>&1) && touch "$done" || echo "memory $3 $4 $5 $6" >> "$O/failed.txt"
}

nominal() {  # dataset codec variant bits, one line per nominal cell
  for ds in "${DATASETS[@]}"; do for bits in "${BITS[@]}"; do
    for variant in "${VARIANTS[@]}"; do echo "$ds tieredq $variant $bits"; done
    echo "$ds rabitq - $bits"
    if [[ $QDRANT_BITS == *" $bits "* ]]; then echo "$ds qdrant - $bits"; fi
  done; done
}

matched() {  # dataset tieredq variant reference_bits code_bits, one line per distinct matched cell
  "$P" - "$O"/matches-*.json <<'PY'
import json, sys
seen = set()
for path in sys.argv[1:]:
    for m in json.load(open(path)):
        line = f"{m['dataset']} tieredq {m['variant']} {m['reference_bits']} {m['code_bits']}"
        if line not in seen:
            seen.add(line)
            print(line)
PY
}

lanes() {  # task-file kind: run the file's tasks round-robin on the recall lanes
  local tasks=$1 kind=$2
  for i in "${!LANES[@]}"; do
    (awk -v n=${#LANES[@]} -v i="$i" 'NR % n == i' "$tasks" | while read -r ds codec variant bits code; do
      case $kind in
        recall) for seed in 1 2; do cell "${LANES[$i]}" "$O/lane-$i.log" "$ds" "$codec" "$variant" "$bits" "$seed" "${code:-}"; done ;;
        memory) memory "${LANES[$i]}" "$O/lane-$i.log" "$ds" "$codec" "$variant" "$bits" ;;
      esac
    done) &
  done
  wait
}

phase() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/phases.log"; }

phase "start $(cat "$W/source/SOURCE_REVISION")"
ps -eo pid,pcpu,rss,psr,args --sort=-pcpu | head -25 > "$O/host-processes-start.txt"
(cd "$B" && taskset -c 8-31 "$P" -c "from native_build import build_native; build_native()") > "$O/native-build.log" 2>&1

phase "1 recall lanes: nominal seeds 1-2, then seed-0 memory cells"
nominal > "$O/tasks-nominal.txt"
lanes "$O/tasks-nominal.txt" recall
lanes "$O/tasks-nominal.txt" memory

phase "2 equal-disk calibration"
for i in "${!DATASETS[@]}"; do
  ds=${DATASETS[$i]}
  [ -f "$O/matches-$ds.json" ] && continue
  (cd "$B" && taskset -c "${LANES[$i]}" "$P" -u "$E/match_disk.py" "$ds" "$DATA" "$O" \
    "$O"/cells/"$ds"-rabitq-b*-s1.json "$O"/cells/"$ds"-qdrant-b*-s1.json >> "$O/match-$ds.log" 2>&1) &
done
wait
for ds in "${DATASETS[@]}"; do [ -f "$O/matches-$ds.json" ] || { echo "match $ds" >> "$O/failed.txt"; exit 1; }; done

phase "3 recall lanes: matched seeds 1-2"
matched > "$O/tasks-matched.txt"
lanes "$O/tasks-matched.txt" recall

phase "4 timing on CPUs 0-7"
ps -eo pid,pcpu,rss,psr,args --sort=-pcpu | head -25 > "$O/host-processes-timing.txt"
for ds in "${DATASETS[@]}"; do for bits in "${BITS[@]}"; do
  # One budget's cells back to back: nominal codecs, then the TieredQ variants matched to that budget's references.
  { nominal | awk -v d="$ds" -v b="$bits" '$1 == d && $4 == b'
    awk -v d="$ds" -v b="$bits" '$1 == d && $4 == b' "$O/tasks-matched.txt"; } |
  while read -r d codec variant b code; do cell 0-7 "$O/timing.log" "$d" "$codec" "$variant" "$b" 0 "${code:-}"; done
done; done

phase "done"
[ -f "$O/failed.txt" ] && { echo 1 > "$O/exit-code"; exit 1; }
echo 0 > "$O/exit-code"
