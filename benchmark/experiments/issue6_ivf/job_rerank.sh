#!/usr/bin/env bash
# Issue #19.1 / #22: IVF + exact rerank sweep. Each CONFIGS line is "label lists mode budget [ENV=value...]" (a line's
# NPROBES=a,b and RERANK=k override the globals, as report_rerank.py select emits them); rows go to
# OUT/<label>-<lists>-<budget>.tsv with the label prepended. Untimed lanes run in parallel on CPUs 8-31 (LANES), timed
# runs (TIMED=1) one at a time on CPUs 0-7. NPROBES / RERANK / LANES / CONFIGS / DATA / CODEC_LISTS from the environment.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?}" "${CONFIGS:?}" "${OUT:?}" "${NPROBES:?}" "${RERANK:=10,20,50,100,200}" "${TIMED:=0}" "${CODEC_LISTS:=64}"
: "${LANES:=8-15 16-23 24-31}" "${BIN:=bin/ivf_eval_rerank}"
mkdir -p "$OUT"
run() {  # cpus label lists mode budget env...
  local cpus=$1 label=$2 lists=$3 mode=$4 budget=$5 nprobes=$NPROBES env=() arg; shift 5
  for arg; do [[ $arg == NPROBES=* ]] && nprobes=${arg#NPROBES=} || env+=("$arg"); done
  env BENCH_SEED=0 RERANK="$RERANK" LISTS="$lists" "${env[@]}" taskset -c "$cpus" "$BIN" "$mode" "$budget" "$DATA" 1536 \
    "$CODEC_LISTS" "$TIMED" ${nprobes//,/ } 2>>"$OUT/stderr.log" | grep -P '^(tieredq|rabitq)\t' |
    sed "s/^/$label\t/" >>"$OUT/$label-$lists-$budget.tsv"
}
mapfile -t lines <<<"$CONFIGS"
if [ "$TIMED" = 1 ]; then
  for line in "${lines[@]}"; do [ -n "$line" ] && run 0-7 $line; done
  exit 0
fi
read -ra lanes <<<"$LANES"
for lane in "${!lanes[@]}"; do
  (for ((i=lane;i<${#lines[@]};i+=${#lanes[@]})); do [ -n "${lines[i]}" ] && run "${lanes[lane]}" ${lines[i]}; done) &
done
wait
