#!/bin/bash
# Issue #13: TieredQ at current main against the reused #4 reference cells. Re-bisects equal-recall budgets (CPUs 8-31),
# then times TieredQ alone on CPUs 0-7 at those and the existing matched budgets through run.py, as the #4 1M job did.
set -u
W=$HOME/experiments/TieredQ-issue13-refs-20261102; OLD=$HOME/experiments/TieredQ-issue4-1m-20261030
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
FINAL_DATA=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data; ONEM=$OLD/data; SHIP=$HOME/experiments/TieredQ-issue9-ship-20261029
B=$W/source/benchmark; E=$B/experiments/issue4_1m
cd $W
(cd $B && taskset -c 16-31 $P -c "from native_build import build_native; build_native()") > native-build.log 2>&1 || { echo build-failed > job.status; exit 1; }
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
mkdir -p equal-recall; date -u > equal-recall/started-utc.txt
ship_matches() { case $1 in sift) d=match-sift ;; glove) d=match-a ;; *) d=match-b ;; esac; echo $SHIP/$d/matches-rabitq.json $SHIP/$d/matches-qdrant.json; }
pids=()
for lane in "1 8-15" "2 16-23" "4 24-31"; do
  read bits cpus <<< "$lane"; mkdir -p equal-recall/b$bits
  ( taskset -c $cpus $P -u $E/equal_recall.py openai1m $ONEM equal-recall/b$bits --bits $bits \
      --matches $OLD/match-b$bits/matches-rabitq.json $OLD/match-b$bits/matches-qdrant.json >> equal-recall/run-1m-b$bits.log 2>&1 || echo openai1m-$bits >> equal-recall/failed.txt ) &
  pids+=($!)
done
wait "${pids[@]}"
for ds in openai gemini glove sift; do
  taskset -c 8-31 $P -u $E/equal_recall.py $ds $FINAL_DATA equal-recall --matches $(ship_matches $ds) >> equal-recall/run.log 2>&1 || echo $ds >> equal-recall/failed.txt
done
date -u > equal-recall/finished-utc.txt; echo bisect-done > job.status
O=$W/timing; mkdir -p $O; date -u > $O/started-utc.txt
cell() { name=$O/cell-$1-tieredq-$3-c$4.json; [ -f $name ] && return
  (cd $B && taskset -c 0-7 $P -u run.py --codec tieredq --datasets $1 --bits $3 --code-bits $4 --seeds 0 --data $2 --output $O --cell $name >> $O/run.log 2>&1) || echo "$1 $3 $4" >> $O/failed.txt; }
budgets() { $P - "$@" <<'PY'
import csv, json, sys
dataset, files = sys.argv[1], sys.argv[2:]
codes = set()
for path in files:
    rows = csv.DictReader(open(path)) if path.endswith(".csv") else json.load(open(path))
    codes |= {int(r["code_bits"]) for r in rows if r["dataset"] == dataset and r["code_bits"]}
print(*sorted(codes))
PY
}
for code in $(budgets openai1m equal-recall/b1/equal-recall-openai1m.csv equal-recall/b2/equal-recall-openai1m.csv equal-recall/b4/equal-recall-openai1m.csv $OLD/match-b{1,2,4}/matches-{rabitq,qdrant}.json); do
  cell openai1m $ONEM $(( (code+1535)/1536 )) $code
done
for pair in "openai 1536" "gemini 768" "glove 200" "sift 128"; do read ds d <<< "$pair"
  for code in $(budgets $ds equal-recall/equal-recall-$ds.csv $(ship_matches $ds)); do cell $ds $FINAL_DATA $(( (code+d-1)/d )) $code; done
done
date -u > $O/finished-utc.txt; echo timing-done > job.status
