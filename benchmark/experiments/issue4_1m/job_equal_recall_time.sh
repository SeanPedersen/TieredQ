#!/bin/bash
# Issue #4 at 1M, after job_match_time.sh's match lanes: equal-recall budgets, then timing alone on CPUs 0-7.
#  1. equal_recall.py: per reference cell, the smallest TieredQ budget reaching its recall. The 100k sets (and
#     SIFT1M) use the issue #9 ship matches' reference recall and run first on CPUs 0-7 while the 1M lanes match;
#     OpenAI-1M waits for all lanes, then runs one process per bits/dim on CPUs 8-31.
#  2. alone on CPUs 0-7, through run.py (seed 0, threads 1/8 x batch 1/32, build time): per bits/dim the RaBitQ
#     reference, TieredQ at every budget matched to either reference (disk, RAM, peak) and at the equal-recall
#     budgets, then Qdrant TQ+. Then TieredQ at the equal-recall budgets of the other datasets; their reference
#     timing is reused from the #4 final provenance.
set -u
W=/home/sean/experiments/TieredQ-issue4-1m-20261030
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
SHIP=/home/sean/experiments/TieredQ-issue9-ship-20261029
B=$W/source/benchmark
E=$B/experiments/issue4_1m
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
cd $W
mkdir -p equal-recall
date -u > equal-recall/started-utc.txt
ship_matches() {  # $1 dataset -> its issue #9 ship match files
  case $1 in sift) d=match-sift ;; glove) d=match-a ;; *) d=match-b ;; esac
  echo $SHIP/$d/matches-rabitq.json $SHIP/$d/matches-qdrant.json
}
for ds in glove gemini openai sift; do
  taskset -c 0-7 $P -u $E/equal_recall.py $ds $FINAL_DATA equal-recall --matches $(ship_matches $ds) \
    >> equal-recall/run.log 2>&1 || echo $ds >> equal-recall/failed.txt
done
until [ -f match-b1-exit-code ] && [ -f match-b2-exit-code ] && [ -f match-b4-exit-code ]; do sleep 30; done
date -u > match-finished-utc.txt
lanes=()
for lane in "1 8-15" "2 16-23" "4 24-31"; do
  read bits cpus <<< "$lane"
  mkdir -p equal-recall/b$bits
  taskset -c $cpus $P -u $E/equal_recall.py openai1m $W/data equal-recall/b$bits --bits $bits \
    --matches match-b$bits/matches-rabitq.json match-b$bits/matches-qdrant.json >> equal-recall/run-1m-b$bits.log 2>&1 &
  lanes+=($!)
done
wait "${lanes[@]}"
date -u > equal-recall/finished-utc.txt
O=$W/timing
mkdir -p $O
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
date -u > $O/started-utc.txt
cell() {  # $1 dataset, $2 data dir, $3 codec, $4 bits, $5 code bits (TieredQ only)
  name=$O/cell-$1-$3-$4${5:+-c$5}.json
  [ -f $name ] && return
  extra=(); [ -n "${5:-}" ] && extra=(--code-bits $5)
  (cd $B && taskset -c 0-7 $P -u run.py --codec $3 --datasets $1 --bits $4 "${extra[@]}" --seeds 0 --data $2 \
    --output $O --cell $name >> $O/run.log 2>&1) || echo "$1 $3 $4 ${5:-}" >> $O/failed.txt
}
budgets() {  # $1 dataset, $2 bits/dim or "", files... -> distinct TieredQ budgets (matched and equal-recall)
  $P - "$@" <<'PY'
import csv, json, sys
dataset, bits, files = sys.argv[1], sys.argv[2], sys.argv[3:]
codes = set()
for path in files:
    rows = csv.DictReader(open(path)) if path.endswith(".csv") else json.load(open(path))
    codes |= {int(r["code_bits"]) for r in rows if r["dataset"] == dataset and r["code_bits"]
              and (not bits or int(r["reference_bits"]) == int(bits))}
print(*sorted(codes))
PY
}
dimensions() { case $1 in sift) echo 128 ;; glove) echo 200 ;; gemini) echo 768 ;; *) echo 1536 ;; esac; }
for bits in 1 2 4; do
  cell openai1m $W/data rabitq $bits
  for code in $(budgets openai1m $bits match-b$bits/matches-rabitq.json match-b$bits/matches-qdrant.json \
                equal-recall/b$bits/equal-recall-openai1m.csv); do
    cell openai1m $W/data tieredq $(( (code + 1535) / 1536 )) $code
  done
  cell openai1m $W/data qdrant $bits
done
for ds in glove gemini openai sift; do
  d=$(dimensions $ds)
  for code in $(budgets $ds "" equal-recall/equal-recall-$ds.csv); do
    cell $ds $FINAL_DATA tieredq $(( (code + d - 1) / d )) $code
  done
done
date -u > $O/finished-utc.txt
echo 0 > job-exit-code
