#!/bin/bash
# Issue #22 phase 1, flat runs on MS MARCO-1M (Cohere v3, 1,024-d): equal memory (run 2), then equal recall (run 4).
#  1. match_memory.py --quick on msmarco1m against RaBitQ, then Qdrant TQ+, at 1/2/4 bits/dim and seeds 0-2: one lane
#     and output directory per bits/dim (CPUs 8-15, 16-23, 24-31). Every cell also scores the doc queries on the same
#     index (companion_recall10), so one match yields both recalls; budgets are selected on storage only.
#  2. equal_recall.py per lane: msmarco1m (real) then msmarco1m-doc, against each reference's mean recall on that set.
#  3. alone on CPUs 0-7, through run.py (seed 0, threads 1/8 x batch 1/32, build time), per bits/dim: real queries
#     (RaBitQ, TieredQ at every matched and real equal-recall budget, TQ+), then doc queries (RaBitQ, TieredQ at the
#     doc equal-recall budgets, TQ+). References are timed on both sets: RaBitQ's bound pruning is query-dependent.
set -u
W=/home/sean/experiments/TieredQ-issue22-flat-20261103
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
DATA=/home/sean/experiments/TieredQ-issue22-msmarco-20261031/data
B=$W/source/benchmark
E=$B/experiments/issue4_1m
D=1024
cd $W
date -u > match-started-utc.txt
(cd $B && taskset -c 8-23 $P -c "from native_build import build_native; build_native()") > native-build.log 2>&1 ||
  { echo native > job-exit-code; exit 1; }
lane() {  # $1 bits/dim, $2 cpus: match against both references, then equal recall on both query sets
  status=0
  for ref in rabitq qdrant; do
    (cd $B && taskset -c $2 $P -u match_memory.py --quick --reference $ref --reference-bits $1 --datasets msmarco1m \
      --data $DATA --output $W/match-b$1 >> $W/match-b$1.log 2>&1) || { status=$?; break; }
  done
  echo $status > match-b$1-exit-code
  [ $status = 0 ] || return
  mkdir -p equal-recall/b$1
  for ds in msmarco1m msmarco1m-doc; do
    (export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
     taskset -c $2 $P -u $E/equal_recall.py $ds $DATA equal-recall/b$1 --bits $1 \
       --matches match-b$1/matches-rabitq.json match-b$1/matches-qdrant.json >> equal-recall/run-b$1.log 2>&1) ||
      echo "$1 $ds" >> equal-recall/failed.txt
  done
}
lane 1 8-15 & lane 2 16-23 & lane 4 24-31 & wait
date -u > match-finished-utc.txt
O=$W/timing
mkdir -p $O
export PYTHONPATH=$B/build-native OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
date -u > $O/started-utc.txt
cell() {  # $1 dataset, $2 codec, $3 bits, $4 code bits (TieredQ only)
  name=$O/cell-$1-$2-$3${4:+-c$4}.json
  [ -f $name ] && return
  extra=(); [ -n "${4:-}" ] && extra=(--code-bits $4)
  (cd $B && taskset -c 0-7 $P -u run.py --codec $2 --datasets $1 --bits $3 "${extra[@]}" --seeds 0 --data $DATA \
    --output $O --cell $name >> $O/run.log 2>&1) || echo "$1 $2 $3 ${4:-}" >> $O/failed.txt
}
budgets() {  # $1 dataset, $2 bits/dim, files... -> distinct TieredQ budgets (matched and equal-recall)
  $P - "$@" <<'PY'
import csv, json, sys
dataset, bits, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
codes = set()
for path in files:
    try:
        rows = list(csv.DictReader(open(path))) if path.endswith(".csv") else json.load(open(path))
    except FileNotFoundError:
        continue
    codes |= {int(r["code_bits"]) for r in rows if r["dataset"] == dataset and r["code_bits"]
              and int(r["reference_bits"]) == bits}
print(*sorted(codes))
PY
}
for bits in 1 2 4; do
  cell msmarco1m rabitq $bits
  for code in $(budgets msmarco1m $bits match-b$bits/matches-rabitq.json match-b$bits/matches-qdrant.json \
                equal-recall/b$bits/equal-recall-msmarco1m.csv); do
    cell msmarco1m tieredq $(( (code + D - 1) / D )) $code
  done
  cell msmarco1m qdrant $bits
done
for bits in 1 2 4; do
  cell msmarco1m-doc rabitq $bits
  for code in $(budgets msmarco1m-doc $bits equal-recall/b$bits/equal-recall-msmarco1m-doc.csv); do
    cell msmarco1m-doc tieredq $(( (code + D - 1) / D )) $code
  done
  cell msmarco1m-doc qdrant $bits
done
date -u > $O/finished-utc.txt
echo 0 > job-exit-code
