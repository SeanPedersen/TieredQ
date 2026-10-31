#!/bin/bash
# Issue #4 at 1M: equal-memory match and timing on DBpedia OpenAI-1536 1M.
#  1. match_memory.py --quick against RaBitQ, then Qdrant TQ+, at 1/2/4 bits/dim and seeds 0-2: one lane and output
#     directory per bits/dim (CPUs 8-15, 16-23, 24-31), so both references share the TieredQ probes near their
#     targets. No reference cells exist at 1M, so each reference is measured once here.
#  2. alone on CPUs 0-7, through run.py (seed 0, threads 1/8 x batch 1/32, build time): per bits/dim the reference,
#     then TieredQ at every budget matched to it (disk, RAM and peak targets of both references), then the other
#     reference.
# TQ+ uses the shim built for the #4 final run (same turboquant/native source).
set -u
W=/home/sean/experiments/TieredQ-issue4-1m-20261030
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
TQ_LIB=/home/sean/experiments/TieredQ-issue4-final-20261029/source-tq/benchmark/turboquant/native/target/release/libqdrant_bench.so
B=$W/source/benchmark
cd $W
date -u > match-started-utc.txt
mkdir -p $B/turboquant/native/target/release && cp -n $TQ_LIB $B/turboquant/native/target/release/
# Build the native module once so the parallel lanes do not race on it.
(cd $B && taskset -c 8-23 $P -c "from native_build import build_native; build_native()") > native-build.log 2>&1 ||
  { echo native > match-exit-code; exit 1; }
match() {  # $1 bits/dim, $2 cpus
  status=0
  for ref in rabitq qdrant; do
    (cd $B && taskset -c $2 $P -u match_memory.py --quick --reference $ref --reference-bits $1 --datasets openai1m \
      --data $W/data --output $W/match-b$1 >> $W/match-b$1.log 2>&1) || { status=$?; break; }
  done
  echo $status > match-b$1-exit-code
}
match 1 8-15 & match 2 16-23 & match 4 24-31 & wait
date -u > match-finished-utc.txt
O=$W/timing
mkdir -p $O
export PYTHONPATH=$B/build-native OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > $O/host-processes.txt
date -u > $O/started-utc.txt
cell() {  # $1 codec, $2 bits, $3 code bits (TieredQ only)
  name=$O/cell-openai1m-$1-$2${3:+-c$3}.json
  [ -f $name ] && return
  extra=(); [ -n "${3:-}" ] && extra=(--code-bits $3)
  (cd $B && taskset -c 0-7 $P -u run.py --codec $1 --datasets openai1m --bits $2 "${extra[@]}" --seeds 0 --data $W/data \
    --output $O --cell $name >> $O/run.log 2>&1) || echo "$1 $2 ${3:-}" >> $O/failed.txt
}
budgets() {  # $1 bits -> distinct TieredQ code budgets matched to either reference at these bits/dim
  $P -c "import json
print(*sorted({r['code_bits'] for ref in ('rabitq', 'qdrant') for r in json.load(open('$W/match-b$1/matches-'+ref+'.json'))
               if r['code_bits']}))"
}
for bits in 1 2 4; do
  cell rabitq $bits
  for code in $(budgets $bits); do cell tieredq $(( (code + 1535) / 1536 )) $code; done
  cell qdrant $bits
done
date -u > $O/finished-utc.txt
echo 0 > job-exit-code
