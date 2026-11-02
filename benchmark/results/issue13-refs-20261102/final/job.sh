#!/bin/bash
# Issue #13: final code. SIFT re-bisected (L2 is exact again); TieredQ timed at every equal-recall budget, alone on CPUs 0-7.
set -u
N=$HOME/experiments/TieredQ-issue13-refs-20261102; G=$N/final; OLD=$HOME/experiments/TieredQ-issue4-1m-20261030
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
F=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data; O=$OLD/data; SHIP=$HOME/experiments/TieredQ-issue9-ship-20261029; B=$N/source/benchmark
(cd $B && taskset -c 16-31 $P -c "from native_build import build_native; build_native()") > $G/build.log 2>&1 || { echo build-failed > $G/status; exit 1; }
export PYTHONPATH=$B/build-native:$B OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
cd $G
taskset -c 8-31 $P -u $B/experiments/issue4_1m/equal_recall.py sift $F equal-recall --matches $SHIP/match-sift/matches-rabitq.json $SHIP/match-sift/matches-qdrant.json > equal-recall/run.log 2>&1
cell() { name=$G/timing/cell-$1-tieredq-$3-c$4.json; [ -f $name ] && return
  (cd $B && taskset -c 0-7 $P -u run.py --codec tieredq --datasets $1 --bits $3 --code-bits $4 --seeds 0 --data $2 --output $G/timing --cell $name >> $G/timing/run.log 2>&1) || echo "$1 $4" >> $G/failed.txt; }
codes() { cut -d, -f5 "$@" | grep -v code_bits | sort -un; }
for c in $(codes $N/equal-recall/b1/equal-recall-openai1m.csv $N/equal-recall/b2/equal-recall-openai1m.csv $N/equal-recall/b4/equal-recall-openai1m.csv); do cell openai1m $O $(((c+1535)/1536)) $c; done
for c in $(codes $N/equal-recall/equal-recall-openai.csv); do cell openai $F $(((c+1535)/1536)) $c; done
for c in $(codes $N/equal-recall/equal-recall-gemini.csv); do cell gemini $F $(((c+767)/768)) $c; done
for c in $(codes $N/equal-recall/equal-recall-glove.csv); do cell glove $F $(((c+199)/200)) $c; done
for c in $(codes equal-recall/equal-recall-sift.csv); do cell sift $F $(((c+127)/128)) $c; done
echo done > $G/status
