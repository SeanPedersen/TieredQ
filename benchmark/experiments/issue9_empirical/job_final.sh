#!/bin/bash
# Issue #9 final verification on source-metric (variance^1.5 fast default, native L2, AVX-512 wide encoder):
#  1. SIFT equal-memory re-match vs Qdrant TQ+ and RaBitQ with the native L2 metric (CPUs 16-23), beside the
#     running OpenAI/Gemini RaBitQ match of job_ship.sh;
#  2. alone on CPUs 0-7: train/add/search cost of c85b0f8 (base), job_ship's tree (ship) and source-metric (final),
#     3 interleaved rounds; then timing at every matched budget of the final tree.
set -u
W=/home/sean/experiments/TieredQ-issue9-ship-20261029
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
cd $W
date -u > final-started-utc.txt
mkdir -p match-sift
[ -f match-sift/cells.csv ] || awk -F, 'NR==1 || ($1=="sift" && $4!="tieredq")' \
  ../TieredQ-issue20-ship-20261029/ship-match-a/cells.csv > match-sift/cells.csv
(cd source-metric/benchmark && for ref in qdrant rabitq; do
  taskset -c 16-23 $P -u match_memory.py --quick --reference $ref --reference-bits 1 2 4 --datasets sift \
    --data $FINAL_DATA --output $W/match-sift >> $W/match-sift.log 2>&1 || exit 1
done); echo $? > match-sift-exit-code
cmake -S source-ship/benchmark/experiments/issue9_empirical -B build-final-time -DTIEREDQ_SOURCE=$W/source-metric \
  -DCMAKE_BUILD_TYPE=Release > build-final-time.log 2>&1 && taskset -c 16-23 cmake --build build-final-time -j8 >> build-final-time.log 2>&1 ||
  { echo build > final-exit-code; exit 1; }
until [ -f match-b-exit-code ]; do sleep 30; done
date -u > final-timing-started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > final-timing-host-processes.txt
timing_budgets() {
  case $1 in glove) echo "200 200 400 600 800" ;; sift) echo "128 128 256 384 512" ;;
    gemini) echo "768 768 1536 2304 3072" ;; openai) echo "1536 1536 3072 4608 6144" ;; esac
}
arms=(base ship final)
for round in 0 1 2; do
  for ds in glove sift gemini openai; do
    read d b <<< "$(timing_budgets $ds)"
    for i in 0 1 2; do
      arm=${arms[$(( (i + round) % 3 ))]}
      OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 build-$arm-time/issue9_build_time $arm data/$ds $d 1 $b \
        >> final-time-$ds.tsv 2>> errors-final-time.log || { echo time > final-exit-code; exit 1; }
    done
  done
done
date -u > final-timing-finished-utc.txt
# Matched budgets: GloVe from match-a (its SIFT rows predate the native metric), SIFT, OpenAI and Gemini.
mkdir -p matched-final/glove matched-final/sift matched-final/dbpedia
for ref in qdrant rabitq; do
  $P -c "import json,sys; r=[x for x in json.load(open('match-a/matches-$ref.json')) if x['dataset']=='glove']; json.dump(r,open('matched-final/glove/matches-$ref.json','w'))"
  cp match-sift/matches-$ref.json matched-final/sift/; cp match-b/matches-$ref.json matched-final/dbpedia/ 2>/dev/null
done
cp /home/sean/experiments/TieredQ-issue20-ship-20261029/time_matched.py .
FINAL_SOURCE=source-metric MATCH_GLOB='matched-final/*' MATCHED_OUTPUT=output-final-matched-timing taskset -c 0-7 $P -u time_matched.py > final-matched-timing.log 2>&1
echo $? > final-matched-timing-exit-code
echo 0 > final-exit-code
date -u > final-finished-utc.txt
