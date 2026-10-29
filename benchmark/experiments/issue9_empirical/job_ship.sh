#!/bin/bash
# Issue #9 ship verification: exact padded allocation on variance^1.5 as the fast default (source-ship) vs c85b0f8.
#  A. in parallel: equal-memory re-match vs Qdrant TQ+ and RaBitQ (SIFT/GloVe on CPUs 0-7, OpenAI/Gemini on 8-15,
#     reusing only the reference cells of the #20 and #4 matches), fast recall lanes (16-31) and the tests (32-35);
#  B. full-mode A/B (source-ship vs source-fullp, which also allocates full targets on variance^1.5), CPUs 16-35;
#  C. alone on CPUs 0-7: train/add/search cost, base vs ship, 3 interleaved rounds; then timing at matched budgets.
# Fast base rows come from the #9 held-out run (c85b0f8, same budgets), so only the ship arm is measured.
set -u
W=/home/sean/experiments/TieredQ-issue9-ship-20261029
R=/home/sean/experiments/TieredQ-issue9-empirical-20261029/heldout
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
export FINAL_DATA=/home/sean/experiments/TieredQ-issue4-equal-memory-after-20261028/data
cd $W
date -u > started-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > host-processes.txt
build() {  # $1 name, $2 source tree, $3 experiment dir, $4 extra cmake args
  cmake -S source-ship/benchmark/experiments/$3 -B build-$1 -DTIEREDQ_SOURCE=$W/$2 -DCMAKE_BUILD_TYPE=Release $4 > build-$1.log 2>&1 &&
    taskset -c 16-35 cmake --build build-$1 -j20 >> build-$1.log 2>&1
}
build ship source-ship issue20_additive "" && build fullp source-fullp issue20_additive "" &&
  build ship-time source-ship issue9_empirical "" && build base-time source-base issue9_empirical "-DISSUE9_TIME=OFF" ||
  { echo build > exit-code; exit 1; }
budgets() {  # $1 dataset -> "d budgets"
  case $1 in
    glove) echo "200 200 300 400 450 500 550 600 650 700 750 800" ;;
    sift) echo "128 128 192 256 288 320 352 384 416 448 480 512" ;;
    gemini) echo "768 768 1152 1536 1920 2304 2688 3072" ;;
    openai) echo "1536 1536 2304 3072 3840 4608 5376 6144" ;;
  esac
}
recall() {  # $1 arm, $2 binary tree, $3 cpus, $4 output prefix, $5.. datasets; environment passes through
  arm=$1 tree=$2 cpus=$3 out=$4; shift 4
  for ds in "$@"; do
    read d b <<< "$(budgets $ds)"
    OPENBLAS_NUM_THREADS=1 taskset -c $cpus build-$tree/ab_eval $arm data/$ds $d 0 $b >> $out-$ds.tsv 2>> errors-$arm.log ||
      { echo 1 > exit-$arm-$ds; return 1; }
  done
}
match() {  # $1 cpus, $2 output, $3 reference cells, $4 datasets
  mkdir -p $2
  [ -f $2/cells.csv ] || awk -F, 'NR==1 || $4!="tieredq"' $3 > $2/cells.csv
  (cd source-ship/benchmark && for ref in qdrant rabitq; do
    taskset -c $1 $P -u match_memory.py --quick --reference $ref --reference-bits 1 2 4 --datasets $4 \
      --data $FINAL_DATA --output $W/$2 >> $W/$2.log 2>&1 || exit 1
  done)
  echo $? > $2-exit-code
}
mkdir -p fast full
for ds in glove sift gemini openai; do grep -P '^(base|layout-p1.5)\t' $R/ab-$ds.tsv > fast/ab-$ds.tsv; done
match 0-7 match-a /home/sean/experiments/TieredQ-issue20-ship-20261029/ship-match-a/cells.csv "sift glove" & MA=$!
sleep 120  # the shared native build settles before part b starts
match 8-15 match-b /home/sean/experiments/TieredQ-issue4-final-20261029/final-match-b/cells.csv "openai gemini" & MB=$!
(cmake -S source-ship -B tests-ship -DCMAKE_BUILD_TYPE=Release > tests-ship.log 2>&1 &&
  taskset -c 32-35 cmake --build tests-ship -j4 >> tests-ship.log 2>&1 &&
  taskset -c 32-35 tests-ship/test_tieredq >> tests-ship.log 2>&1; echo $? > tests-ship-exit-code) &
lanes=()
recall ship ship 16-23 fast/ab openai & lanes+=($!); recall ship ship 24-27 fast/ab gemini & lanes+=($!)
recall ship ship 28-29 fast/ab sift & lanes+=($!); recall ship ship 30-31 fast/ab glove & lanes+=($!)
wait "${lanes[@]}"
date -u > fast-finished-utc.txt
export AB_MODE=full
lanes=()
recall full-p1 ship 16-21 full/ab openai & lanes+=($!); recall full-p1.5 fullp 22-27 full/ab openai & lanes+=($!)
recall full-p1 ship 28-29 full/ab gemini & lanes+=($!); recall full-p1.5 fullp 30-31 full/ab gemini & lanes+=($!)
recall full-p1 ship 32-33 full/ab sift glove & lanes+=($!); recall full-p1.5 fullp 34-35 full/ab sift glove & lanes+=($!)
wait "${lanes[@]}"
unset AB_MODE
date -u > full-finished-utc.txt
wait $MA $MB
date -u > match-finished-utc.txt
ps -eo pid,pcpu,rss,args --sort=-pcpu | head -15 > timing-host-processes.txt
timing_budgets() {
  case $1 in glove) echo "200 200 400 600 800" ;; sift) echo "128 128 256 384 512" ;;
    gemini) echo "768 768 1536 2304 3072" ;; openai) echo "1536 1536 3072 4608 6144" ;; esac
}
arms=(base ship)
for round in 0 1 2; do
  for ds in glove sift gemini openai; do
    read d b <<< "$(timing_budgets $ds)"
    for i in 0 1; do
      arm=${arms[$(( (i + round) % 2 ))]}
      OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 build-$arm-time/issue9_build_time $arm data/$ds $d 1 $b \
        >> time-$ds.tsv 2>> errors-time.log || { echo time > exit-code; exit 1; }
    done
  done
done
date -u > timing-finished-utc.txt
cp /home/sean/experiments/TieredQ-issue20-ship-20261029/time_matched.py .
FINAL_SOURCE=source-ship MATCH_GLOB='match-*' MATCHED_OUTPUT=output-matched-timing taskset -c 0-7 $P -u time_matched.py > matched-timing.log 2>&1
echo $? > matched-timing-exit-code
echo 0 > exit-code
date -u > finished-utc.txt
