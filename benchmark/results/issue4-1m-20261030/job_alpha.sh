#!/bin/bash
# Issue #4 at 1M: fast-mode α check on DBpedia OpenAI-1536 1M (999k base, the 100k set's 1k held-out queries).
# Exports the prepared set for ab_eval, then measures R@10 (seeds 0-2, even/odd query halves) at 1-4 bits/dim for
# allocation power 1, 1.25, 1.5 (the shipped inner-product default, arm `base`), 1.75 and 2. Each process holds the
# 6.1 GB base, so at most three lanes run at once. Choose with `heldout.py alpha` (even ids), report odd ids.
set -u
W=/home/sean/experiments/TieredQ-issue4-1m-20261030
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
cd $W
date -u > alpha-started-utc.txt
[ -f abdata/openai1m/truth.u32 ] || taskset -c 16-23 $P source/benchmark/experiments/issue20_additive/export_data.py \
  data abdata openai1m > export.log 2>&1 || { echo export > alpha-exit-code; exit 1; }
mkdir -p alpha
arm() {  # $1 label, $2 allocation power ("" keeps the default), $3 cpus
  [ -n "$2" ] && export AB_POWER=$2 || unset AB_POWER
  OPENBLAS_NUM_THREADS=1 taskset -c $3 build-ab/ab_eval $1 abdata/openai1m 1536 0 1536 3072 4608 6144 \
    >> alpha/ab-openai1m.tsv 2>> alpha/errors-$1.log || echo $1 >> alpha/failed.txt
}
arm base "" 8-15 & arm p1 1 16-23 & arm p1.25 1.25 24-31 & wait
arm p1.75 1.75 8-15 & arm p2 2 16-23 & wait
$P source/benchmark/experiments/issue9_empirical/heldout.py alpha > alpha/heldout.txt
date -u > alpha-finished-utc.txt
echo 0 > alpha-exit-code
