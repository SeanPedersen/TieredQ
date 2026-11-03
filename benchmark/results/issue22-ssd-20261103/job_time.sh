#!/bin/bash
# Issue #22 SSD comparison, phase B: equal-recall configs from recall.tsv timed alone on CPUs 0-7, 2 interleaved rounds,
# seed 0. RaBitQ native IVF all in RAM; TieredQ nested IVF with the prefix_and_suffix tier on the home SSD (Samsung 960
# EVO, LUKS + btrfs) read with direct I/O, 32 reads in flight per query.
S=$HOME/experiments/TieredQ-issue22-ssd-20261103; D=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
cd $S; O=time.tsv; : > $O; TIER=$S/tier.rows
run() { local label=$1 mode=$2 budget=$3 probes=$4; shift 4
  env "$@" BENCH_SEED=0 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 build/issue6_ivf_eval $mode $budget $D 1536 256 1 ${probes//,/ } \
    2>/dev/null | grep -P '^(tieredq-nested|rabitq)\t' | sed "s/^/$label\t/" >> $O; }
ssd() { run $@ NESTED_TIER=direct NESTED_FILE=$TIER NESTED_QUEUE_DEPTH=32; }
for r in 1 2; do
  run rq3 rabitq 3 128 X=1
  ssd nq512 tieredq-nested 512 32,64,128 NESTED_BITS=4 RERANK=100
  run rq4 rabitq 4 128 X=1
  ssd nq768 tieredq-nested 768 32,64 NESTED_BITS=4 RERANK=50
  run rq5 rabitq 5 64,128 X=1
  ssd nq1024 tieredq-nested 1024 128 NESTED_BITS=4 RERANK=50
  run rq6 rabitq 6 64 X=1
done
rm -f $TIER
echo done > time.status
