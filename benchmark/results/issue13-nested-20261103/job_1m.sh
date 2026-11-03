#!/bin/bash
# Issue #13 option 2 on OpenAI-1M: recall sweep (seed 0, CPUs 8-15), then RaBitQ 4-bit and the nested configs at its
# recall timed alone on CPUs 0-7, 2 interleaved rounds. Flat: one list, nprobe 1, suffix-only RAM tier.
N=$HOME/experiments/TieredQ-issue13-nested-20261103; D=$HOME/experiments/TieredQ-issue4-1m-20261030/abdata/openai1m
cd $N; S=sweep-1m.tsv; T=time-1m.tsv; : > $S; : > $T
probe() { local cpus=$1 timed=$2 out=$3 label=$4 mode=$5 budget=$6; shift 6
  env "$@" BENCH_SEED=0 LISTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c $cpus build/issue6_ivf_eval $mode $budget $D 1536 256 $timed 1 \
    2>/dev/null | grep -P '^(tieredq-nested|rabitq)\t' | sed "s/^/$label\t/" >> $out; }
echo sweep > status-1m
probe 8-15 0 $S rabitq4 rabitq 4 X=1
for p in 512 768 1024; do probe 8-15 0 $S t3p$p tieredq-nested $p NESTED_BITS=3 NESTED_ROWS=suffix RERANK=30,50,100; done
echo timing > status-1m
target=$(awk -F'\t' '$2=="rabitq"{print $8}' $S)
for r in 1 2; do
  probe 0-7 1 $T rabitq4 rabitq 4 X=1
  for p in 512 768 1024; do
    k=$(awk -F'\t' -v p=t3p$p -v t=$target '$1==p && $9>=t {print $8; exit}' $S)
    [ -n "$k" ] && probe 0-7 1 $T t3p$p tieredq-nested $p NESTED_BITS=3 NESTED_ROWS=suffix RERANK=$k
  done
done
echo done > status-1m
