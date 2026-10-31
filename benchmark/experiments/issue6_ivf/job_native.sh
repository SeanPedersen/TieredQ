#!/usr/bin/env bash
# Native RaBitQ IVF at 5/6/7 bits, no rerank (untimed recall), CPUs 24-31.
set -euo pipefail
cd "$(dirname "$0")"
for bits in 5 6 7; do
  BENCH_SEED=0 taskset -c 24-31 bin/ivf_eval_rescore rabitq $bits "$DATA" 1536 64 0 8 16 32 64 2>>"$OUT/stderr.log" |
    grep -P "^rabitq\t" | sed "s/^/native\t/" >>"$OUT/native-0-$bits.tsv"
done
