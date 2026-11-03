#!/bin/bash
# Same-budget A/B of daf7184 (before nibble scoring) vs HEAD, one tree per run, interleaved, 3 rounds, alone on CPUs 0-7.
N=$HOME/experiments/TieredQ-issue13-refs-20261102; A=$N/ab
P=$HOME/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
F=$HOME/experiments/TieredQ-issue4-equal-memory-after-20261028/data; O=$HOME/experiments/TieredQ-issue4-1m-20261030/data
(cd $A/base/benchmark && taskset -c 16-31 $P -c "from native_build import build_native; build_native()") > $A/build.log 2>&1 || { echo build-failed > $A/status; exit 1; }
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
for r in 1 2 3; do for c in "sift $F 346 128" "sift $F 354 128" "glove $F 165 200" "openai $F 1257 1536" "openai $F 5689 1536" "gemini $F 2972 768" "openai1m $O 6173 1536"; do read ds data code d <<< "$c"
  for arm in base head; do [ $arm = base ] && B=$A/base/benchmark || B=$N/source/benchmark
    (cd $B && PYTHONPATH=$B/build-native:$B taskset -c 0-7 $P -u run.py --codec tieredq --datasets $ds --bits $(((code+d-1)/d)) --code-bits $code --seeds 0 --data $data --output $A/out --cell $A/cell-$arm-$r-$ds-$code.json >> $A/run.log 2>&1) || echo "$arm $r $ds $code" >> $A/failed.txt
  done
done; done; echo done > $A/status
