#!/bin/sh
# #45: production assignment bound (assignment_bound_dimensions 224, the automatic default from 8,192 lists) vs the
# same training with the bound off. (1) TieredQIVFIndex on MS MARCO 1M x 1024 (2 bits/dim, router off, 8 threads on
# cores 0-7), 8,192 and 32,768 lists, seeds 0 and 1, each pair in one process. (2) TieredQNestedIVFIndex #52 defaults on
# Qwen/MS MARCO 8.84M x 1024, 64K lists, 32 build threads, tier on NVMe, 750 queries on pinned CPU 2, order ABBA.
# Takes the benchmark lock; resumable per step.
set -e
E=/home/sean/experiments/TieredQ-issue45-bound-20261119
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
O=/home/sean/experiments/TieredQ-issue45-emeans-ivfflat-20261118
SSD=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/ssd
export UV_CACHE_DIR=/home/sean/experiments/uv-cache
PY="uv run --project $E/source --no-dev --with threadpoolctl python"
OFF={\"iterations\":4,\"assignment_supers\":256,\"assignment_probes\":16,\"sample_rows_per_list\":256,\"emeans_epochs\":2.0,\"emeans_probes\":4,\"assignment_bound_dimensions\":0}
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
cd $E/knn
for seed in 0 1; do
  for spec in 8192:16,32,64,128,256 32768:32,64,128,256,512; do
    lists=${spec%%:*}; probes=${spec#*:}; mark=$E/results/ivf1m-$lists-seed$seed.done
    [ -f $mark ] && continue
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 $PY ivf_lists_check.py $O/data/msmarco1m.npy \
      $O/results/msmarco1m-queries/query_truth.npz $E/results/ivf1m.jsonl \
      --queries $Q/data/calibration_queries.fvecs,$Q/data/test_queries.fvecs --lists $lists --probes $probes --seed $seed \
      --only auto-nobound,auto >> $E/results/ivf1m.log 2>&1
    touch $mark
  done
done
NPROBES=96,128,192,256,512,1024
for spec in off:0 on:0 on:7 off:7; do
  mode=${spec%:*}; seed=${spec#*:}; name=bound-$mode-seed$seed; out=$E/results/$name; index=$SSD/$name.tq; tier=$SSD/$name.rows
  mkdir -p $out/build $out/serve
  training=""; [ $mode = off ] && training=$OFF
  [ -f $out/build/build.json ] || OMP_NUM_THREADS=32 $PY $E/build.py $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 \
    $index $tier $out/build/build.json $seed $training > $out/build/run.log 2>&1
  [ -f $out/serve/run.csv ] || OMP_NUM_THREADS=1 taskset -c 2 $PY $E/serve.py $index $tier $Q/data/test_queries.fvecs \
    $Q/data/test_truth.ivecs 750 $NPROBES $out/serve/run.csv > $out/serve/run.log 2>&1
  rm -f $index $tier
  echo "done $name"
done
echo ALLDONE
