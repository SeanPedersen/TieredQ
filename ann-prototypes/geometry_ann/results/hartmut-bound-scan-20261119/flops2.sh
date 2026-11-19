#!/bin/sh
# #58 step 2: step 1 (B) vs probe-range slot grouping + parallel reduction (C), ABBA, bound 224.
B=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops/build/tieredq_assign_flops_benchmark
C2=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops2/build/tieredq_assign_flops_benchmark
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
L=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops2.log
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for v in B C2 C2 B; do
  eval bin=\$$v; echo "== $v" >> $L
  OMP_NUM_THREADS=32 python3 -c 'import resource,subprocess,sys; subprocess.run(sys.argv[1:]); print("peak RSS", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "KB", flush=True)' $bin $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 224 2>&1 | grep -v '^peaks' >> $L
done
echo DONE >> $L
