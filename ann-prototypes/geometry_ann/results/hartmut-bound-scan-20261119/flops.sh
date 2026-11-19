#!/bin/sh
# #58 step 1: FLOP stages of the bounded assignment, 7761b5f (A) vs vectorized bound scan (B), ABBA, bound 224.
A=/home/sean/experiments/TieredQ-issue45-flops-20261119/source/build/tieredq_assign_flops_benchmark
B=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops/build/tieredq_assign_flops_benchmark
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
C=/home/sean/experiments/TieredQ-issue45-pruning-20261118/results/centroids-64k-seed0.f32
L=/home/sean/experiments/TieredQ-issue58-scan-20261119/flops.log
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for v in A B B A; do
  eval bin=\$$v; echo "== $v" >> $L
  OMP_NUM_THREADS=32 python3 -c 'import resource,subprocess,sys; subprocess.run(sys.argv[1:]); print("peak RSS", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "KB", flush=True)' $bin $Q/base.i8 $Q/scales.f32 $Q/offsets.f32 1024 $C 65536 224 >> $L 2>&1
done
echo DONE >> $L
