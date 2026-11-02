cd $(dirname $0); export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
arm() { n=$1; cpus=$2; shift 2; env "$@" taskset -c $cpus ~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python -u arm.py $n sift 253 354 573 >> out.tsv 2>>err.log; }
arm nibble 8-15 X=1 &
arm additive-exact 16-23 TQ_NO_LOOKUP=1 &
arm lloyd-exact 24-31 TQ_NO_LOOKUP=1 TQ_LLOYD=1 &
wait
arm sections4 8-15 TQ_SECTIONS=4 &
arm sections2 16-23 TQ_SECTIONS=2 &
arm nosort 24-31 TQ_NO_SORT=1 &
wait
echo done > status
