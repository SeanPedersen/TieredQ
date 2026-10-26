set -eu
cd /home/sean/experiments/TieredQ-serving-allocation-20261026-XQJU1q
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
python=/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/.venv/bin/python
data=/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data
previous=/home/sean/experiments/TieredQ-tiny-budgets-20261026-9NAZyQ/results
ps -eo pid,psr,pcpu,args --sort=-pcpu | head -12 > results/isolated-host-before.txt
for variant in register byte; do
  native=build
  if [ "$variant" = byte ]; then native=build-wide; fi
  echo "Final $variant comparisons pinned to physical cores 10-17"
  taskset -c 10-17 "$python" benchmark/experiments/serving_allocation/wide_sift.py --data "$data" --native "$native" --previous "$previous" --results results --variant "$variant" --controlled > "wide-isolated-$variant.log" 2>&1
  taskset -c 10-17 "$python" benchmark/experiments/serving_allocation/cross_dataset.py --data "$data" --native "$native" --results results --mode timing --variant "$variant" --controlled > "cross-isolated-$variant.log" 2>&1
  tail -1 "cross-isolated-$variant.log"
done
taskset -c 10-17 "$python" benchmark/experiments/serving_allocation/costs.py --data "$data" --native build --previous "$previous" --results results --output-name costs-isolated.jsonl > costs-isolated.log 2>&1
taskset -c 10-17 "$python" benchmark/experiments/serving_allocation/validate.py --native build --previous "$previous" --results results > validation-final.log 2>&1
"$python" benchmark/experiments/serving_allocation/analyze.py results
ps -eo pid,psr,pcpu,args --sort=-pcpu | head -12 > results/isolated-host-after.txt
printf '0\n' > results/isolated-exit-code
echo 'Final isolated comparison complete'
