set -eu
cd /home/sean/experiments/TieredQ-serving-allocation-20261026-XQJU1q
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
python=/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/.venv/bin/python
data=/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data
previous=/home/sean/experiments/TieredQ-tiny-budgets-20261026-9NAZyQ/results
for variant in register byte; do
  native=build
  if [ "$variant" = byte ]; then native=build-wide; fi
  echo "Checking and timing $variant whole-coordinate tables on SIFT, GloVe and OpenAI"
  taskset -c 0-7 "$python" benchmark/experiments/serving_allocation/wide_sift.py --data "$data" --native "$native" --previous "$previous" --results results --variant "$variant" > "wide-$variant.log" 2>&1
  tail -1 "wide-$variant.log"
  taskset -c 0-7 "$python" benchmark/experiments/serving_allocation/cross_dataset.py --data "$data" --native "$native" --results results --mode timing --variant "$variant" > "cross-$variant.log" 2>&1
  tail -1 "cross-$variant.log"
done
echo 'Calibrating serving costs in shuffled order, without overlapping experiment jobs'
taskset -c 0-7 "$python" benchmark/experiments/serving_allocation/costs.py --data "$data" --native build --previous "$previous" --results results --output-name costs-controlled.jsonl > costs-controlled.log 2>&1
echo 'Recording separate diagnostic phases'
taskset -c 0-7 "$python" benchmark/experiments/serving_allocation/diagnose.py --data "$data" --native build-profile --previous "$previous" --results results > phases.log 2>&1
"$python" benchmark/experiments/serving_allocation/analyze.py results
printf '0\n' > results/finish-exit-code
echo 'Study validation and measurement complete'
