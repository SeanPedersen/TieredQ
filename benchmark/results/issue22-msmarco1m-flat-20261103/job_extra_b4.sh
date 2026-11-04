#!/bin/bash
# Follow-up to job_flat.sh: equal_recall.py bisects on seed-0 recall, and for RaBitQ 4-bit (real queries) its budget
# (2,362 code bits) reaches 0.9557 mean over seeds 0-2, below RaBitQ's 0.9571. Step up in RESOLUTION (8) code-bit steps
# until the seed mean reaches it (CPUs 8-31), record it in equal-recall/b4/equal-recall-msmarco1m-mean.csv, then time
# that budget alone on CPUs 0-7 through run.py.
set -u
W=/home/sean/experiments/TieredQ-issue22-flat-20261103
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
DATA=/home/sean/experiments/TieredQ-issue22-msmarco-20261031/data
B=$W/source/benchmark
export PYTHONPATH=$B/build-native:$B:$B/experiments/issue4_1m OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 BENCH_SEED=0
cd $W
taskset -c 8-31 $P -u - <<'PY' > equal-recall/run-b4-mean.log 2>&1 || { echo search > extra-exit-code; exit 1; }
import csv, statistics
import codec_native
from data import prepare
from equal_recall import FIELDS, RESOLUTION, SEEDS, THREADS, recall_at
START, TARGET, MAX_STEPS = 2362, 0.9571333333333334, 8
codec_native.set_threads(THREADS)
base, queries, truth = prepare(__import__("pathlib").Path("/home/sean/experiments/TieredQ-issue22-msmarco-20261031/data"), "msmarco1m")
for step in range(1, MAX_STEPS + 1):
    code_bits = START + RESOLUTION * step
    recalls = [recall_at(base, queries, truth, False, code_bits, seed) for seed in SEEDS]
    print(code_bits, recalls, flush=True)
    if statistics.mean(recalls) >= TARGET:
        break
with open("equal-recall/b4/equal-recall-msmarco1m-mean.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerow(dict(dataset="msmarco1m", reference_codec="rabitq", reference_bits=4, reference_recall10=TARGET,
                         code_bits=code_bits, recall10_seed0=recalls[0], recall10_mean=statistics.mean(recalls),
                         recall10_std=statistics.stdev(recalls), probes=step))
PY
code=$($P -c "import csv; print(next(csv.DictReader(open('equal-recall/b4/equal-recall-msmarco1m-mean.csv')))['code_bits'])")
export PYTHONPATH=$B/build-native
name=timing/cell-msmarco1m-tieredq-$(( (code + 1023) / 1024 ))-c$code.json
[ -f $name ] || (cd $B && taskset -c 0-7 $P -u run.py --codec tieredq --datasets msmarco1m --bits $(( (code + 1023) / 1024 )) \
  --code-bits $code --seeds 0 --data $DATA --output $W/timing --cell $W/$name >> $W/timing/run-extra.log 2>&1) ||
  { echo timing > extra-exit-code; exit 1; }
echo 0 > extra-exit-code
