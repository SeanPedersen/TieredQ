#!/usr/bin/env bash
# Issue #6 sweep on Hartmut: zvec IVF_RABITQ at total_bits 1-7, TieredQ at matched index sizes, one pinned core.
# zvec's index is ~768B+420 bits/vector and TieredQ's ~768b+300, so b = B + 0.16 matches sizes within ~1%.
set -euo pipefail
cd "$(dirname "$0")/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { taskset -c 17 uv run python ../ivf_vs_zvec.py --data ../data --nprobe 4 8 16 32 64 128 256 "$@"; }
run --codec zvec --bits 1 2 3 4 5 6 7 --workdir ../work --out ../zvec.tsv 2> ../zvec.err
run --codec tieredq --bits 1.16 2.16 3.16 4.16 5.16 6.16 7.16 --out ../tieredq.tsv
run --codec tieredq --confidence 8 --bits 1.16 2.16 3.16 4.16 5.16 6.16 7.16 --out ../tieredq.tsv
echo __DONE__
