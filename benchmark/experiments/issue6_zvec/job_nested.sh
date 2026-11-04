#!/usr/bin/env bash
# Issue #6 follow-up on Hartmut: TieredQNestedIVFIndex (prefix IVF + in-RAM suffix tier) at zvec's memory levels.
set -euo pipefail
cd "$(dirname "$0")/env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { taskset -c 17 uv run python ../ivf_vs_zvec.py --data ../data --codec nested --bits 3 4 5 6 7 \
  --nprobe 4 8 16 32 64 128 256 --out ../nested.tsv "$@"; }
run --prefix-bits 1 --candidates 50
run --prefix-bits 1 --candidates 20
run --prefix-bits 2 --candidates 20
echo __DONE__
