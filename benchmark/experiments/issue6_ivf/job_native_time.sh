#!/usr/bin/env bash
# Native RaBitQ IVF at BITS (1-bit scan plus ex-bit refinement inside its IVF, no rerank), timed alone on CPUs 0-7:
# the best-case speed of RaBitQ's compressed codes next to the rabitq:B rescore rows. Rows go to OUT/native-<lists>-<bits>.tsv.
# DATA / OUT / NLIST / NPROBES / BITS / DIM from the environment; LISTS (coarse lists) is passed through.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?}" "${OUT:?}" "${NLIST:?}" "${NPROBES:?}" "${BITS:=5 6 7}" "${BIN:=bin/ivf_eval_refinement5}" "${DIM:=1536}"
mkdir -p "$OUT"
for bits in $BITS; do
  BENCH_SEED=0 OPENBLAS_NUM_THREADS=1 taskset -c 0-7 "$BIN" rabitq "$bits" "$DATA" "$DIM" "$NLIST" 1 ${NPROBES//,/ } \
    2>>"$OUT/stderr.log" | grep -P "^rabitq\t" | sed "s/^/native\t/" >>"$OUT/native-${LISTS:-$NLIST}-$bits.tsv"
done
