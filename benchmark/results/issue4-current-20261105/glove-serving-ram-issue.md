## Observed regression

The ongoing current-main refresh for #4 found a large GloVe-200 100K accuracy loss against Qdrant TQ+ at its 1-bit reference **when matching measured serving RAM**, especially peak RAM. Investigate this as a possible bug/regression or harness effect; the cause is not established.

Library: `c74b4e354b92d848a004e0f50199c3f7db7e0ca4`. TieredQ uses the default flat fast index with `fine_tiers=True`; verify whether these low-budget models actually engage fine tiers. Dataset: 100,000 base vectors, 200 dimensions, 1,000 cosine queries. Recall is mean ± sample std across seeds 0–2.

### Confirmed measurements

Resource values below are bytes: maximum across TieredQ seeds, minimum across reference seeds. RAM is measured RSS growth over a pre-reload baseline after preloading codec libraries, rather than component accounting or nominal code bits. Serialized sizes include model state, metadata and padding.

| Configuration | Requested TieredQ code bits/vector | Recall@10 | Serialized bytes | Steady RSS growth | Peak RSS growth |
|---|---:|---:|---:|---:|---:|
| Qdrant TQ+ 1-bit reference | — | 0.3553 ± 0.0000 | 3,304,457 | 3,747,840 | 4,190,208 |
| TieredQ steady-RAM match | 152 | 0.3403 ± 0.0061 | 2,671,444 | 3,698,688 | 5,328,896 |
| TieredQ peak-RAM match | 62 | 0.1897 ± 0.0019 | 1,534,484 | 2,519,040 | 4,136,960 |

The steady-RAM point exceeds the peak budget. To fit the peak budget, the matcher cuts the requested TieredQ code budget to 62 bits/vector, and recall falls well below TQ+.

### Per-phase clue, seed 0

| Configuration | Post-reload RSS growth | Reload peak growth | Serving peak growth | Serving peak minus post-reload RSS |
|---|---:|---:|---:|---:|
| TieredQ, 62 requested code bits | 2,519,040 | 2,871,296 | 4,136,960 | 1,617,920 |
| TieredQ, 152 requested code bits | 3,698,688 | 3,973,120 | 5,328,896 | 1,630,208 |
| Fresh TQ+ 1-bit check | 3,588,096 | 3,588,096 | 4,108,288 | 520,192 |

The large extra RSS appears during serving, not just reload. These figures do not yet distinguish live scratch buffers, retained allocator/thread-local memory, faulted code pages, projection/LUT state, unintended allocations or measurement effects.

Reference targets reuse unchanged seed-0–2 reference cells from `benchmark/results/issue9-ship-20261030/match-a/cells.csv`, as agreed for #4. The fresh seed-0 reference check reproduces recall and serialized bytes; RSS differs somewhat, as shown above. Audit baseline/allocator effects before concluding that the library alone is responsible.

### Measurement conditions

- Hartmut, Intel i9-7980XE; native adapter built with `-O3 -march=native -ffp-contract=off`.
- Each memory cell builds in a fresh process and reloads/serves in another fresh process.
- Peak covers reload and serving at threads 1/8 × batches 1/32, with one untimed pass of 1,000 queries per schedule (`--quick`).
- TieredQ allocation is selected by memory only, then confirmed on all three seeds. No FP32 reranking.
- Cached GloVe base, queries and exact truth were recovered byte-identically to the archived #4 array hashes.
- Qdrant revision: `222b2e44ab1c3f17e8246396ead6558e45041df7`; the existing production TQ+ shim is preserved.

### Reproduction on Hartmut

Run after the current #4 benchmark finishes, to avoid competing work. The measured snapshot, validated native adapter and prepared inputs are retained here:

```bash
W=/home/sean/experiments/TieredQ-issue4-current-20261105
P=/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python
cd "$W/source/benchmark"
export PYTHONPATH="$PWD/build-native:$PWD" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
taskset -c 8-15 "$P" -u - <<'PY'
from pathlib import Path
from match_memory import Cells
from measure_memory import prepare_native
w = Path('/home/sean/experiments/TieredQ-issue4-current-20261105')
out = w / 'glove-ram-investigation-repro'
out.mkdir(exist_ok=False)
env, metadata = prepare_native()
cells = Cells(out, env, w / 'data', True)
for seed in (0, 1, 2):
    cells.measure('glove', 'qdrant', 1, seed)
for code in (62, 152):
    for seed in (0, 1, 2):
        cells.measure('glove', 'tieredq', 1, seed, code)
PY
```

Existing evidence in that workspace:

- `match-b1/cells.csv` and `match-b1/calibration.csv` (Qdrant confirmation rows).
- `match-b1/cell-glove-tieredq-c{62,152}-{0,1,2}/{build,serve}.json`.
- `reference-build-b1/cell-glove-qdrant-1-0/{build,serve}.json`.
- `glove-1bit-memory-breakdown.json`, `recovered-data-provenance.json`, `reference-audit.json` and `frozen-native.json`.

The complete `benchmark/results/issue4-current-20261105/` bundle is being prepared with #4 and has not been pushed yet.

### Investigation goals

1. Reproduce the per-phase RSS increase and verify the matcher/header/baseline accounting.
2. Profile serving allocations and retained memory by schedule and thread count; compare accounted bytes with RSS and absolute process totals.
3. Check projection/LUT caches, temporary query buffers, thread-local state, allocator retention and unexpected fallback paths; establish whether this is a regression against an earlier revision.
4. If a bug or avoidable overhead is found, fix it and rerun the affected GloVe cells. Preserve the original #4 measurements as provenance rather than silently replacing them.

This is a follow-up investigation; the current #4 benchmark remains pinned to the measured revision and should report the observed loss plainly.
