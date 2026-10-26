# Issue #12: exact packed lookup for every tier wider than four bits

Archived bytes/vector values use the earlier memory counters; see the
[final-review accounting correction](../../../../docs/high-bit-recall.md#final-review-accounting-correction).
Use the [refreshed current storage table](../storage-refresh/README.md) for the eleven reported configurations.

The threshold is now 5 rather than 7. The user approved the switch after reviewing the throughput losses. New 5/6-bit models use payload kind 4 and scalar-codec FP32 scores; existing kind 2 numerical-level files retain their representation through load, search, resave and append. Kind 3 byte-code files still repack to kind 4.

## Same-session A/B

Full datasets, 1,000 held-out queries, top 10, seed 0/1/2 mean recall; seed 0 QPS. Both paths were run in the same session on hartmut (AVX-512), pinned to CPUs 0–7. Batch 32 uses 8 threads; batch 1 includes 1 and 8 threads. The driver runs new then baseline for each batch setting, rather than alternating individual repetitions. RaBitQ control timings vary by up to about 12%; the large losses below exceed that noise.

| Point | Bytes/vector old → packed | Recall old → packed | QPS batch 32, 8 threads | QPS batch 1, 1 thread | QPS batch 1, 8 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| glove 5-bit | 212.8 → 140.8 | 0.94507 → 0.94820 | 12,983 → 8,799 (-32%) | 1,000 → 526 (-47%) | 4,535 → 2,651 (-42%) |
| glove 6-bit | 212.8 → 164.8 | 0.96527 → 0.97320 | 12,277 → 7,911 (-36%) | 942 → 489 (-48%) | 4,001 → 2,586 (-35%) |
| openai 2-bit | 1489.8 → 493.8 | 0.93837 → 0.94683 | 2,588 → 1,213 (-53%) | 99 → 77 (-22%) | 392 → 342 (-13%) |
| openai 3-bit | 1651.5 → 695.5 | 0.95870 → 0.97320 | 2,334 → 1,085 (-54%) | 90 → 67 (-25%) | 342 → 297 (-13%) |
| sift 2-bit | 101.3 → 41.3 | 0.76010 → 0.76390 | 2,315 → 2,368 (+2%) | 149 → 81 (-46%) | 827 → 563 (-32%) |

These five points supersede the corresponding rows of the historical [packed-storage curve](../pack-opt/README.md). Other budget points were not rerun for this policy change. The switch removes the one-byte-per-coordinate expansion at these points; narrow lookup padding can still produce storage plateaus.

Raw measurements: `new/`, `baseline/`, `new-single/`, `baseline-single/`; aggregated by `summarize.py` into `summary.json`. `run.sh` and `run.log` record the actual sequence and validation. The original pruning estimate is retained as `prune-study-initial.*`; it used float64 reductions and is diagnostic only.

## Validation

Local ARM: `cmake --build build && ./build/test_tieredq` passes; `uv run --python 3.12 --reinstall-package tieredq pytest -q` passes all 44 tests. C++ tests cover exact 5/6-bit scores and legacy kind 2 persistence/append; Python tests check packed kind 4 and storage at mid-width learned tiers. The A/B log contains remote x86 benchmark validation.

## Tier-prefix pruning feasibility

See [the pruning study](pruning.md) for every tier prefix at 4/8 bits. The study is Python only; no C++ pruning kernel was added. Optional encoder-release and AVX2 changes are deferred at the user’s request.
