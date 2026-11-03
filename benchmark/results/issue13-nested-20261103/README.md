# Issue #13 option 2: flat nested search against RaBitQ at 4 bits/dim recall (2026-11-03)

Option 2 of the single-query profile (`issue13-gap4-20261103`): scan a narrow TieredQ prefix for every document, then
rescore the top k' candidates with prefix plus a `TieredQRefinement` suffix. Measured through the existing engine flow,
`TieredQNestedIVFIndex` (#23), with one list and nprobe 1, i.e. a flat scan. The suffix-only tier is in RAM, and the
prefix rows are decoded from the list. RaBitQ runs in the same eval (`rabitq` mode, one list) on the same data.

`issue6_ivf_eval` built from f3f7db6 (`SOURCE_REVISION`), Release, OpenBLAS. Notation: `t3p512` = 512-bit prefix, 3
bits/dim total (prefix + suffix). Stored bits = the prefix IVF's serialized bits per vector plus the suffix tier's.
- **Recall sweeps:** seed 0, CPUs 8-15 (`sweep.tsv`, `sweep-1m.tsv`).
- **100K seeds 1-2:** `seeds.tsv`.
- **Timing:** `job_time.sh` and `job_1m.sh`, 2 interleaved rounds, median, alone on CPUs 0-7. `summary.py` prints
  `tables.md`.

## OpenAI-100K (`time.tsv`)
| Arm | k' | Stored bits (vs RaBitQ) | R@10 seed 0 | QPS 1t·b1 (vs RaBitQ) | QPS 8t·b32 (vs RaBitQ) |
|---|---:|---|---:|---|---|
| RaBitQ 4-bit | - | 6,337 | 0.9664 | 574 | 4,235 |
| t3p512 | 50 | 4,995 (0.79x) | 0.9671 | 1,251 (2.18x) | 9,746 (2.30x) |
| t3p768 | 30 | 4,862 (0.77x) | 0.9680 | 995 (1.73x) | 8,097 (1.91x) |
| t3p1024 | 20 | 4,988 (0.79x) | 0.9669 | 830 (1.45x) | 7,304 (1.72x) |
| t3p1536 | 20 | 4,839 (0.76x) | 0.9734 | 565 (0.99x) | 4,920 (1.16x) |
| RaBitQ 2-bit | - | 3,265 | 0.9041 | 592 | 4,358 |
| t2p1536 | 20 | 3,271 (1.00x) | 0.9465 | 558 (0.94x) | 5,010 (1.15x) |

Mean R@10 over seeds 0-2: RaBitQ 4-bit 0.9674; t3p512 k'=50 0.9686; t3p768 k'=30 0.9677; t3p1024 k'=20 0.9670.

## OpenAI-1M (`time-1m.tsv`)
| Arm | k' | Stored bits (vs RaBitQ) | R@10 seed 0 | QPS 1t·b1 (vs RaBitQ) | QPS 8t·b32 (vs RaBitQ) |
|---|---:|---|---:|---|---|
| RaBitQ 4-bit | - | 6,336 | 0.9706 | 49 | 370 |
| t3p512 | 100 | 4,805 (0.76x) | 0.9716 | 142 (2.90x) | 1,106 (2.99x) |
| t3p768 | 100 | 4,641 (0.73x) | 0.9716 | 99 (2.02x) | 798 (2.16x) |
| t3p1024 | 30 | 4,747 (0.75x) | 0.9725 | 82 (1.66x) | 636 (1.72x) |

k' is the smallest of 30 / 50 / 100 whose seed-0 recall reaches RaBitQ's.

## Findings
- At RaBitQ's 4-bit recall, flat nested search is 2.2x (100K) and 2.9x (1M) RaBitQ's single-query QPS and 2.3x / 3.0x
  its batched QPS, with 21-24% less stored memory. The plain flat default is at 0.54x single-query on the same task.
- The gain grows with the collection: the first pass reads ~77 bytes per document (512-bit prefix) instead of 469.
- At RaBitQ's 2-bit memory the 1536-bit prefix with 2 bits/dim total gains +4.2 pt R@10 at 0.94x / 1.15x its QPS.
- The 1M recall margins are seed 0 only (+0.1 to +0.2 pt). The eval's IVF with one list stands in for a flat nested
  index, which the library does not have yet.
