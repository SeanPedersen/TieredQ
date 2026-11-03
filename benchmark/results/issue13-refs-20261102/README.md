# Issue #13: flat TieredQ on c0910d6 against the reused #4 reference cells (2026-11-02)

TieredQ only, through `run.py` alone on CPUs 0-7 (seed 0, one pass per cell, as the reference cells); budgets re-bisected
with `equal_recall.py`. References are not re-timed. `tables.md` is `report_tables.py` over these cells. `prof.log`
profiles the IVF single-query loop; the kernels inline into `search_rows`, so it does not split scan cost by tier.

## Equal recall vs RaBitQ, before -> after (batched QPS ratio 8t.b32; p50 ratio 1t.b1)
| Set | bits/dim | Code bits | Batched | p50 |
|---|---:|---:|---:|---:|
| OpenAI-100K | 2 | 1598 -> 1538 | 1.05x -> 1.29x | 1.02x -> 0.82x |
| OpenAI-100K | 4 | 3554 -> 3764 | 0.41x -> 0.38x | 6.23x -> 2.71x |
| Gemini | 2 | 860 -> 860 | 1.19x -> 1.21x | 0.84x -> 0.87x |
| Gemini | 4 | 1856 -> 1844 | 0.43x -> 0.55x | 5.35x -> 1.69x |
| OpenAI-1M | 2 | 1814 -> 1820 | 1.09x -> 1.08x | 0.98x -> 1.00x |
| OpenAI-1M | 4 | 3968 -> 4034 | 0.47x -> 0.38x | 3.82x -> 2.66x |
| SIFT1M | 2 | 146 -> 154 | 1.49x -> 1.33x | 2.00x -> 0.70x |
| SIFT1M | 4 | 346 -> 354 | 1.17x -> 0.92x | 2.65x -> 1.25x |

## Regressions at equal budget (old cell -> this cell, seed 0)
- Batched QPS with 7-8 bit tiers: OpenAI-1M 6173 bits 157 -> 99 (8t.b32), 22 -> 13 (1t.b32); Gemini 2972 bits 2818 -> 1953.
  Likely cause, not verified: nibble models scan with query tile 1 (`scan_fast_avx512_batch`), the code path grouped queries.
- SIFT1M recall, L2: 253 bits 0.7585 -> 0.7398, 573 bits 0.9553 -> 0.9452. Not in the #13 recall validation, which was IP only.

## Update: final code 9e7bb30 (`final/`)
Nibble models now share code blocks across a query group, and L2 models keep exact scoring. The equal-budget notes above
describe c0910d6. `final/tables.md` is authoritative for the equal-recall rows; its other sections mix older cells.

| Set vs RaBitQ | bits/dim | Batched, c0910d6 parent -> final | p50, parent -> final |
|---|---:|---:|---:|
| OpenAI-100K | 2 | 1.05x -> 1.60x | 1.02x -> 0.89x |
| OpenAI-100K | 4 | 0.41x -> 0.57x | 6.23x -> 2.69x |
| Gemini | 2 | 1.19x -> 1.45x | 0.84x -> 0.86x |
| Gemini | 4 | 0.43x -> 0.80x | 5.35x -> 1.74x |
| OpenAI-1M | 2 | 1.09x -> 1.93x | 0.98x -> 1.01x |
| OpenAI-1M | 4 | 0.47x -> 0.59x | 3.82x -> 2.67x |
| GloVe | 4 | 0.96x -> 1.11x | 1.01x -> 0.99x |
| SIFT1M (exact) | 4 | 1.17x -> 1.28x | 2.65x -> 3.40x |

The SIFT1M recall figures above compare with cells from an older tree and overstate the loss. `diag-sift/out.tsv`, same
tree, three seeds, nibble vs Lloyd-Max exact: -0.72 / -0.24 / -0.52 pt at 253 / 354 / 573 bits.
The SIFT1M p50 rise on the unchanged exact path is unexplained. One timing pass per cell, as the reference cells.

## Same-budget A/B: daf7184 vs current code (`ab-daf7184/`)
Interleaved, 3 rounds, alone on CPUs 0-7, median. Supersedes the same-budget notes above, whose baselines come from older
trees. SIFT1M p50 is unchanged here: the 11.33 -> 14.56 ms rise above was a stale old cell.

Models with 7-8 bit tiers (QPS ratio, new / old):
| Cell | 8t.b32 | 1t.b32 | 8t.b1 | 1t.b1 | R@10, seed 0 |
|---|---:|---:|---:|---:|---:|
| Gemini 2972 bits | 0.96x | 0.83x | 1.86x | 1.73x | 0.9794 -> 0.9801 |
| OpenAI-100K 5689 bits | 0.97x | 0.89x | 1.67x | 1.96x | 0.9850 -> 0.9841 |
| OpenAI-1M 6173 bits | 0.96x | 0.88x | 1.61x | 1.36x | 0.9897 -> 0.9859 |

Other cells (SIFT1M L2, GloVe 165 bits, OpenAI-100K 1257 bits) show no slowdown beyond their round-to-round spread.
Walking query groups over 64-document chunks instead of 256-document blocks is about 30-40% slower: rejected.

## Forced backends (`backend/`)
OpenAI-100K IVF, 64 lists, nprobe 16, seed 0, forced with a workspace-only hook on the AVX-512 host, so not a real AVX2 machine.
QPS 1 thread / 8 threads, exact -> nibble. Recall is identical across backends within an arm.
| Backend | 2 bits/dim | 4 bits/dim |
|---|---:|---:|
| AVX-512 | 590 / 4,657 -> 923 / 7,739 | 277 / 2,011 -> 397 / 3,069 |
| AVX2 | 32 / 248 -> 535 / 4,543 | 24 / 184 -> 168 / 1,285 |
| Scalar | 12 / 89 -> 29 / 219 | 9 / 68 -> 11 / 81 |

## IVF pruning vs nibble scoring (`pruning/`)
Exact (full-target) with TieredQPruning against the nibble default, which ignores pruning. 2 rounds, median, seed 0,
alone on CPUs 0-7. QPS 1 thread / 8 threads, R@10.
| Set, bits/dim, nprobe | exact, z=0 | exact, z=8 p=4 | exact, z=8 p=3 | nibble |
|---|---|---|---|---|
| 100K, 2, 16 | 585 / 4,602, 0.9381 | 625 / 4,874, 0.9381 | 910 / 7,160, 0.9367 | 949 / 7,972, 0.9369 |
| 100K, 2, 64 | 180 / 1,376, 0.9570 | 185 / 1,440, 0.9570 | 292 / 2,396, 0.9557 | 278 / 2,264, 0.9560 |
| 100K, 4, 16 | 276 / 1,996, 0.9631 | 423 / 3,240, 0.9630 | 588 / 4,436, 0.9607 | 406 / 3,137, 0.9623 |
| 100K, 4, 64 | 93 / 643, 0.9865 | 124 / 961, 0.9863 | 184 / 1,474, 0.9836 | 113 / 866, 0.9853 |
| 1M, 2, 32 | 125 / 916, 0.9355 | 126 / 930, 0.9342 | 207 / 1,478, 0.9251 | 186 / 1,204, 0.9333 |
| 1M, 2, 128 | 34 / 258, 0.9549 | 34 / 260, 0.9536 | 59 / 427, 0.9433 | 50 / 328, 0.9521 |
| 1M, 4, 32 | 67 / 456, 0.9644 | 90 / 652, 0.9584 | 132 / 980, 0.9402 | 84 / 558, 0.9623 |
| 1M, 4, 128 | 21 / 140, 0.9881 | 24 / 180, 0.9815 | 37 / 278, 0.9609 | 22 / 149, 0.9858 |

Gemini-768 99K, 64 lists (`pruning/prune-gemini.tsv`), same arms:
| bits/dim, nprobe | exact, z=0 | exact, z=8 p=4 | exact, z=8 p=3 | nibble |
|---|---|---|---|---|
| 2, 16 | 880 / 6,410, 0.9171 | 1,051 / 7,998, 0.9171 | 1,244 / 8,836, 0.9171 | 2,642 / 19,018, 0.9170 |
| 2, 64 | 288 / 2,079, 0.9311 | 316 / 2,494, 0.9311 | 446 / 3,199, 0.9311 | 743 / 5,810, 0.9316 |
| 4, 16 | 458 / 3,259, 0.9638 | 674 / 4,936, 0.9638 | 726 / 4,970, 0.9638 | 897 / 6,801, 0.9620 |
| 4, 64 | 130 / 1,003, 0.9819 | 202 / 1,540, 0.9819 | 254 / 1,747, 0.9819 | 242 / 1,952, 0.9797 |

## Range-checkpoint pruning on nibble tables, 08173a5 (`pruning-port/`)
`bin/ivf_eval_prune` built from 08173a5's `src/`, IVF with rerank 10, 2 interleaved rounds, median, seed 0, alone on CPUs
0-7. Arms: nibble-z0 is the new default (exact bound), nibble-off disables pruning (`PRUNE_PREFIX=64`), z8 sets
`PRUNE_Z=8`, exact arms force full-target scoring (`TQ_EXACT=1`, workspace-only hook). `summary.py port.tsv` prints the
table. QPS 1 thread / 8 threads, R@10.
| Set, bits/dim, nprobe | exact-z0 | exact-z8 | nibble-off | nibble-z0 | nibble-z8 |
|---|---|---|---|---|---|
| OpenAI-100K, 2, 16 | 592 / 4,620, 0.9381 | 615 / 4,868, 0.9381 | 979 / 8,322, 0.9369 | 1,150 / 9,268, 0.9369 | 1,227 / 9,977, 0.9369 |
| OpenAI-100K, 2, 64 | 176 / 1,382, 0.9570 | 180 / 1,408, 0.9570 | 281 / 2,300, 0.9560 | 377 / 3,161, 0.9560 | 396 / 3,268, 0.9560 |
| OpenAI-100K, 4, 16 | 272 / 1,932, 0.9631 | 422 / 3,220, 0.9630 | 400 / 3,060, 0.9623 | 400 / 3,014, 0.9623 | 604 / 4,668, 0.9623 |
| OpenAI-100K, 4, 64 | 92 / 646, 0.9865 | 124 / 939, 0.9863 | 117 / 855, 0.9853 | 114 / 860, 0.9853 | 204 / 1,641, 0.9853 |
| Gemini, 2, 16 | 870 / 6,624, 0.9171 | 1,037 / 7,886, 0.9171 | 2,492 / 18,302, 0.9170 | 2,455 / 18,127, 0.9170 | 2,704 / 18,876, 0.9170 |
| Gemini, 2, 64 | 274 / 2,048, 0.9311 | 317 / 2,498, 0.9311 | 704 / 5,492, 0.9316 | 838 / 6,088, 0.9316 | 865 / 6,192, 0.9316 |
| Gemini, 4, 16 | 453 / 3,290, 0.9638 | 659 / 4,790, 0.9638 | 896 / 6,864, 0.9620 | 866 / 6,734, 0.9620 | 974 / 7,253, 0.9620 |
| Gemini, 4, 64 | 130 / 975, 0.9819 | 198 / 1,520, 0.9819 | 246 / 1,944, 0.9797 | 243 / 1,908, 0.9797 | 323 / 2,418, 0.9797 |
| MS MARCO, 2, 32 | 178 / 1,296, 0.8955 | 180 / 1,257, 0.8955 | 316 / 2,078, 0.8939 | 336 / 2,260, 0.8939 | 334 / 2,251, 0.8939 |
| MS MARCO, 2, 128 | 48 / 348, 0.9396 | 47 / 350, 0.9396 | 79 / 516, 0.9371 | 88 / 588, 0.9371 | 88 / 578, 0.9371 |
| MS MARCO, 4, 32 | 95 / 688, 0.9244 | 98 / 714, 0.9244 | 137 / 914, 0.9236 | 158 / 1,008, 0.9236 | 176 / 1,148, 0.9236 |
| MS MARCO, 4, 128 | 26 / 180, 0.9768 | 26 / 184, 0.9768 | 36 / 240, 0.9755 | 46 / 282, 0.9755 | 49 / 311, 0.9755 |
| OpenAI-1M, 2, 32 | 126 / 902, 0.9355 | 128 / 950, 0.9342 | 180 / 1,133, 0.9333 | 237 / 1,551, 0.9333 | 243 / 1,592, 0.9333 |
| OpenAI-1M, 2, 128 | 35 / 258, 0.9549 | 35 / 257, 0.9536 | 50 / 320, 0.9521 | 69 / 444, 0.9521 | 68 / 442, 0.9521 |
| OpenAI-1M, 4, 32 | 65 / 448, 0.9644 | 88 / 647, 0.9584 | 85 / 554, 0.9623 | 86 / 554, 0.9623 | 150 / 981, 0.9612 |
| OpenAI-1M, 4, 128 | 21 / 137, 0.9881 | 24 / 178, 0.9815 | 22 / 150, 0.9858 | 24 / 150, 0.9858 | 42 / 282, 0.9847 |

- Default pruning (nibble-z0 vs nibble-off) leaves recall unchanged. At 2 bits/dim it adds 6-39% QPS on OpenAI and
  MS MARCO, 11-19% on Gemini nprobe 64, none on Gemini nprobe 16. At 4 bits/dim it adds -3 to +9% on OpenAI and Gemini
  (no gain) and 10-28% on MS MARCO.
- nibble-z8 costs 0-0.11 pt R@10 against nibble-z0 and is 1.5-1.9x faster at 4 bits/dim on OpenAI.
- Against exact-z8, nibble-z8 is 1.4-2.7x faster. On OpenAI-1M 4 bits/dim it also has higher recall
  (0.9612 vs 0.9584, 0.9847 vs 0.9815). On the other cells it is up to 0.25 pt lower.
- z8 is not the default: its bound is not exact.
