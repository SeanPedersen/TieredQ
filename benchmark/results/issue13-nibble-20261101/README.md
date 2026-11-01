# Issue #13: additive nibble scoring for tiers above 4 bits (2026-11-01)

Fast-target models with tiers above 4 bits now use additive codebooks (`src/tieredq_additive_books.h`) and score every
code through the nibble byte-table kernel. `exact_fast_scores=True` (C++: `target_mode=full`) keeps Lloyd-Max books and
bit-exact FP32 scoring. Existing Lloyd-Max model files keep their layout.

Hartmut (i9-7980XE, OpenBLAS). Timing: alone on CPUs 0-7, 6 interleaved rounds, median, seed 0. Recall: seeds 0/1/2,
paired with the exact arm. Sections: 32 groups per byte-table scale; 4 when a model has 7-8 bit tiers.

## Speed, exact -> nibble (QPS 1 thread / 8 threads)
| Set | bits/dim | nprobe | Exact | Nibble | Gain |
|---|---:|---:|---:|---:|---:|
| OpenAI-100K | 2 | 16 | 592 / 4,664 | 962 / 8,016 | 1.62x / 1.72x |
| OpenAI-100K | 2 | 64 | 178 / 1,411 | 284 / 2,316 | 1.60x / 1.64x |
| OpenAI-100K | 4 | 16 | 278 / 1,986 | 404 / 3,063 | 1.45x / 1.54x |
| OpenAI-100K | 4 | 64 | 92 / 634 | 114 / 858 | 1.24x / 1.35x |
| OpenAI-1M | 2 | 32 | 126 / 918 | 188 / 1,198 | 1.49x / 1.31x |
| OpenAI-1M | 2 | 128 | 35 / 258 | 51 / 327 | 1.46x / 1.27x |
| OpenAI-1M | 4 | 32 | 67 / 456 | 85 / 557 | 1.27x / 1.22x |
| OpenAI-1M | 4 | 128 | 21 / 140 | 23 / 150 | 1.10x / 1.07x |

Flat OpenAI-100K single-query p50 (`flat-p50.tsv`, exact includes range pruning): 2 bits/dim 9.09 -> 3.91 ms (R@10
0.9537 -> 0.9542), 4 bits/dim 13.98 -> 7.92 ms (R@10 0.9885 -> 0.9858). One seed.

## Recall@10 change, points (mean of 3 paired seeds; `final-*` vs `recall2-*` base)
| Set | 2 bits/dim | 4 bits/dim |
|---|---:|---:|
| OpenAI-100K | -0.03, -0.03 | -0.22, -0.27 |
| OpenAI-1M | -0.19, -0.21 | -0.12, -0.17 |
| MS MARCO-1M real queries | -0.15, -0.21 | -0.13, -0.15 |
| MS MARCO-1M document queries | -0.33, -0.29 | -0.37, -0.39 |

Each cell gives the low then high nprobe. Measured disk is within 8 bits per vector of the exact arm.

## Rejected
- `MAX_BITS=5/6` (`time-*`, `recall2-*` mb5/mb6): lower recall, and slower on OpenAI-1M at every cell.
- Scale sections closing on expected range: no better than uniform 32 groups (-0.47 to -0.59 pt at 4 bits/dim).
- Clipping table scales to c * RMS: -0.7 to -10 pt.
- Nested (sorted-code) additive books: no better than uniform at 7-8 bits (`mse.tsv`).

`recall2-*` is the exact arm plus additive books under exact FP32 scoring (`additive*`) and the caps. Before these runs
`seed` in `ivf_eval.cpp` was constant, so earlier TieredQ "seed" rows from that eval repeat one model.
