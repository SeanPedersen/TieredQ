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
