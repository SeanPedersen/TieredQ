# Issue #13: closing the 4 bits/dim gap to RaBitQ on OpenAI-1536 (2026-11-03)

Flat search, OpenAI-1536 100K and 1M, at RaBitQ's 4 bits/dim R@10. Models with 7-8 bit tiers scan with fine (4-group)
scale sections, about 8x the float conversions of the 32-group default. Variants that avoid them:

- `mb6`, `mb5`: `TieredQConfig::max_bits` 6 / 5 (workspace hook `TQ_MAX_BITS`, `hook.diff`).
- `f8`: 8-group fine sections (one-constant build, `hook.diff`).

`job.sh` bisects each variant's smallest budget reaching RaBitQ's mean R@10 (`equal_recall.py`, seed 0, CPUs 8-31). It
then times every variant at its own budget through `run.py`: 3 interleaved rounds, alone on CPUs 0-7, seed 0, tree
05ee4d3. `summary.py` prints the medians (`tables.md`). RaBitQ cells are the reference cells of
`issue13-refs-20261102/final` and were not re-timed. Stored bits are measured: max(saved file, serving RAM of the
reloaded index) per vector, model included.

| Set | Variant | Code bits | R@10 s0 / mean s0-2 | Stored bits TQ / RaBitQ | QPS 8t.b32 TQ / ref | QPS 1t.b32 TQ / ref | p50 1t.b1 ms TQ / ref |
|---|---|---:|---:|---|---|---|---|
| 100K | base | 3764 | 0.9681 / 0.9677 | 4,272 / 6,305 (0.68x) | 2,133 / 3,917 (0.54x) | 298 / 539 (0.55x) | 5.23 / 1.83 (2.86x) |
| 100K | mb6 | 3752 | 0.9683 / 0.9664 | 4,264 / 6,305 (0.68x) | 3,453 / 3,917 (0.88x) | 487 / 539 (0.90x) | 3.40 / 1.83 (1.86x) |
| 100K | mb5 | 4016 | 0.9687 / 0.9677 | 4,544 / 6,305 (0.72x) | 3,288 / 3,917 (0.84x) | 462 / 539 (0.86x) | 4.19 / 1.83 (2.29x) |
| 100K | f8 | 3800 | 0.9675 / 0.9684 | 4,312 / 6,305 (0.68x) | 2,746 / 3,917 (0.70x) | 368 / 539 (0.68x) | 4.46 / 1.83 (2.44x) |
| 1M | base | 4034 | 0.9715 / 0.9695 | 4,121 / 6,304 (0.65x) | 216 / 375 (0.57x) | 29 / 51 (0.57x) | 49.01 / 19.27 (2.54x) |
| 1M | mb6 | 4082 | 0.9705 / 0.9694 | 4,169 / 6,304 (0.66x) | 362 / 375 (0.97x) | 48 / 51 (0.93x) | 34.51 / 19.27 (1.79x) |
| 1M | mb5 | 4472 | 0.9706 / 0.9705 | 4,564 / 6,304 (0.72x) | 333 / 375 (0.89x) | 44 / 51 (0.85x) | 45.69 / 19.27 (2.37x) |
| 1M | f8 | 4034 | 0.9712 / 0.9695 | 4,121 / 6,304 (0.65x) | 282 / 375 (0.75x) | 38 / 51 (0.74x) | 42.24 / 19.27 (2.19x) |

RaBitQ's mean R@10 is 0.9674 (100K) and 0.9701 (1M).

- mb6 wins every timing mode at about equal memory. Batched QPS is 1.62x (100K) and 1.68x (1M) base, and p50 is 35% and
  30% lower. It now runs at 0.88-0.97x RaBitQ batched, with 32-34% less stored memory, and is still 1.8-1.9x slower
  single-query. mb5 needs 7-11% more bits; f8 keeps the budget but gains less.
- Shipped as the default: `max_bits` 0 (automatic) caps fast-target IP/cosine training at 6 bits, or at the per-rank
  budget when that is larger. L2, full-target models and refinement's finer model keep 8. The working tree's default
  reproduces the mb6 recall exactly: 0.9683 at 3752 bits on 100K, 0.9705 at 4082 bits on 1M (`capcheck.py`, TQ_MAX_BITS=0).
- The cap is a no-op where no tier exceeds 6 bits: OpenAI-100K at 794 / 1538 / 1652 bits and Gemini at 860 / 1844 bits
  (RaBitQ 2 bits/dim and Gemini 4 bits/dim budgets) give identical R@10 capped and uncapped.
- Base timings here run 3-5% below the earlier one-pass cells (0.54x vs 0.57x on 100K): run-to-run spread, as the
  references are not re-timed.

## Single-query profile at 4 bits/dim (`single-query/`)
OpenAI-100K, 3752 code bits (the new default at RaBitQ's 4-bit recall), flat, 1 thread, batch 1, alone on CPUs 0-7.
`prof.sh` runs perf on a build whose table build, query preparation, survivor test, epilogue and per-block scan are
marked noinline (`hooks.diff`). The section loop stays inlined; `perf annotate` splits it.

| Part | Share of query time |
|---|---:|
| Nibble scan (`scan_range_lookup_chunks_avx512`) | 82% |
| of which code loads / shuffle-add loop / section conversion / other | 47 / 43 / 4 / 6% of the scan |
| Query projection (`project_panel_block_avx512`) | ~8% |
| Exact range survivor test (`lookup_vector_survives`) | 5% |
| Table build, epilogue | ~1% |

- The scan reads every document's 469 code bytes. After the per-query fixed cost (~0.29 ms) it takes 31.8 ns/doc over
  100K documents (47 MB, from DRAM) and 22.6 ns/doc over 16K documents (in L3): memory costs ~29%, the shuffle work
  the rest. RaBitQ's whole query is ~18 ns/doc.
- Section conversion is 4% of the scan: with the 6-bit cap there is nothing left to gain there.
- The exact bound rejects almost nothing at 4 bits/dim, so its survivor test is a 5% loss.

Statistical pruning in flat search (`flatz.py`; workspace hook `TQ_FLAT_Z` / `TQ_FLAT_PREFIX`, `hooks.diff`). One
index, arms interleaved over 3 rounds, median ms/query, R@10 of the single-query results (seed 0, default 0.9683):

| Confidence, prefix tiers | ms/query | R@10 |
|---|---:|---:|
| 0, 4 (default) | 3.42-3.45 | 0.9683 |
| 8, 4 | 3.26 | 0.9683 |
| 4, 4 | 3.18 | 0.9683 |
| 8, 3 | 2.73-2.77 | 0.9683 |
| 4, 3 | 2.54 | 0.9652 |
| 2, 3 | 2.56 | 0.9548 |
| 8, 2 | 3.91 | 0.9679 |
| 4, 2 | 2.52 | 0.9511 |
| 2, 2 | 1.84 | 0.8410 |

- At no recall loss the best is confidence 8 with prefix 3: -20%, 1.5x RaBitQ's 1.83 ms. Below that, recall falls
  faster than time: the bound prunes per 64-document chunk, and a 3-tier prefix already holds the widest tiers.
- Closing the single-query gap needs fewer bytes per document in the first pass, as RaBitQ's 1-bit pass: a narrow
  prefix model scanned for all documents and a suffix read only for candidates (`TieredQRefinement`, #23).
