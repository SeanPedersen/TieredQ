# L2 automatic scan width: keep 8 bits (#44 finding)

Remote root: `/home/sean/experiments/TieredQ-issue44-l2scancap-20261111`, TieredQ 86f003d (Python package, Release).
`l2_scan_cap.py` builds one index per process on 32 threads, measures held-out recall@10 with one batch call, then
times 200 single queries alone on pinned CPU 2. `study.py` runs the models under the shared lock; `REPORT.md` (from
`summarize.py`) lists every row.

**Question.** `hartmut-compressed-20261110` and `hartmut-centroidq-20261110` found that L2's automatic
`max_scan_bits` of 8 sends single queries to the generic scan, and that a 4-bit cap was 22× faster for 2-bit centroid
routing on a laptop. Should L2 get an automatic cap?
- IP/cosine reach their cap of 6 through additive 3-bit books, which L2 dropped in 9e7bb30 (−0.72 pt R@10 on SIFT1M).
- Any L2 cap above 4 bits per coordinate still scans on the generic kernel. So the candidate rule is
  `max(4, ceil(bits/dim))`, which only matters up to 4 bits/dim.

**Data.** SIFT1M (1M × 128; 1,000 held-out queries, exact L2 truth) and the first 1M Qwen/MS MARCO rows decoded from
the SQ8 store (1M × 1024; 750 held-out queries, truth exact L2 over those rows, from `hartmut-defaults-20261111`). Seed
1 permutes the rows before `add()`, so the codec trains on a different sample (the flat models). For IVF it trains the
lists on a seeded sample.

## Flat `TieredQIndex` (full scan), recall@10 @ single-query p50, seeds 0 / 1

| Data, bits/dim | automatic (8) | cap 4 | Δ recall |
|---|---|---|---|
| SIFT1M, 1 | 0.5049 / 0.5062 @ 2.55 ms | identical @ 2.45 ms | 0 |
| SIFT1M, 2 | 0.7584 / 0.7636 @ 11.2 ms | 0.7601 / 0.7504 @ 4.2 ms | +0.002 / −0.013 |
| SIFT1M, 3 | 0.8755 / 0.8780 @ 14.4 ms | 0.8246 / 0.8201 @ 5.2 ms | −0.051 / −0.058 |
| SIFT1M, 4 | 0.9333 / 0.9333 @ 18.0 ms | 0.8303 / 0.8297 @ 7.6 ms | −0.103 |
| Qwen 1M, 2 | 0.9397 / 0.9376 @ 37.4 ms | 0.9279 / 0.9251 @ 21.5 ms | −0.012 |
| Qwen 1M, 3 | 0.9713 / 0.9721 @ 64.5 ms | 0.9416 / 0.9361 @ 30.7 ms | −0.030 / −0.036 |

The cap is 1.7–3× faster, not 22×; the 22× was the centroid-routing case at k = 10 on a laptop. From 2 bits/dim it
loses far more recall than the codec seeds spread, so at equal bits it buys speed with recall the 8-bit width keeps. At
1 bit/dim the allocation never goes above 4 bits, so the cap changes nothing. **Decision: L2 keeps the automatic width
of 8.** The planned Qwen 4-bit and IVF 3/4-bit models were dropped once these rows had decided it. One stopped model
(`qwen1m-flat-b4-cap0-s0`) was removed, and `qwen1m-flat-b3-cap4-s1/exit.json` was rewritten by hand while the driver
was retargeted.

## IVF L2 at 2 bits/dim: the existing nibble rule

`TieredQIVFIndex` already caps scans at 4 bits up to 2 bits/dim for every metric (`automatic_scan_bits`; the Python
binding's default `fine_tiers=False` keeps it on). Here it is 2,048 lists on Qwen 1M, cap 4 (the default) against an
explicit 8:

| nprobe | cap 4, seed 0 / 1 | cap 8, seed 0 / 1 |
|---:|---|---|
| 16 | 0.8497 @ 0.80 / 0.8401 @ 0.84 ms | 0.8579 @ 1.08 / 0.8520 @ 1.13 ms |
| 32 | 0.8823 @ 0.99 / 0.8787 @ 1.01 | 0.8912 @ 1.49 / 0.8912 @ 1.60 |
| 64 | 0.9029 @ 1.51 / 0.8995 @ 1.52 | 0.9131 @ 2.40 / 0.9139 @ 2.13 |
| 128 | 0.9153 @ 2.50 / 0.9104 @ 2.53 | 0.9255 @ 3.34 / 0.9261 @ 3.35 |
| 256 | 0.9220 @ 4.52 / 0.9168 @ 4.53 | 0.9328 @ 5.73 / 0.9321 @ 5.54 |

At equal latency the cap wins below about 0.91. At 1.5 ms it reaches 0.900–0.903, against about 0.884–0.891
interpolated for 8 bits, and only the cap gets under 1.1 ms. Above about 2.5 ms the 8-bit width wins by about 0.007,
and it has the higher ceiling (0.933 against 0.922 at nprobe 256). **Decision: no change.** The rule is a speed-for-recall
trade at low recall, not a loss, and no shipping configuration scans L2 at 2 bits/dim. The nested default's 1-bit
prefix is unaffected, as the 1-bit rows show.
