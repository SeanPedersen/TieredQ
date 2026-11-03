# TieredQ vs RaBitQ and Qdrant TQ+: benchmark report

Final report for [issue #4](https://github.com/SeanPedersen/TieredQ/issues/4). TieredQ is compared against the
official RaBitQ library and Qdrant's TurboQuant+ (TQ+) in flat exhaustive top-10 search on one CPU host, at **equal
measured memory**: TieredQ's code budget is calibrated to each reference's measured disk, steady-RAM or peak-RAM
bits per vector. Library: `072ea48` (fast mode). Harness: `d89d27d`. Data: five datasets from 128 to 1,536
dimensions, including DBpedia OpenAI-1536 at 1M vectors.

**At equal measured memory, TieredQ has higher recall@10 than RaBitQ and Qdrant TQ+ on every target**: all 30
disk targets, all 30 steady-RAM targets and all 30 peak-RAM targets (5 datasets × 2 references × 1/2/4 bits/dim).
The margins are +1.4 to +34.4 R@10 points at equal disk, largest at 1 bit/dim and at low dimension (SIFT, GloVe),
smallest at 4 bits/dim on OpenAI (+1.4 to +1.9). The earlier single loss (GloVe 4-bit at equal peak RAM against
TQ+) is now a +0.7 pt win with the current library.

- **Compression at R@10 ≥ 0.95:** TieredQ needs about half the disk of either reference on OpenAI-1536 (3,089 vs
  6,304 bits/vector at 1M; 3,215 vs 6,305 at 100K) and at most 0.86× on Gemini. RaBitQ and TQ+ do not reach 0.95 on
  SIFT or GloVe at ≤ 4 bits/dim; TieredQ does at 618 and 1,151 bits/vector.
- **Speed at equal recall:** where both codecs reach the same R@10 with TieredQ using 0.5–0.95× the bits, TieredQ is
  faster than RaBitQ at the 1- and 2-bit targets (1.05–2.15× batched) and slower at the 4-bit targets on OpenAI and
  Gemini (0.41–0.47× batched, 3.8–6.2× higher single-query p50). Against TQ+ it is 1.8–14× faster batched; that
  ratio is specific to this host (TQ+ runs AVX2 here).
- **Build:** TieredQ trains a model, so its build peak RSS is the largest of the three (5.0K–14.4K bits/vector at
  1M vs 1.6K–6.3K for RaBitQ and 3.3K–12.5K for TQ+); build time at 1M (21–34 s) is in line with RaBitQ at 2/4
  bits and below TQ+.
- **Scale:** from 100K to 1M on OpenAI the recall margins hold (+6.6 / +4.7 / +1.9 pt over RaBitQ at equal disk)
  and TieredQ's fixed model cost shrinks from ~320 to ~74 bits/vector.

All of this is flat exhaustive search on one shared Skylake-X host. TieredQ inside an IVF index with rerank, and
real search queries (MS MARCO) instead of held-out documents, are the follow-up work in
[#23](https://github.com/SeanPedersen/TieredQ/issues/23), [#22](https://github.com/SeanPedersen/TieredQ/issues/22)
and [#19](https://github.com/SeanPedersen/TieredQ/issues/19) (see [Follow-up](#follow-up-ivf-rerank-and-real-queries)).

## Answers to the four questions

1. **Recall at equal memory, and compression at R@10 = 0.95.** TieredQ wins every equal-disk, equal-steady-RAM and
   equal-peak-RAM target against both references (+1.4 to +34.4 pt at equal disk). To reach R@10 ≥ 0.95 it needs
   0.49× the disk bits of RaBitQ on OpenAI-1M (15.9× vs 7.8× compression) and 0.86× on Gemini; on SIFT and GloVe
   neither reference reaches 0.95 at ≤ 4 bits/dim.
2. **QPS and latency at equal recall.** At recall levels the references reach at 1 bit/dim, TieredQ is 1.5–2.2×
   faster than RaBitQ batched and has 0.5–0.8× its single-query p50 on all five datasets. At the 2-bit levels it
   is 1.05–1.5× batched with mixed p50. At the 4-bit levels on OpenAI and Gemini it is 0.41–0.47× RaBitQ batched and 3.8–6.2× slower single-query: there TieredQ's 3/4-bit tiers cost
   more per vector than RaBitQ's extra-bit scan. Against TQ+, TieredQ is faster on every batched schedule
   (1.8–14×) and on single-query p50 except OpenAI-100K 4-bit (1.12×).
3. **Build cost.** TieredQ's build is the heaviest in memory: it trains k-means and PCA on up to 32,768 rows, so
   its build peak at 100K is 22K–31K bits/vector on OpenAI (RaBitQ 1.8K–6.5K, TQ+ 3.7K–12.9K), falling to 5.0K–14.4K
   at 1M. Build time at 1M is 21–34 s, against 1.0 / 24.9 / 41.9 s for RaBitQ at 1/2/4 bits and 32–52 s for TQ+.
4. **Crossover by bits/dim and dimensionality.** Recall: no crossover in the measured range; TieredQ's margin
   shrinks with bits/dim (OpenAI-1M: +6.6 → +4.7 → +1.9 pt) and is largest at low d. Speed at equal memory: TieredQ
   is faster than RaBitQ at 1 bit/dim on every dataset (1.18–1.60× batched), about level at 2 bits/dim on
   d ≤ 200 (0.98–1.08×), and at 2–4 bits/dim on d = 768–1,536 RaBitQ is faster (1.3–2.9× batched). Speed at equal
   recall moves the crossover up: TieredQ stays faster through the 2-bit recall levels and loses only at the 4-bit
   levels on d ≥ 768.

## Setup

| Dataset | d | Base | Queries | Metric | Notes |
|---|---:|---:|---:|---|---|
| SIFT1M | 128 | 1,000,000 | 1,000 | L2 | TieredQ serves it with its native L2 metric |
| GloVe-200 | 200 | 100,000 | 1,000 | cosine | sorted 100K sample of ANN Benchmarks |
| DBpedia OpenAI-1536 | 1,536 | 100,000 | 1,000 | cosine | Qdrant's dataset, 101K seed-42 sample |
| DBpedia OpenAI-1536 1M | 1,536 | 999,000 | 1,000 | cosine | the same 1,000 queries against every other row |
| DBpedia Gemini-768 | 768 | 99,000 | 1,000 | cosine | 1,000 held-out documents; no 1M source exists |

- **Codecs.** TieredQ fast mode (`072ea48`), RaBitQ (official library `ced3763`, direct flat FHT adapter, native
  extra-bit bounds, high-accuracy scoring), Qdrant TQ+ (production `quantization` crate behind a Rust shim). Each
  uses its own native kernels; nothing is hand-tuned for one side. References run at 1, 2 and 4 bits per dimension.
- **Host.** Hartmut: Intel i9-7980XE (18 cores, AVX-512), 62 GB RAM, GCC 16.2.1, `-O3 -march=native
  -ffp-contract=off`, shared with other tenants. Timing runs pin CPUs 0–7 with no other benchmark on them.
- **Truth.** Exact FP32 top-10 by chunked matmul, on normalized vectors for cosine sets. Recall@10 is the mean over
  seeds 0–2 (each seed retrains or re-rotates); timing uses seed 0.
- **OpenAI-1M split.** The 1,000 queries are bit-identical to the 100K set's (same seed-42 draws), and its 100K base
  rows are a subset of the 1M base, so the two scales differ only in N
  ([manifest](../benchmark/results/issue4-1m-20261030/data/openai1m-999000-1000-v2.json)).

## What TieredQ does

1. **Train** on a seeded sample of up to 32,768 rows: 256 k-means centroids, then PCA of the residuals (FP64
   covariance in row chunks, stored as an FP16 basis).
2. **Allocate bits per PCA rank** (0–8 bits each) to minimise predicted score error at the requested code budget.
   Ranks with similar widths form tiers. Fast mode stores tiers of at most 4 bits in a nibble lookup layout whose
   padding is part of the budget; an exact O(d) allocator chooses the best allocation *including* that padding
   (#9), and 3-bit tiers use an additive codebook so they store at exactly 3 bits (#20).
3. **Weight ranks by variance^α.** α = 1 (plain MSE) is optimal when queries are whitened, as in TieredQ's L2 lift;
   for inner product and cosine, queries follow the data distribution and high-variance ranks matter more, so
   α = 1.5 is the default (#9). The 1M check below confirms α = 1.5.
4. **Encode** each row as a centroid id, scalar codes per rank, and a binary16 scale factor and score correction.
5. **Search** scans every row with SIMD lookup tables (AVX-512 here), per query or batched with OpenMP.

RaBitQ and TQ+ instead rotate randomly and spend the same number of bits on every dimension. TieredQ's advantage
comes from spending bits where the data's variance is; its costs are a trained model (fixed bytes amortised over N)
and a heavier build.

## How memory is matched

`benchmark/match_memory.py --quick` measures each reference at 1/2/4 bits and seeds 0–2, each cell built in one
fresh process and reloaded and served in another:

- **Disk:** saved index size, all model state included.
- **Steady RAM:** RSS growth after reload.
- **Peak RAM:** VmHWM growth over reload plus one pass of the query workload (threads 1/8 × batch 1/32).

Each (reference bits, metric) pair is a target: the minimum of that metric over the reference's seeds. The matcher
finds the largest TieredQ `code_bits` whose seed-0 cell fits (bisection on storage only, never on recall), confirms
it on seeds 1 and 2, and steps down if any seed overshoots. TieredQ's reported memory is the maximum over its seeds,
so every match is conservative for TieredQ. Compression ratio = 32·d / target bits per vector.

## Results

### Recall@10 at equal disk and equal steady RAM

TieredQ's code budget calibrated to each reference's measured disk or steady-RAM bits/vector; Δ is TieredQ − reference
R@10 in points (mean over seeds 0–2). Compression = 32·d / target bits.

| Dataset | Reference | bits/dim | Compression | Reference R@10 | TieredQ R@10 (disk) | Δ pt (disk) | TieredQ R@10 (ram) | Δ pt (ram) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT1M (L2) | RaBitQ | 1 | 18.3× | 0.2495 | 0.5932 | +34.37 | 0.5932 | +34.37 |
| SIFT1M (L2) | RaBitQ | 2 | 9.8× | 0.5484 | 0.8533 | +30.49 | 0.8533 | +30.49 |
| SIFT1M (L2) | RaBitQ | 4 | 6.1× | 0.8450 | 0.9586 | +11.36 | 0.9586 | +11.36 |
| SIFT1M (L2) | Qdrant TQ+ | 1 | 18.3× | 0.2685 | 0.5932 | +32.47 | 0.5932 | +32.47 |
| SIFT1M (L2) | Qdrant TQ+ | 2 | 11.6× | 0.5606 | 0.7925 | +23.19 | 0.7925 | +23.19 |
| SIFT1M (L2) | Qdrant TQ+ | 4 | 6.7× | 0.8525 | 0.9413 | +8.88 | 0.9413 | +8.88 |
| GloVe-200 100k | RaBitQ | 1 | 18.2× | 0.4078 | 0.5311 | +12.34 | 0.4838 | +7.60 |
| GloVe-200 100k | RaBitQ | 2 | 9.5× | 0.6597 | 0.8208 | +16.12 | 0.8029 | +14.33 |
| GloVe-200 100k | RaBitQ | 4 | 5.4× | 0.8818 | 0.9626 | +8.07 | 0.9586 | +7.68 |
| GloVe-200 100k | Qdrant TQ+ | 1 | 24.2× | 0.3553 | 0.4032 | +4.79 | 0.3956 | +4.03 |
| GloVe-200 100k | Qdrant TQ+ | 2 | 13.8× | 0.6276 | 0.6687 | +4.11 | 0.6687 | +4.11 |
| GloVe-200 100k | Qdrant TQ+ | 4 | 7.4× | 0.8826 | 0.8979 | +1.53 | 0.8979 | +1.53 |
| OpenAI-1536 100k | RaBitQ | 1 | 30.1× | 0.8140 | 0.8809 | +6.69 | 0.8762 | +6.22 |
| OpenAI-1536 100k | RaBitQ | 2 | 15.2× | 0.9013 | 0.9507 | +4.94 | 0.9483 | +4.70 |
| OpenAI-1536 100k | RaBitQ | 4 | 7.8× | 0.9674 | 0.9857 | +1.84 | 0.9853 | +1.79 |
| OpenAI-1536 100k | Qdrant TQ+ | 1 | 30.7× | 0.8142 | 0.8784 | +6.42 | 0.8762 | +6.20 |
| OpenAI-1536 100k | Qdrant TQ+ | 2 | 15.7× | 0.9046 | 0.9467 | +4.21 | 0.9467 | +4.21 |
| OpenAI-1536 100k | Qdrant TQ+ | 4 | 7.9× | 0.9715 | 0.9854 | +1.39 | 0.9847 | +1.32 |
| Gemini-768 99k | RaBitQ | 1 | 28.4× | 0.7216 | 0.8234 | +10.18 | 0.8093 | +8.77 |
| Gemini-768 99k | RaBitQ | 2 | 14.5× | 0.8549 | 0.9271 | +7.22 | 0.9243 | +6.93 |
| Gemini-768 99k | RaBitQ | 4 | 7.6× | 0.9516 | 0.9831 | +3.16 | 0.9807 | +2.92 |
| Gemini-768 99k | Qdrant TQ+ | 1 | 29.5× | 0.6980 | 0.8162 | +11.82 | 0.8093 | +11.13 |
| Gemini-768 99k | Qdrant TQ+ | 2 | 15.3× | 0.8398 | 0.9259 | +8.61 | 0.9227 | +8.29 |
| Gemini-768 99k | Qdrant TQ+ | 4 | 7.8× | 0.9521 | 0.9819 | +2.98 | 0.9810 | +2.89 |
| OpenAI-1536 1M | RaBitQ | 1 | 30.1× | 0.8296 | 0.8955 | +6.60 | 0.8942 | +6.46 |
| OpenAI-1536 1M | RaBitQ | 2 | 15.2× | 0.9101 | 0.9575 | +4.73 | 0.9575 | +4.73 |
| OpenAI-1536 1M | RaBitQ | 4 | 7.8× | 0.9701 | 0.9888 | +1.87 | 0.9887 | +1.86 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 30.7× | 0.8270 | 0.8935 | +6.65 | 0.8925 | +6.55 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 15.7× | 0.9104 | 0.9531 | +4.27 | 0.9531 | +4.27 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 7.9× | 0.9723 | 0.9882 | +1.59 | 0.9879 | +1.56 |

- **Every target is a win**, at both disk and steady RAM. Steady-RAM margins are within 0.5 pt of the disk ones
  except at the smallest targets (GloVe 1/2-bit, Gemini 1-bit: 0.8–4.7 pt lower), where per-process RSS overhead
  is a large share of a 264–672-bit target.
- Margins are largest where the references spend bits evenly on low-variance dimensions: SIFT (+9 to +34) and
  GloVe/Gemini at 1–2 bits (+4 to +16). On OpenAI they are +6.4 to +6.7 at 1 bit/dim and +1.4 to +1.9 at 4.
- Seed std of TieredQ recall is 0.0002–0.0042 (≤ 0.4 pt), below every margin.

### Scale: OpenAI-1536 at 100K and 1M

Same 1,000 queries, the 100K base a subset of the 1M base ([setup](#setup)):

| | 100K | 1M |
|---|---:|---:|
| Δ R@10 vs RaBitQ at equal disk, 1 / 2 / 4 bits/dim | +6.7 / +4.9 / +1.8 | +6.6 / +4.7 / +1.9 |
| Δ R@10 vs TQ+ at equal disk, 1 / 2 / 4 bits/dim | +6.4 / +4.2 / +1.4 | +6.7 / +4.3 / +1.6 |
| TieredQ code bits at RaBitQ 1-bit disk (1,632 bits/vector) | 1,307 | 1,557 |
| TieredQ disk beyond its code bits | ~320 bits/vector | ~74 bits/vector |
| Min disk bits/vector for R@10 ≥ 0.95, TieredQ vs RaBitQ | 3,215 vs 6,305 | 3,089 vs 6,304 |
| QPS 8t·b32 vs RaBitQ at equal RAM, 1 / 2 / 4 bits/dim | 1.50 / 0.42 / 0.35× | 1.23 / 0.53 / 0.42× |
| QPS 8t·b32 vs RaBitQ at equal recall, 1 / 2 / 4-bit levels | 2.15 / 1.05 / 0.41× | 1.90 / 1.09 / 0.47× |

- The recall margins are stable with N: the model's fixed cost (centroids, FP16 basis) is amortised over 10× more
  rows, so at equal disk TieredQ gets 250 more code bits per vector at 1M, which offsets the harder 1M search
  (both codecs' recall at fixed bits is slightly higher at 1M).
- Speed ratios are within run-to-run noise of the 100K ones; absolute QPS falls ~10× for every codec, as expected
  for a flat scan.

### Minimum measured bits per vector for R@10 ≥ 0.95

| Dataset | Codec | Min disk bits/vector at R@10 ≥ 0.95 | Compression | R@10 there | Best below 0.95 |
|---|---|---:|---:|---:|---|
| SIFT1M (L2) | TieredQ | 618 (573 code bits) | 6.6× | 0.9528 | 0.9484 at 618 (550 code bits) |
| SIFT1M (L2) | RaBitQ | not reached (max 0.8450) | – | – | 0.8450 at 672 (4 bits/dim) |
| SIFT1M (L2) | Qdrant TQ+ | not reached (max 0.8525) | – | – | 0.8525 at 608 (4 bits/dim) |
| GloVe-200 100k | TieredQ | 1,151 (1077 code bits) | 5.6× | 0.9575 | 0.9491 at 1,087 (1019 code bits) |
| GloVe-200 100k | RaBitQ | not reached (max 0.8818) | – | – | 0.8818 at 1,184 (4 bits/dim) |
| GloVe-200 100k | Qdrant TQ+ | not reached (max 0.8826) | – | – | 0.8826 at 864 (4 bits/dim) |
| OpenAI-1536 100k | TieredQ | 3,215 (2761 code bits) | 15.3× | 0.9509 | 0.9483 at 3,183 (2750 code bits) |
| OpenAI-1536 100k | RaBitQ | 6,305 (4 bits/dim) | 7.8× | 0.9674 | 0.9013 at 3,233 (2 bits/dim) |
| OpenAI-1536 100k | Qdrant TQ+ | 6,211 (4 bits/dim) | 7.9× | 0.9715 | 0.9046 at 3,139 (2 bits/dim) |
| Gemini-768 99k | TieredQ | 2,784 (2585 code bits) | 8.8× | 0.9719 | 0.9459 at 1,934 (1730 code bits) |
| Gemini-768 99k | RaBitQ | 3,232 (4 bits/dim) | 7.6× | 0.9516 | 0.8549 at 1,696 (2 bits/dim) |
| Gemini-768 99k | Qdrant TQ+ | 3,137 (4 bits/dim) | 7.8× | 0.9521 | 0.8398 at 1,601 (2 bits/dim) |
| OpenAI-1536 1M | TieredQ | 3,089 (2992 code bits) | 15.9× | 0.9540 | 0.9004 at 1,703 (1632 code bits) |
| OpenAI-1536 1M | RaBitQ | 6,304 (4 bits/dim) | 7.8× | 0.9701 | 0.9101 at 3,232 (2 bits/dim) |
| OpenAI-1536 1M | Qdrant TQ+ | 6,208 (4 bits/dim) | 7.9× | 0.9723 | 0.9104 at 3,136 (2 bits/dim) |

- TieredQ's minimum is the smallest *measured* cell at or above 0.95 on the matcher's budget grid, so it is an upper
  bound; the gap to the best cell below 0.95 shows the grid step (OpenAI-1M has no measured cell between 1,703 and
  3,089 bits/vector, Gemini none between 1,934 and 2,784). The equal-recall budgets in the next sections refine
  this: on OpenAI-1M, 3,968 code bits match RaBitQ 4-bit's 0.970.
- The references only exist at 1/2/4 bits/dim, so their minimum is their 4-bit cell (RaBitQ 3-bit, measured in
  the earlier #4 runs, reaches 0.944 on OpenAI-100K and stays below 0.95).

### Speed at equal memory

Speed ratios against TQ+ are host-specific: on this CPU, TQ+ runs its AVX2 kernels (no AVX-512 VNNI), while
TieredQ and RaBitQ run AVX-512 (see [Scope and limitations](#scope-and-limitations)).

TieredQ timed at the budget matched to each reference's steady RAM; ratios are TieredQ / reference (seed 0, CPUs 0–7).

| Dataset | Reference | bits/dim | TieredQ code bits | QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref | p50 1t·b1 ms TQ / ref | Build s TQ / ref |
|---|---|---:|---:|---|---|---|---|
| SIFT1M (L2) | RaBitQ | 1 | 160 | 2,713 / 2,307 (1.18×) | 461 / 300 (1.54×) | 7.73 / 3.25 (2.38×) | 0.9 / 0.1 (7.16×) |
| SIFT1M (L2) | RaBitQ | 2 | 350 | 1,930 / 1,976 (0.98×) | 303 / 249 (1.22×) | 11.36 / 3.97 (2.86×) | 1.4 / 1.5 (0.93×) |
| SIFT1M (L2) | RaBitQ | 4 | 607 | 1,637 / 1,535 (1.07×) | 249 / 231 (1.08×) | 20.19 / 4.28 (4.72×) | 2.6 / 2.7 (0.97×) |
| SIFT1M (L2) | Qdrant TQ+ | 1 | 160 | 2,713 / 260 (10.43×) | 461 / 35 (13.28×) | 7.73 / 28.10 (0.28×) | 0.9 / 2.8 (0.34×) |
| SIFT1M (L2) | Qdrant TQ+ | 2 | 285 | 2,172 / 430 (5.05×) | 335 / 55 (6.13×) | 9.42 / 18.18 (0.52×) | 1.3 / 3.5 (0.37×) |
| SIFT1M (L2) | Qdrant TQ+ | 4 | 542 | 1,572 / 416 (3.78×) | 256 / 53 (4.82×) | 16.31 / 18.66 (0.87×) | 2.1 / 4.5 (0.47×) |
| GloVe-200 100k | RaBitQ | 1 | 252 | 21,974 / 16,371 (1.34×) | 3,262 / 2,446 (1.33×) | 0.32 / 0.41 (0.79×) | 0.5 / 0.0 (12.21×) |
| GloVe-200 100k | RaBitQ | 2 | 581 | 13,801 / 12,753 (1.08×) | 2,021 / 1,984 (1.02×) | 0.49 / 0.52 (0.96×) | 0.6 / 0.3 (1.62×) |
| GloVe-200 100k | RaBitQ | 4 | 1083 | 10,504 / 12,131 (0.87×) | 1,812 / 1,718 (1.05×) | 2.10 / 0.58 (3.60×) | 0.6 / 0.6 (0.95×) |
| GloVe-200 100k | Qdrant TQ+ | 1 | 190 | 25,473 / 2,292 (11.12×) | 3,428 / 337 (10.18×) | 0.28 / 2.93 (0.10×) | 0.5 / 0.5 (1.05×) |
| GloVe-200 100k | Qdrant TQ+ | 2 | 393 | 18,228 / 2,404 (7.58×) | 2,471 / 328 (7.54×) | 0.39 / 3.06 (0.13×) | 0.5 / 0.6 (0.80×) |
| GloVe-200 100k | Qdrant TQ+ | 4 | 795 | 11,529 / 2,227 (5.18×) | 1,675 / 326 (5.14×) | 0.60 / 3.04 (0.20×) | 0.6 / 0.8 (0.75×) |
| OpenAI-1536 100k | RaBitQ | 1 | 1257 | 6,180 / 4,118 (1.50×) | 854 / 615 (1.39×) | 1.25 / 1.58 (0.79×) | 12.4 / 0.1 (108.02×) |
| OpenAI-1536 100k | RaBitQ | 2 | 2750 | 1,943 / 4,589 (0.42×) | 305 / 598 (0.51×) | 7.02 / 1.66 (4.23×) | 12.9 / 2.5 (5.15×) |
| OpenAI-1536 100k | RaBitQ | 4 | 5689 | 1,360 / 3,917 (0.35×) | 224 / 539 (0.42×) | 15.90 / 1.83 (8.70×) | 13.6 / 4.1 (3.29×) |
| OpenAI-1536 100k | Qdrant TQ+ | 1 | 1257 | 6,180 / 1,003 (6.16×) | 854 / 124 (6.89×) | 1.25 / 7.67 (0.16×) | 12.4 / 3.2 (3.84×) |
| OpenAI-1536 100k | Qdrant TQ+ | 2 | 2686 | 1,925 / 928 (2.07×) | 311 / 114 (2.74×) | 7.02 / 8.64 (0.81×) | 12.9 / 4.3 (3.01×) |
| OpenAI-1536 100k | Qdrant TQ+ | 4 | 5626 | 1,315 / 801 (1.64×) | 216 / 101 (2.15×) | 15.89 / 9.80 (1.62×) | 14.2 / 5.8 (2.47×) |
| Gemini-768 99k | RaBitQ | 1 | 650 | 12,297 / 7,675 (1.60×) | 1,714 / 1,209 (1.42×) | 0.58 / 0.82 (0.71×) | 3.1 / 0.1 (40.88×) |
| Gemini-768 99k | RaBitQ | 2 | 1445 | 6,319 / 8,394 (0.75×) | 833 / 1,119 (0.74×) | 1.18 / 0.86 (1.37×) | 3.6 / 1.2 (3.12×) |
| Gemini-768 99k | RaBitQ | 4 | 2972 | 2,818 / 8,284 (0.34×) | 448 / 1,152 (0.39×) | 6.82 / 0.90 (7.57×) | 3.7 / 2.0 (1.84×) |
| Gemini-768 99k | Qdrant TQ+ | 1 | 650 | 12,297 / 1,718 (7.16×) | 1,714 / 237 (7.22×) | 0.58 / 4.17 (0.14×) | 3.1 / 1.7 (1.83×) |
| Gemini-768 99k | Qdrant TQ+ | 2 | 1396 | 6,659 / 1,691 (3.94×) | 939 / 209 (4.49×) | 1.10 / 4.69 (0.23×) | 3.5 / 2.1 (1.63×) |
| Gemini-768 99k | Qdrant TQ+ | 4 | 2904 | 2,587 / 1,430 (1.81×) | 456 / 185 (2.46×) | 6.99 / 5.43 (1.29×) | 3.7 / 2.8 (1.34×) |
| OpenAI-1536 1M | RaBitQ | 1 | 1549 | 477 / 387 (1.23×) | 60 / 51 (1.18×) | 16.48 / 19.36 (0.85×) | 21.4 / 1.0 (21.89×) |
| OpenAI-1536 1M | RaBitQ | 2 | 3136 | 201 / 378 (0.53×) | 29 / 51 (0.56×) | 69.79 / 19.49 (3.58×) | 24.1 / 24.9 (0.96×) |
| OpenAI-1536 1M | RaBitQ | 4 | 6173 | 157 / 375 (0.42×) | 22 / 51 (0.44×) | 105.96 / 19.27 (5.50×) | 34.0 / 41.9 (0.81×) |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 1518 | 493 / 96 (5.15×) | 62 / 13 (4.64×) | 16.28 / 74.67 (0.22×) | 21.1 / 31.8 (0.66×) |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 3038 | 214 / 92 (2.31×) | 33 / 12 (2.79×) | 67.38 / 84.63 (0.80×) | 24.5 / 40.5 (0.61×) |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 6080 | 162 / 81 (2.01×) | 26 / 10 (2.58×) | 109.10 / 98.52 (1.11×) | 32.6 / 51.7 (0.63×) |

- **1 bit/dim:** TieredQ is faster than RaBitQ batched on every dataset (1.18–1.60×) with lower single-query p50
  on GloVe, OpenAI and Gemini, at +6 to +34 pt higher recall. SIFT p50 is 2.4× higher (L2 lift and 1M rows).
- **2–4 bits/dim, d ≥ 768:** RaBitQ is faster (TieredQ 0.34–0.75× batched, 1.4–8.7× higher p50); TieredQ's 3- and
  4-bit tiers are the cost (#13, #19.4). At these budgets TieredQ spends the same memory on +1.8 to +6.9 pt recall;
  the next table compares speed at equal recall instead.
- **Against TQ+:** TieredQ is 1.6–11× faster batched everywhere; single-query p50 is lower except OpenAI/Gemini
  4-bit and OpenAI-1M 4-bit (1.1–1.6× higher).

### Speed at equal recall

For each reference cell, the smallest TieredQ budget whose seed-0 R@10 reaches the reference's mean R@10
(`equal_recall.py`, bisection to 8 code bits), timed like the matched cells. Stored bits are measured disk bits/vector.

| Dataset | Reference | bits/dim | Reference R@10 | TieredQ code bits | TieredQ R@10 | Stored bits TQ / ref | QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref | p50 1t·b1 ms TQ / ref |
|---|---|---:|---:|---:|---:|---|---|---|---|
| SIFT1M (L2) | RaBitQ | 1 | 0.2495 | 66 | 0.2614 | 115 / 224 (0.52×) | 3,805 / 2,307 (1.65×) | 510 / 300 (1.70×) | 1.94 / 3.25 (0.60×) |
| SIFT1M (L2) | RaBitQ | 2 | 0.5484 | 146 | 0.5555 | 203 / 416 (0.49×) | 2,940 / 1,976 (1.49×) | 473 / 249 (1.90×) | 7.92 / 3.97 (2.00×) |
| SIFT1M (L2) | RaBitQ | 4 | 0.8450 | 346 | 0.8471 | 396 / 672 (0.59×) | 1,793 / 1,535 (1.17×) | 301 / 231 (1.30×) | 11.33 / 4.28 (2.65×) |
| SIFT1M (L2) | Qdrant TQ+ | 1 | 0.2685 | 74 | 0.3024 | 123 / 224 (0.55×) | 3,668 / 260 (14.10×) | 494 / 35 (14.20×) | 2.00 / 28.10 (0.07×) |
| SIFT1M (L2) | Qdrant TQ+ | 2 | 0.5606 | 154 | 0.5757 | 203 / 352 (0.58×) | 2,767 / 430 (6.43×) | 457 / 55 (8.35×) | 7.80 / 18.18 (0.43×) |
| SIFT1M (L2) | Qdrant TQ+ | 4 | 0.8525 | 354 | 0.8551 | 428 / 608 (0.70×) | 1,925 / 416 (4.63×) | 296 / 53 (5.58×) | 11.66 / 18.66 (0.62×) |
| GloVe-200 100k | RaBitQ | 1 | 0.4078 | 204 | 0.4164 | 272 / 352 (0.77×) | 24,007 / 16,371 (1.47×) | 3,501 / 2,446 (1.43×) | 0.32 / 0.41 (0.78×) |
| GloVe-200 100k | RaBitQ | 2 | 0.6597 | 391 | 0.6687 | 465 / 672 (0.69×) | 17,125 / 12,753 (1.34×) | 2,511 / 1,984 (1.27×) | 0.40 / 0.52 (0.77×) |
| GloVe-200 100k | RaBitQ | 4 | 0.8818 | 746 | 0.8811 | 818 / 1,184 (0.69×) | 11,679 / 12,131 (0.96×) | 1,719 / 1,718 (1.00×) | 0.59 / 0.58 (1.01×) |
| GloVe-200 100k | Qdrant TQ+ | 1 | 0.3553 | 165 | 0.3584 | 231 / 264 (0.87×) | 27,080 / 2,292 (11.82×) | 3,663 / 337 (10.88×) | 0.27 / 2.93 (0.09×) |
| GloVe-200 100k | Qdrant TQ+ | 2 | 0.6276 | 359 | 0.6295 | 433 / 464 (0.93×) | 18,951 / 2,404 (7.88×) | 2,646 / 328 (8.08×) | 0.37 / 3.06 (0.12×) |
| GloVe-200 100k | Qdrant TQ+ | 4 | 0.8826 | 746 | 0.8811 | 818 / 864 (0.95×) | 11,679 / 2,227 (5.25×) | 1,719 / 326 (5.27×) | 0.59 / 3.04 (0.19×) |
| OpenAI-1536 100k | RaBitQ | 1 | 0.8140 | 812 | 0.8162 | 1,088 / 1,633 (0.67×) | 8,844 / 4,118 (2.15×) | 1,224 / 615 (1.99×) | 0.80 / 1.58 (0.51×) |
| OpenAI-1536 100k | RaBitQ | 2 | 0.9013 | 1598 | 0.9027 | 1,953 / 3,233 (0.60×) | 4,838 / 4,589 (1.05×) | 630 / 598 (1.05×) | 1.69 / 1.66 (1.02×) |
| OpenAI-1536 100k | RaBitQ | 4 | 0.9674 | 3554 | 0.9651 | 4,067 / 6,305 (0.65×) | 1,590 / 3,917 (0.41×) | 308 / 539 (0.57×) | 11.38 / 1.83 (6.23×) |
| OpenAI-1536 100k | Qdrant TQ+ | 1 | 0.8142 | 812 | 0.8162 | 1,088 / 1,603 (0.68×) | 8,844 / 1,003 (8.82×) | 1,224 / 124 (9.87×) | 0.80 / 7.67 (0.10×) |
| OpenAI-1536 100k | Qdrant TQ+ | 2 | 0.9046 | 1664 | 0.9045 | 2,032 / 3,139 (0.65×) | 4,679 / 928 (5.04×) | 637 / 114 (5.61×) | 1.68 / 8.64 (0.19×) |
| OpenAI-1536 100k | Qdrant TQ+ | 4 | 0.9715 | 4022 | 0.9717 | 4,538 / 6,211 (0.73×) | 1,426 / 801 (1.78×) | 232 / 101 (2.31×) | 10.99 / 9.80 (1.12×) |
| Gemini-768 99k | RaBitQ | 1 | 0.7216 | 440 | 0.7235 | 585 / 864 (0.68×) | 14,642 / 7,675 (1.91×) | 2,137 / 1,209 (1.77×) | 0.48 / 0.82 (0.59×) |
| Gemini-768 99k | RaBitQ | 2 | 0.8549 | 860 | 0.8578 | 1,017 / 1,696 (0.60×) | 10,030 / 8,394 (1.19×) | 1,384 / 1,119 (1.24×) | 0.72 / 0.86 (0.84×) |
| Gemini-768 99k | RaBitQ | 4 | 0.9516 | 1856 | 0.9530 | 2,034 / 3,232 (0.63×) | 3,563 / 8,284 (0.43×) | 601 / 1,152 (0.52×) | 4.83 / 0.90 (5.35×) |
| Gemini-768 99k | Qdrant TQ+ | 1 | 0.6980 | 404 | 0.7007 | 543 / 833 (0.65×) | 15,704 / 1,718 (9.14×) | 2,268 / 237 (9.56×) | 0.45 / 4.17 (0.11×) |
| Gemini-768 99k | Qdrant TQ+ | 2 | 0.8398 | 770 | 0.8416 | 926 / 1,601 (0.58×) | 10,768 / 1,691 (6.37×) | 1,485 / 209 (7.11×) | 0.69 / 4.69 (0.15×) |
| Gemini-768 99k | Qdrant TQ+ | 4 | 0.9521 | 1856 | 0.9530 | 2,034 / 3,137 (0.65×) | 3,563 / 1,430 (2.49×) | 601 / 185 (3.24×) | 4.83 / 5.43 (0.89×) |
| OpenAI-1536 1M | RaBitQ | 1 | 0.8296 | 932 | 0.8312 | 1,001 / 1,632 (0.61×) | 735 / 387 (1.90×) | 95 / 51 (1.87×) | 10.43 / 19.36 (0.54×) |
| OpenAI-1536 1M | RaBitQ | 2 | 0.9101 | 1814 | 0.9122 | 1,889 / 3,232 (0.58×) | 412 / 378 (1.09×) | 52 / 51 (1.02×) | 19.18 / 19.49 (0.98×) |
| OpenAI-1536 1M | RaBitQ | 4 | 0.9701 | 3968 | 0.9706 | 4,054 / 6,304 (0.64×) | 176 / 375 (0.47×) | 26 / 51 (0.50×) | 73.52 / 19.27 (3.82×) |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 0.8270 | 902 | 0.8261 | 968 / 1,600 (0.61×) | 768 / 96 (8.02×) | 98 / 13 (7.32×) | 10.15 / 74.67 (0.14×) |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 0.9104 | 1814 | 0.9122 | 1,889 / 3,136 (0.60×) | 412 / 92 (4.45×) | 52 / 12 (4.41×) | 19.18 / 84.63 (0.23×) |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 0.9723 | 4190 | 0.9735 | 4,280 / 6,208 (0.69×) | 172 / 81 (2.13×) | 25 / 10 (2.46×) | 75.39 / 98.52 (0.77×) |

**Update, #13 (2026-11-03): OpenAI-1536 against RaBitQ on current main.** Nibble scoring for wide tiers (c0910d6),
flat range pruning (08173a5) and the default 6-bit cap for fast IP/cosine tiers (178ef93) change the OpenAI rows
below. 2 bits/dim: one pass per cell on 9e7bb30
([issue13-refs-20261102/final](../benchmark/results/issue13-refs-20261102/README.md)), unchanged by the cap. 4 bits/dim:
median of 3 interleaved rounds on 05ee4d3 with the cap
([issue13-gap4-20261103](../benchmark/results/issue13-gap4-20261103/README.md)). TieredQ R@10 is the mean over seeds 0-2.
The other rows of the table above predate #13.

| Dataset | bits/dim | Reference R@10 | TieredQ code bits | TieredQ R@10 | Stored bits TQ / ref | QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref | p50 1t·b1 ms TQ / ref |
|---|---:|---:|---:|---:|---|---|---|---|
| OpenAI-1536 100k | 2 | 0.9013 | 1538 | 0.9007 | 1,895 / 3,233 (0.59×) | 7,351 / 4,589 (1.60×) | 999 / 598 (1.67×) | 1.48 / 1.66 (0.89×) |
| OpenAI-1536 100k | 4 | 0.9674 | 3752 | 0.9664 | 4,264 / 6,305 (0.68×) | 3,453 / 3,917 (0.88×) | 487 / 539 (0.90×) | 3.40 / 1.83 (1.86×) |
| OpenAI-1536 1M | 2 | 0.9101 | 1820 | 0.9112 | 1,898 / 3,232 (0.59×) | 730 / 378 (1.93×) | 97 / 51 (1.89×) | 19.76 / 19.49 (1.01×) |
| OpenAI-1536 1M | 4 | 0.9701 | 4082 | 0.9694 | 4,169 / 6,304 (0.66×) | 362 / 375 (0.97×) | 48 / 51 (0.93×) | 34.51 / 19.27 (1.79×) |

- On OpenAI at RaBitQ's 4-bit recall, TieredQ is now at 0.88–0.97× RaBitQ's batched QPS (was 0.41–0.47×) with 32–34%
  less stored memory. Single-query p50 is still 1.8–1.9× RaBitQ's.

- At equal recall TieredQ stores 0.49–0.95× the reference's bits (0.58–0.73× on OpenAI and Gemini).
- **Against RaBitQ:** faster batched at every 1- and 2-bit level (1.05–2.15×); at the 4-bit levels it is
  faster on SIFT (1.17×), equal on GloVe (0.96×) and slower on OpenAI and Gemini (0.41–0.47× batched, p50
  3.8–6.2× higher). Single-query p50 is lower at the 1-bit levels except SIFT.
- **Against TQ+:** faster on every batched schedule (1.8–14×); p50 lower except OpenAI-100K 4-bit (1.12×).
- Three TieredQ budgets land slightly below the reference's mean over seeds (GloVe 4-bit −0.07 pt, OpenAI-100K
  RaBitQ 4-bit −0.23 pt, OpenAI-1M TQ+ 1-bit −0.09 pt) because the budget is chosen on seed 0; see
  [limitations](#scope-and-limitations).

### Build cost

Seed 0. Build time from the timing runs (alone on CPUs 0–7: train + encode + pack for TieredQ, rotate + encode for
the references); build peak RSS growth from the matching runs, in bits per indexed vector.

| Dataset | Codec | 1 bit/dim: time, peak | 2 bits/dim | 4 bits/dim |
|---|---|---|---|---|
| OpenAI-1536 100K | TieredQ (RAM-matched to RaBitQ) | 12.4 s, 21.8K | 12.9 s, 24.5K | 13.6 s, 31.3K |
| OpenAI-1536 100K | RaBitQ | 0.1 s, 1.8K | 2.5 s, 3.4K | 4.1 s, 6.5K |
| OpenAI-1536 100K | Qdrant TQ+ | 3.2 s, 3.7K | 4.3 s, 6.9K | 5.8 s, 12.9K |
| OpenAI-1536 1M | TieredQ (RAM-matched to RaBitQ) | 21.4 s, 5.0K | 24.1 s, 8.3K | 34.0 s, 14.4K |
| OpenAI-1536 1M | RaBitQ | 1.0 s, 1.6K | 24.9 s, 3.3K | 41.9 s, 6.3K |
| OpenAI-1536 1M | Qdrant TQ+ | 31.8 s, 3.3K | 40.5 s, 6.3K | 51.7 s, 12.5K |

- **Memory:** TieredQ's build peak is a fixed training cost (FP64 covariance and k-means over up to 32,768 rows)
  plus the index. At 100K that fixed part dominates (≈ 20K bits/vector beyond the index); at 1M it is ≈ 3.4K at the
  1-bit budget and ≈ 8.2K at 4 bits. RaBitQ builds within 1–9% of its index size; TQ+ peaks at about twice its
  index.
- **Time:** TieredQ's training is a fixed ~10 s at d = 1,536 (k-means, covariance, eigendecomposition), so it is the
  slowest at 100K but in line with RaBitQ's 2/4-bit encoding at 1M, and faster than TQ+. RaBitQ 1-bit (no extra
  bits) is the fastest build by far.
- The other datasets follow the same pattern (speed tables above, `Build s` column): TieredQ 0.5–3.7 s against
  0.0–2.7 s for RaBitQ on SIFT, GloVe and Gemini. Bounding the training sample inside the library and shrinking its
  build peak are tracked in #19.3.

### α at 1M

Fast mode on OpenAI-1M, R@10 (mean over seeds 0–2), allocation power α chosen on even query ids and reported on
odd ones ([alpha/](../benchmark/results/issue4-1m-20261030/alpha/)):

| α | 1 bit/dim | 2 | 3 | 4 | Δ vs 1.5, even ids (selection) | Δ vs 1.5, odd ids (held out) |
|---|---:|---:|---:|---:|---:|---:|
| 1 (MSE) | 0.8928 | 0.9481 | 0.9717 | 0.9855 | −0.49 pt | −0.37 pt |
| 1.25 | **0.8953** | 0.9527 | 0.9750 | 0.9867 | −0.15 | −0.12 |
| **1.5 (default)** | 0.8930 | **0.9546** | **0.9790** | 0.9887 | – | – |
| 1.75 | 0.8937 | 0.9534 | 0.9776 | **0.9892** | −0.03 | −0.04 |
| 2 | 0.8893 | 0.9522 | 0.9754 | 0.9886 | −0.36 | −0.13 |

The shipped α = 1.5 holds at 1M: nothing beats it on the selection half, 1.75 is flat (−0.03 / −0.04 pt), and
plain MSE (α = 1) loses 0.4–0.5 pt, as at 100K (#9). α = 1.25 is ahead only at 1 bit/dim.

## Follow-up: IVF, rerank and real queries

These are outside #4's flat-search scope and are tracked in their own issues:

- **IVF + rerank** (#19.1, #23): TieredQ inside the same 256-list IVF partition as RaBitQ on OpenAI-1M, with INT8,
  FP32 or nested TieredQ refinement as the rerank tier, reported per resource (RAM, disk, reads per query, QPS):
  [cost-view-1m.md](../benchmark/results/issue6-ivf-rerank-20261030/cost-view-1m.md).
- **Real search queries** (#22): MS MARCO-1M (Cohere embed-v3) with real dev questions and held-out passages on one
  base: [issue22-msmarco1m-20261031](../benchmark/results/issue22-msmarco1m-20261031/README.md).
- **Training and drift, build cost, 2–4 bit speed, error bounds, scale** (#19).

## Scope and limitations

- **One shared host.** Hartmut runs other tenants; timing pins CPUs 0–7 but memory bandwidth and frequency are
  shared. Expect ±10–30% run-to-run timing noise; ratios below ~1.3× are within noise.
- **Reference timings are from earlier runs.** RaBitQ was timed on 26–27 October and TQ+ on 29 October, on the same
  host, inputs and harness; their code has not changed since, so their cells are reused rather than re-run. A
  same-session cross-check of RaBitQ on OpenAI/Gemini 100K (29–30 October) agreed within −10% to +22%
  ([cross-check](../benchmark/results/issue4-1m-20261030/dbpedia-speed-100k/)). The 1M references were timed in the
  same session as TieredQ.
- **100K scale for three of five sets.** Fixed per-index costs are divided by N. On OpenAI at 1 bit/dim, TieredQ's
  disk beyond its code bits is ~320 bits/vector at 100K and ~74 at 1M (model plus per-row metadata), so 100K
  results understate TieredQ at deployment scale; the OpenAI 1M run measures that directly. Gemini exists at 100K
  only.
- **Peak RAM** varies by up to ~20 bits/vector between seeds at 100K, so peak-matched margins carry calibration
  noise; they are in the appendix, and disk and steady RAM are the headline.
- **Speed at equal recall** uses seed-0 recall to choose the TieredQ budget (mean over seeds is reported and can sit
  slightly below the reference).
- **TQ+ runs its AVX2 kernels on this host.** Qdrant's TQ+ kernels dispatch at runtime in the order AVX-512 VNNI →
  AVX2 → SSE. The i9-7980XE (Skylake-X) has AVX-512 F/BW/VL but no VNNI, so TQ+ uses AVX2 while TieredQ and RaBitQ
  use AVX-512. The shim is built with `-C target-cpu=native`, LTO and Qdrant's production batch scorer, so the setup
  is correct, but TQ+ speed ratios here are host-specific and likely overstate TieredQ's lead over TQ+. Recall is
  unaffected.
- **TQ+ single-query search uses one thread.** The shim parallelises across queries only, so TQ+'s 8-thread
  batch-1 latency is single-threaded by design, while TieredQ splits one query's scan across threads. Qdrant itself
  would spread a query over its segments, so that schedule is not reported for TQ+. Compare TQ+ on 1-thread and
  batch-32 schedules only.
- **Fast mode only**, flat search only, top-10 only, CPU only. No rerank, no ANN index.
- **α was chosen on the same datasets** (split by query: chosen on even query ids, reported on odd ones).

## Reproduction

All runs: Hartmut, `~/experiments/`, Python `TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv`.

| Result | Bundle | Workspace / jobs |
|---|---|---|
| OpenAI-1M data, α, match, equal recall, timing | [issue4-1m-20261030](../benchmark/results/issue4-1m-20261030/README.md) | `TieredQ-issue4-1m-20261030`, [`issue4_1m/`](../benchmark/experiments/issue4_1m/) |
| 100K / SIFT matches and TieredQ timing (current library) | [issue9-ship-20261030](../benchmark/results/issue9-ship-20261030/README.md) | `TieredQ-issue9-ship-20261029`, `issue9_empirical/job_ship.sh`, `job_final.sh` |
| Reference cells and reference timing | [issue4-final-20261029](../benchmark/results/issue4-final-20261029/README.md), [issue4-range-refresh-20261027](../benchmark/results/issue4-range-refresh-20261027/) | `TieredQ-issue4-final-20261029` |

```sh
cd benchmark
# Data (OpenAI-1M streams the cached HF dataset; ~6.1 GB cache)
python -c "from pathlib import Path; from data import prepare; prepare(Path('data'), 'openai1m')"
# Equal-memory match against each reference
python match_memory.py --quick --reference rabitq --reference-bits 1 2 4 --datasets openai1m --output out-match
python match_memory.py --quick --reference qdrant --reference-bits 1 2 4 --datasets openai1m --output out-match
# Timing at a matched budget (seed 0, threads 1/8 x batch 1/32)
python run.py --codec tieredq --datasets openai1m --bits 2 --code-bits <matched> --seeds 0 --cell cell.json
# Tables of this report
python experiments/issue4_1m/report_tables.py results/issue4-1m-20261030
```

## Appendix: recall at equal peak RAM

Peak RAM is VmHWM growth over reload plus one query pass (threads 1/8 × batch 1/32). It carries up to ~20 bits/vector
of seed-to-seed noise at 100K, so these margins are less precise than the disk and steady-RAM ones.

| Dataset | Reference | bits/dim | Compression | Reference R@10 | TieredQ R@10 (peak) | Δ pt (peak) |
|---|---|---:|---:|---:|---:|---:|
| SIFT1M (L2) | RaBitQ | 1 | 17.9× | 0.2495 | 0.5241 | +27.46 |
| SIFT1M (L2) | RaBitQ | 2 | 9.7× | 0.5484 | 0.8245 | +27.61 |
| SIFT1M (L2) | RaBitQ | 4 | 6.1× | 0.8450 | 0.9528 | +10.77 |
| SIFT1M (L2) | Qdrant TQ+ | 1 | 17.7× | 0.2685 | 0.5241 | +25.56 |
| SIFT1M (L2) | Qdrant TQ+ | 2 | 11.4× | 0.5606 | 0.7564 | +19.58 |
| SIFT1M (L2) | Qdrant TQ+ | 4 | 6.7× | 0.8525 | 0.9357 | +8.32 |
| GloVe-200 100k | RaBitQ | 1 | 16.1× | 0.4078 | 0.4300 | +2.22 |
| GloVe-200 100k | RaBitQ | 2 | 8.9× | 0.6597 | 0.6721 | +1.24 |
| GloVe-200 100k | RaBitQ | 4 | 5.2× | 0.8818 | 0.9173 | +3.55 |
| GloVe-200 100k | Qdrant TQ+ | 1 | 19.1× | 0.3553 | 0.3584 | +0.31 |
| GloVe-200 100k | Qdrant TQ+ | 2 | 12.0× | 0.6276 | 0.6295 | +0.19 |
| GloVe-200 100k | Qdrant TQ+ | 4 | 6.9× | 0.8826 | 0.8894 | +0.68 |
| OpenAI-1536 100k | RaBitQ | 1 | 29.0× | 0.8140 | 0.8726 | +5.86 |
| OpenAI-1536 100k | RaBitQ | 2 | 14.9× | 0.9013 | 0.9418 | +4.06 |
| OpenAI-1536 100k | RaBitQ | 4 | 7.7× | 0.9674 | 0.9820 | +1.47 |
| OpenAI-1536 100k | Qdrant TQ+ | 1 | 28.9× | 0.8142 | 0.8726 | +5.84 |
| OpenAI-1536 100k | Qdrant TQ+ | 2 | 15.2× | 0.9046 | 0.9383 | +3.37 |
| OpenAI-1536 100k | Qdrant TQ+ | 4 | 7.8× | 0.9715 | 0.9835 | +1.20 |
| Gemini-768 99k | RaBitQ | 1 | 27.2× | 0.7216 | 0.8001 | +7.85 |
| Gemini-768 99k | RaBitQ | 2 | 14.2× | 0.8549 | 0.9243 | +6.94 |
| Gemini-768 99k | RaBitQ | 4 | 7.5× | 0.9516 | 0.9773 | +2.58 |
| Gemini-768 99k | Qdrant TQ+ | 1 | 27.0× | 0.6980 | 0.8093 | +11.13 |
| Gemini-768 99k | Qdrant TQ+ | 2 | 14.7× | 0.8398 | 0.9205 | +8.07 |
| Gemini-768 99k | Qdrant TQ+ | 4 | 7.7× | 0.9521 | 0.9754 | +2.33 |
| OpenAI-1536 1M | RaBitQ | 1 | 30.0× | 0.8296 | 0.8942 | +6.46 |
| OpenAI-1536 1M | RaBitQ | 2 | 15.2× | 0.9101 | 0.9565 | +4.64 |
| OpenAI-1536 1M | RaBitQ | 4 | 7.8× | 0.9701 | 0.9893 | +1.92 |
| OpenAI-1536 1M | Qdrant TQ+ | 1 | 30.6× | 0.8270 | 0.8925 | +6.55 |
| OpenAI-1536 1M | Qdrant TQ+ | 2 | 15.6× | 0.9104 | 0.9531 | +4.27 |
| OpenAI-1536 1M | Qdrant TQ+ | 4 | 7.9× | 0.9723 | 0.9888 | +1.65 |

Every target is a win; the smallest margins are GloVe against TQ+ (+0.2 to +0.7 pt) and GloVe 1/2-bit against
RaBitQ (+1.2 to +2.2), where per-process overhead is a large share of a 264–672-bit target.
