# #4 precheck after #31: fine tiers on / off / exact vs RaBitQ and Qdrant TQ+ (2026-11-05)

Small matrix ahead of the full #4 re-run: which TieredQ variant(s) #4 should report, and whether any budget needs an
extra row to be fair to the references. Tables: [summary-openai.md](summary-openai.md),
[summary-gemini.md](summary-gemini.md). Raw rows: `results.csv` (every cell, every seed, k and schedule),
`memory.csv`, `matches.csv`/`matches-*.json`, `calibration-*.csv`, and `equal-ram-memory.csv` for the follow-up.

**Follow-ups, same day:**
- **"Equal disk" below means equal effective bytes.** The match target is run.py's effective storage, the larger of
  the file and the accounted serving bytes. Fine tiers' accounted bytes exceeded their file, so the default got
  106-561 fewer code bits than off. The tables are unchanged; read "equal disk" as this.
- **Finding 4's RAM premium is fixed.** See [fine-ram-20261105](../fine-ram-20261105/README.md): the default and exact
  now use off's steady RAM, with the same hits and no QPS/latency change. The recommendation below is updated for it.

## Provenance

- **Library and harness:** `fef7b75` (main; #32 IVF fine tiers on top of the harness commit `f71095a`). The tree was a
  `git archive` snapshot in `hartmut:~/experiments/TieredQ-issue4-precheck-20261105/source` (`SOURCE_REVISION`).
- **RaBitQ:** the pinned #4 adapter (flat, `flat-fht-native-bounds-hacc`), with its default #4 config.
- **TQ+:** the #4 Qdrant shim, the same `libqdrant_bench.so` as `issue4-final-20261029/source-tq`.
- **Host:** hartmut, Intel i9-7980XE (18 cores / 36 threads, AVX-512). Kernel 7.3.1-cachyos, GCC 16.2.1. Native
  module built `-O3 -march=native` with AVX-512BW and AVX2 compiled in. Python is the #4 venv (3.12).
- **Data:** the cached #4 splits, OpenAI-1536 100K (100,000 × 1536) and Gemini-768 99K (99,000 × 768), each with
  1,000 queries and cosine metric.
  - The #4 cache links in the follow-up prompt all point at deleted directories. The caches were rebuilt from the raw
    float32 exports in `issue20-ship-20261029/data`.
  - Their base/queries/truth SHA-256s match the #4 records in
    `issue4-range-refresh-20261027/reuse/{data-verification,gemini-manifest}.json` (`cache-manifest.json`).
  - Exact top-100 truth was computed by `data.deep_truth`. Its top 10 must agree with the cached truth.
- **Job:** `benchmark/experiments/issue4_precheck/job.sh` ran from 09:51 to 11:17 UTC with exit code 0 and no failed
  cells (`phases.log`).
  1. Recall lanes on CPUs 8-31: seeds 1-2 of every nominal cell, then seed-0 memory cells.
  2. Equal-disk calibration.
  3. Recall lanes for the matched cells.
  4. Timing alone on CPUs 0-7 (66 min), with CPUs 18-25 (their SMT siblings) idle and the host otherwise quiet
     (`host-processes-timing.txt`). The cells ran in one session, ordered by dataset and budget, with all codecs of
     a budget interleaved.

## Method

- **Codecs:**
  - **TieredQ default:** fast index, `fine_tiers` on (`fast-native`).
  - **TieredQ off:** the same with `fine_tiers` off (`fast-native-nofine`, the pre-#31 fast mode).
  - **TieredQ exact:** a full-target model served by the fast index, i.e. exact fast scores with their default fine
    tiers (`exact-native`).
  - **RaBitQ** and **Qdrant TQ+.** TQ+ has only 1/2/4-bit modes, so its 8-bit cells are n/a. Its shim searches a fixed
    top-10, so TQ+ has no R@100 and no k=100 timing.
- **Budgets:** 1, 2, 4 and 8 bits/dim nominal. On top of that, each TieredQ variant is matched to each reference
  cell's measured bytes/vector (`match_disk.py`).
  - The match picks the largest seed-0 `code_bits` at or below the target, using `match_memory`'s bracket search at
    8-code-bit resolution.
  - Seeds 1-2 of a matched budget can differ by a few bytes. 8 of the 84 matched seed-1/2 cells are 0.1-1.8 B/vector
    (at most 0.9%) over target.
- **bytes/vector:** run.py's effective storage, i.e. the larger of the saved file and the reloaded index's accounted
  serving bytes. This counts codes, per-vector metadata, padding and the model amortized over n. Matching uses the
  same measure for both sides.
- **Recall:** mean ± sample std over seeds 0-2. R@10 comes from k=10 searches and R@100 from k=100 searches. TQ+ is
  seed-independent (std 0).
- **Timing:** seed 0, `measure.py` (3 warmups, 3 repeats over the 1,000 queries).
  - Schedules: 8 threads at batch 32 and batch 1,000 (QPS), and batch 1 at 1 and 8 threads (p50/p95 per call).
  - Each at k=10 and k=100.
- **Build:** train + encode + pack wall time at 8 threads. Encode is measured separately as vectors/s.
- **Memory:** `measure_memory` quick cells (seed 0, nominal budgets only): disk, steady RSS after reload, peak RSS,
  and build peak.

## Sanity checks against #31 (R@10, mean of 3 seeds)

| | Gemini default vs off | OpenAI default vs off |
|---|---|---|
| 1-2 bits/dim | identical (0.8387, 0.9330) | identical (0.9009, 0.9553) |
| 4 bits/dim | 0.9783 → 0.9832 (#31: 0.9791 → 0.9829) | 0.9835 → 0.9889 (#31: 0.9832 → 0.9885) |
| 8 bits/dim | 0.9932 → 0.9983 (#31: 0.9934 → 0.9985) | 0.9955 → 0.9991 (#31: 0.9956 → 0.9994) |

- Exact wins TieredQ batch throughput at 8 bits/dim: 1,271 vs 674 QPS on OpenAI and 2,427 vs 1,335 on Gemini.
- Its single-query p50 is 2.4× worse: 32.9 vs 13.9 ms on OpenAI and 16.2 vs 6.6 ms on Gemini. Both match #31.

## Findings at equal disk (equal effective bytes; R@10 / batch QPS@10 8t·b32 / single p50@10 1t·b1 ms)

| dataset, reference cell | reference | TieredQ default | TieredQ off | TieredQ exact |
|---|---|---|---|---|
| OpenAI, RaBitQ 4-bit (788 B) | 0.9674 / 4,250 / 1.64 | 0.9859 / 2,005 / 5.53 | 0.9814 / 2,467 / 5.12 | 0.9859 / 1,529 / 13.79 |
| OpenAI, RaBitQ 8-bit (1,556 B) | 0.9967 / 4,447 / 1.63 | 0.9987 / 700 / 13.16 | **0.9957** / 736 / 13.33 | 0.9988 / 1,281 / 30.94 |
| Gemini, RaBitQ 4-bit (404 B) | 0.9516 / 9,327 / 0.83 | 0.9821 / 3,796 / 2.65 | 0.9780 / 4,439 / 2.57 | 0.9827 / 2,965 / 6.01 |
| Gemini, RaBitQ 8-bit (788 B) | 0.9955 / 8,640 / 0.82 | 0.9976 / 1,409 / 6.49 | **0.9920** / 1,406 / 6.74 | 0.9979 / 2,461 / 15.53 |

1. **1-2 bits/dim:** fine tiers never engage, because no rank goes wide. Default and off give the same recall, bytes
   and speed (within noise). Exact matches their recall but halves batch QPS from 2 bits/dim. TieredQ beats both
   references on recall at every equal-disk point:
   - +0.05-0.10 R@10 over RaBitQ and +0.04-0.12 over TQ+.
   - At 1 bit/dim it is also faster than RaBitQ on batch and single-query. At 2 bits/dim it is faster on batch on
     OpenAI (4,575 vs 4,002) but slower on Gemini (7,756 vs 9,474).
   - Against TQ+, every TieredQ variant is faster at every budget: 2.5-8× on batch.
2. **4 bits/dim:** fine tiers add about +0.004-0.005 R@10 at equal disk.
   - They cost about 15-20% batch QPS (OpenAI 2,467 → 2,005).
   - The k=100 small-batch path costs more: at nominal 4 bits on OpenAI, 8t·b1 p50@100 rises 1.61 → 3.71 ms and
     QPS@100 8t·b32 falls 2,140 → 1,355.
   - RaBitQ stays 2.1× (OpenAI) to 2.5× (Gemini) faster on batch than the default, at 0.019-0.031 lower R@10.
3. **8 bits/dim:** without fine tiers TieredQ **loses** recall to RaBitQ at equal disk (0.9957 vs 0.9967 on OpenAI,
   0.9920 vs 0.9955 on Gemini). Default and exact win it back (0.9976-0.9988). RaBitQ remains 3.5-6× faster on batch
   and 8-19× faster on single query.
4. **Steady RAM:** fine tiers keep disk bytes unchanged, but not RAM.
   - Default vs off at the nominal budget, steady RSS after reload: OpenAI 4-bit 987 vs 851 B/vector (+16%) and
     8-bit 1,756 vs 1,624 (+8%). Gemini 4-bit 465 vs 422 (+10%) and 8-bit 855 vs 808 (+6%). Exact is close to default.
   - The library's accounted serving bytes catch only part of this. OpenAI 4-bit default is 859.5 B accounted vs
     987 B RSS.
   - So the equal-RAM matches in #4 (`match_memory` ram/peak targets) will give the default fewer code bits than the
     equal-disk ones here.
   - **Equal steady RAM** (`equal-ram-memory.csv`, seed 0): off was probed at the default's 4-bit steady RSS.
     - OpenAI: off at 7,168 code bits (980 B) reaches R@10 0.9875 vs the default's 0.9878. That's a tie.
     - Gemini: off at 3,392 code bits (464 B) reaches 0.9816 vs 0.9841. The default still leads.
     - At 8 bits/dim the default wins at any measure: off is capped at 8 bits/dim.
     - Both results predate the RAM fix, which removes the premium.

## Recommendation for the full #4 run

- **Headline:** report TieredQ default (fine tiers on) at every budget. It is the shipped default, and it is never
  worse on recall than off at equal disk.
- **No extra rows at 1-2 bits/dim.** Off is identical there, and exact is dominated.
- **4 bits/dim:** add a TieredQ off row in the timing tables. Off is the faster TieredQ config by 15-20% batch QPS
  and up to 2.3× k=100 single-query, at about −0.004 R@10. Without it, #4 would understate the best TieredQ speed
  against RaBitQ. With the RAM fix (fine-ram-20261105), off no longer saves RAM. Run #4 on a library that includes
  that fix, or the default's equal-RAM rows carry a 6-13% RAM premium at 100K.
- **8 bits/dim** (if #4 reports it): add an exact row. It is the fastest TieredQ batch config, about 1.8× default,
  at the same recall, but with 2.3× worse single-query latency. Show off too, because it is the case where TieredQ
  loses recall to RaBitQ at equal disk.
- **Exact below 8 bits/dim:** don't add it. It is slower than default on both batch and latency at equal recall.
- **Report plainly:** at equal disk on this AVX-512 host, RaBitQ beats every TieredQ variant on single-query latency
  from 2 bits/dim up (1.4-19×).
  - On batch, RaBitQ is 1.2× faster at 2 bits on Gemini, 2.1-2.5× at 4 bits and 3.5-6× at 8 bits.
  - TieredQ's wins are recall (+0.02-0.12 R@10 at 1-4 bits/dim), speed at 1 bit/dim, 2-bit batch on OpenAI, and
    every TQ+ comparison.
