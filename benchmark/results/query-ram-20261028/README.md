# Serving RAM: fold the centroid lookup projection

Issue #4 follow-up (tasks 4a and 4b). Goal: lower TieredQ's steady and peak serving RAM without losing QPS, latency or recall, so that recall per peak RAM improves against RaBitQ.

Hartmut (Intel i9-7980XE, AVX-512), GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. Timing A/Bs ran on CPUs 0–7. The issue #4 equal-memory job was paused with SIGSTOP during each A/B and resumed afterwards. Indexes are the seed-0 TieredQ 1/2/4-bit OpenAI, Gemini and GloVe indexes at 100K vectors (Gemini 99K), saved once and reused by every tree.

## Where TieredQ's RAM goes beyond disk

Allocation accounting of the reloaded index ([breakdown.json](breakdown.json), [breakdown.py](breakdown.py); bits/vector):

| Cell | Disk | Resident | Codes | Lookup projection panels | Basis | Centroids |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI 1-bit | 2,298 | 2,552 | 1,682 | **252** | 418 | 126 |
| Gemini 1-bit | 1,110 | 1,238 | 873 | **127** | 101 | 64 |
| GloVe 1-bit | 310 | 343 | 208 | **33** | 13 | 16 |

The 2- and 4-bit OpenAI/Gemini layouts keep less than 5 bits/vector above disk. All of the excess is `FastQueryProjection.panels`, a d × 2k FP32 table built at load. For each dimension i and centroid c it holds `centroid[c][i] + correction[i]` and `centroid[c][i] / (2·half_norm[c])`.

## 4b: fold the panels into one dot product per centroid

**Exact regeneration fails.** Rebuilding the panel values per query in the same FP32 order (a 16-lane gather from `centroids_` plus an IEEE divide per element) kept IDs byte-identical. However, one interleaved round ([ab-exact-regen-round1.jsonl](ab-exact-regen-round1.jsonl)) measured QPS of 0.84–0.90× at 1 thread and 0.60–0.75× at t8-b1 on the lookup layouts. That variant was dropped.

**Folded projection (committed).** Both outputs derive from one `q·centroid_c` dot product: `shifted = q·c + q·correction` and `scaled = q·c / (2·half_norm_c)`. It reads `centroids_` (k × d, already in the model) contiguously, i.e. half the bytes the panels needed, and keeps only the d-float `correction`. The algebra is identical but FP32 rounding changes. Those values only feed the int8 lookup scans, which are approximate already.

- **RAM:** −252 / −127 / −33 bits/vector of steady and peak RAM on OpenAI / Gemini / GloVe 1-bit. That is 9.8% / 10.1% / 9.7% of steady RAM, and 8.9% / 8.5% / 5.8% of the previous peak RAM in `issue4-memory-20261028`. GloVe 2- and 4-bit save the same 33 bits.
- **Recall and IDs** ([fold-ids-vs-source.jsonl](fold-ids-vs-source.jsonl)): recall@10 is identical on all five lookup-layout indexes, and 4,999 of 5,000 top-10 rows are byte-identical. The remaining Gemini row returns the same ten IDs in a different order.
- **Speed** ([ab-fold-table.md](ab-fold-table.md), 6 interleaved rounds, rotating order, fresh process per run): on the lookup layouts, 1-thread QPS is +3.2% to +5.4% and p50 is 3.6–5.6% lower. At t8-b1 QPS is +10–20%. t8-b32 is within the reference tree's own spread. The 2-bit OpenAI/Gemini indexes don't use this path and stay within noise.

## 4a: query-tile scratch

At 8 threads × batch 32, each document-segment worker holds a 32 × 4,096 FP32 score tile. The peak query transient is 7.5 MiB on OpenAI 2-bit and 6.3 MiB on Gemini 2-bit, and other schedules stay at 1.3 MiB or below (`query_peak.py`/`query_peak.sh`). glibc arena and mmap-threshold settings barely change it. A 2048 tile saves 2.0 MiB and a 1024 tile saves 3.0 MiB of the transient.

Uniformly smaller tiles fail the speed bar at 1 thread. At Gemini 2-bit t1-b32, QPS was 0.958× (2048) and 0.937× (1024), with every paired round below 0.961 against a 1.008 source spread. The same tiles give +11–18% QPS at 8 threads. A worker-dependent tile is evaluated separately.

## Limitations

The 100K-vector corpora make these fixed per-process costs large per vector; at 1M vectors they are about a tenth. Timing comes from a shared host with a paused (not stopped) background job, and the paired ratio ranges in the table show the noise. RAM numbers above are allocation accounting; the RSS matrix is re-measured with the final library in the equal-memory rerun.
