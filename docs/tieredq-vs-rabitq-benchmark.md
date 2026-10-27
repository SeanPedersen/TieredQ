# TieredQ fast versus RaBitQ: recall, serving storage, and CPU search cost

An audited flat-search benchmark on Hartmut · main measurements: 26–27 October 2026

## Findings

TieredQ’s strongest main-grid result is SIFT1M: at every measured nominal budget it delivers higher recall with less serving storage and higher batch throughput. GloVe also shows useful storage savings. The added OpenAI and Gemini measurements compare actual accounted storage directly, resolving gaps left by the nominal-budget grid. At all six tested OpenAI/Gemini storage caps, TieredQ achieves higher mean recall with 0.37–0.87% less accounted serving storage. Recall gains range from 0.96 to 3.14 percentage points. Search throughput remains a separate tradeoff.

- **SIFT:** main-grid recall gains are 8.88–24.81 percentage points. Serving storage is 1.5–20.5% lower; eight-thread batch-32 throughput is 1.09–1.17× RaBitQ’s. The smaller timing gains need caution because the runs were on different days.
- **GloVe:** TieredQ’s 2- and 4-bit main points save 19.1% and 20.9% storage respectively, with slightly higher mean recall. Its 4-bit batch throughput is lower; its 1-bit mean recall is slightly below RaBitQ’s.
- **Actual stored-bit comparison:** the OpenAI point just below RaBitQ’s 2-bit storage cap reaches **0.9326 ± 0.0011** recall at **3,215.7 stored bits/vector**, versus RaBitQ’s **0.9013 ± 0.0022** at **3,232.6**. All six storage-cap comparisons are reported below.
- **Serving speed:** at the low storage caps, TieredQ’s observed eight-thread batch-32 QPS is 1.69–1.75× RaBitQ’s. At the middle and high caps, RaBitQ is 1.46–4.09× faster, with lower recall. These separately timed runs do not establish equal-recall throughput superiority.
- **Construction:** TieredQ takes longer to build at every main-grid point. Encoding alone has a different tradeoff: RaBitQ leads at 1 bit; TieredQ leads at 2 and 4 bits.

## Experimental design

Both codecs use native C++ kernels in a flat exhaustive top-10 search, with no reranking or ANN graph/partition index. Native bounds may avoid parts of a candidate’s score calculation; “exhaustive” here describes the flat candidate space, not necessarily full arithmetic for every candidate. RaBitQ uses the direct-flat FHT adapter, implicit row IDs, high-accuracy scoring, and native extra-bit bounds.

The host is **Hartmut, Intel Core i9-7980XE**, pinned to logical CPUs **0–7**. Builds use **GCC 16.2.1**, C++20, and **`-O3 -march=native -ffp-contract=off`**. Python 3.12 orchestrates the native measurements; the retained environment uses uv. No web sources or newly downloaded datasets are needed to interpret these results.

| Dataset | Base vectors | Original dimensions | Queries | Distance / query semantics |
| --- | --- | --- | --- | --- |
| SIFT1M | 1,000,000 | 128 | 1,000 | L2; benchmark query vectors |
| GloVe-200 | 100,000 | 200 | 1,000 | Normalized cosine |
| OpenAI3-1536 | 100,000 | 1,536 | 1,000 | Normalized cosine |
| DBpedia Gemini-768 | 99,000 | 768 | 1,000 | Normalized cosine; disjoint held-out document vectors |

Gemini’s queries are held-out document embeddings, not independently embedded search queries. Its [manifest](../benchmark/results/issue4-range-refresh-20261027/output/gemini-manifest.json) records the source revision, split seed, FP32 normalization, exhaustive FP32 dot-product truth, and corpus/query row hashes.

The main grid uses average code budgets of **1, 2, and 4 bits per original dimension**. TieredQ can allocate **0–8 bits to individual coordinates**. Its SIFT adapter uses a metric-preserving, query-conditioned L2 lift into an extra dimension, calibrated from base vectors; the code budget still refers to the original 128 dimensions. Base transformation is included in build cost, query transformation in timed search, and transform state in storage. This is a comparison of complete native adapters, not isolated quantizer arithmetic.

Each configuration evaluates **1,000 queries**, recall seeds **0/1/2**, thread settings **1/8**, and query batches **1/32**. Recall@10 is the mean fraction of exact top-10 neighbors returned. Tables report the mean and **population standard deviation** across the three seeds. Seeds 1 and 2 are recall-only runs, evaluated with eight threads and batch 32, whose recall values are copied across scheduling rows; seed 0 is measured separately for each schedule. Small scheduling-dependent recall differences can therefore remain in the summaries.

Timing and storage use **seed 0**. The harness performs **three single-query warmup calls**, then **three repeats** over the query set. QPS is total queries processed divided by accumulated search-call time. Latency percentiles are computed from search-call durations; at batch 32 they are whole-batch response times, weighted by queries in each batch, and are not per-query latency. The final call contains the remaining queries when the count is not divisible by the batch size.

**Serving storage** is the larger of serialized size and the adapter’s accounted serving memory, amortized over base vectors, including model state and the SIFT transform where applicable. For TieredQ this uses a reloaded fast index’s serving state. It is neither raw packed-code size nor process RSS, and excludes transient search buffers, allocator overhead, and training-only state. Stored bits/vector are exactly eight times the bytes/vector reported below. Equal nominal bits do not imply equal measured storage.

Construction uses eight native threads, with external BLAS thread pools limited to one. TieredQ trains on a seeded sample of up to 32,768 base vectors. Build time includes preprocessing, training, and encoding, but excludes dataset loading, truth generation, serialization/storage inspection, and compilation. Encode throughput is base vectors divided by the adapter’s encode timer; for TieredQ this surrounds index addition. It does not include all training or preprocessing costs.

RaBitQ timings were reused from **26 October**; TieredQ main timings are from **27 October**. Dataset and truth hashes, native RaBitQ adapter hashes, and timing-code hashes match. Nevertheless, these are separate runs on a **shared host**: frequency changes, contention, and cache state can cause between-run variation. Three repeats and recall-seed deviations do not provide confidence intervals for timing differences.

## Main results: recall, actual storage, and batch throughput

All main-grid points are shown. Recall and storage in this table use the eight-thread, batch-32 configuration. QPS columns retain the thread counts separately; both use batch 32. B/vector means actual accounted serving bytes per base vector. Values are rounded only for display.

| Dataset | Codec | Nominal bits/dim | Recall@10 mean ± SD | B/vector | QPS · 1 thread | QPS · 8 threads |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | TieredQ fast | 1 | 0.4976 ± 0.0026 | 27.574 | 372 | 2,690 |
| SIFT1M | RaBitQ | 1 | 0.2495 ± 0.0047 | 28.001 | 300 | 2,307 |
| SIFT1M | TieredQ fast | 2 | 0.7639 ± 0.0014 | 41.320 | 343 | 2,270 |
| SIFT1M | RaBitQ | 2 | 0.5484 ± 0.0025 | 52.001 | 249 | 1,976 |
| SIFT1M | TieredQ fast | 4 | 0.9338 ± 0.0008 | 73.343 | 254 | 1,666 |
| SIFT1M | RaBitQ | 4 | 0.8450 ± 0.0030 | 84.001 | 231 | 1,535 |
| GloVe-200 | TieredQ fast | 1 | 0.4044 ± 0.0056 | 42.878 | 3,248 | 24,353 |
| GloVe-200 | RaBitQ | 1 | 0.4078 ± 0.0020 | 44.012 | 2,446 | 16,371 |
| GloVe-200 | TieredQ fast | 2 | 0.6665 ± 0.0038 | 67.939 | 2,390 | 16,931 |
| GloVe-200 | RaBitQ | 2 | 0.6597 ± 0.0032 | 84.012 | 1,984 | 12,753 |
| GloVe-200 | TieredQ fast | 4 | 0.9015 ± 0.0023 | 117.042 | 1,634 | 10,887 |
| GloVe-200 | RaBitQ | 4 | 0.8818 ± 0.0008 | 148.012 | 1,718 | 12,131 |
| OpenAI3-1536 | TieredQ fast | 1 | 0.8950 ± 0.0016 | 318.978 | 559 | 3,833 |
| OpenAI3-1536 | RaBitQ | 1 | 0.8140 ± 0.0014 | 204.070 | 615 | 4,118 |
| OpenAI3-1536 | TieredQ fast | 2 | 0.9468 ± 0.0004 | 493.905 | 240 | 1,234 |
| OpenAI3-1536 | RaBitQ | 2 | 0.9013 ± 0.0022 | 404.070 | 598 | 4,589 |
| OpenAI3-1536 | TieredQ fast | 4 | 0.9839 ± 0.0011 | 887.805 | 213 | 1,086 |
| OpenAI3-1536 | RaBitQ | 4 | 0.9674 ± 0.0007 | 788.070 | 539 | 3,917 |
| DBpedia Gemini-768 | TieredQ fast | 1 | 0.8344 ± 0.0006 | 154.759 | 1,203 | 9,015 |
| DBpedia Gemini-768 | RaBitQ | 1 | 0.7216 ± 0.0034 | 108.044 | 1,209 | 7,675 |
| DBpedia Gemini-768 | TieredQ fast | 2 | 0.9243 ± 0.0009 | 228.472 | 545 | 2,552 |
| DBpedia Gemini-768 | RaBitQ | 2 | 0.8549 ± 0.0017 | 212.044 | 1,119 | 8,394 |
| DBpedia Gemini-768 | TieredQ fast | 4 | 0.9792 ± 0.0009 | 425.247 | 435 | 2,253 |
| DBpedia Gemini-768 | RaBitQ | 4 | 0.9516 ± 0.0006 | 404.044 | 1,152 | 8,284 |

![Recall against accounted serving storage for all four datasets](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/recall-storage-all.png)

*Figure 1. All main-grid and supplemental points, using eight-thread, batch-32 serving storage and recall. Storage is shown in actual accounted bits/vector. Stars M1/M2/M4 mark TieredQ points selected below the corresponding RaBitQ storage caps. Lines connect all measured points in code-budget order as visual guides; no unmeasured results are inferred. [Vector PDF](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/recall-storage-all.pdf).*

## Single-query latency: batch 1

Each cell is **p50 / p95 milliseconds** for a complete single-query search call. These are seed-0 observations, not timing confidence intervals. TieredQ can parallelize a single-query scan; RaBitQ’s adapter parallelizes across queries, so setting eight threads does not accelerate its batch-1 call in the same way. Eight configured threads therefore do not imply eight active workers for both implementations.

| Dataset | Bits/dim | TieredQ · 1 thread | RaBitQ · 1 thread | TieredQ · 8 threads | RaBitQ · 8 threads |
| --- | --- | --- | --- | --- | --- |
| SIFT1M | 1 | 2.719 / 3.005 | 3.249 / 3.512 | 0.452 / 0.482 | 3.284 / 3.533 |
| SIFT1M | 2 | 9.032 / 10.309 | 3.965 / 4.586 | 1.382 / 1.480 | 4.023 / 4.657 |
| SIFT1M | 4 | 15.406 / 18.370 | 4.279 / 5.419 | 2.427 / 2.875 | 4.305 / 5.460 |
| GloVe-200 | 1 | 0.289 / 0.328 | 0.407 / 0.442 | 0.097 / 0.114 | 0.407 / 0.454 |
| GloVe-200 | 2 | 0.406 / 0.452 | 0.515 / 0.610 | 0.119 / 0.142 | 0.519 / 0.628 |
| GloVe-200 | 4 | 0.620 / 0.680 | 0.583 / 0.756 | 0.167 / 0.185 | 0.587 / 0.770 |
| OpenAI3-1536 | 1 | 2.072 / 2.349 | 1.582 / 1.825 | 0.881 / 0.983 | 1.592 / 1.853 |
| OpenAI3-1536 | 2 | 9.843 / 11.178 | 1.659 / 1.902 | 2.597 / 2.748 | 1.574 / 1.824 |
| OpenAI3-1536 | 4 | 11.303 / 12.764 | 1.827 / 1.985 | 3.042 / 3.197 | 1.840 / 2.009 |
| DBpedia Gemini-768 | 1 | 0.868 / 0.991 | 0.816 / 0.912 | 0.286 / 0.308 | 0.815 / 0.919 |
| DBpedia Gemini-768 | 2 | 4.582 / 5.283 | 0.860 / 0.958 | 1.007 / 1.093 | 0.859 / 0.971 |
| DBpedia Gemini-768 | 4 | 7.171 / 8.140 | 0.902 / 1.011 | 1.553 / 1.634 | 0.860 / 0.961 |

SIFT illustrates why latency and throughput must be separate: TieredQ’s higher-budget single-query calls are slower with one thread, despite its batch-32 throughput advantage. GloVe favors TieredQ latency at 1 and 2 bits with one thread and at every budget with eight threads. On OpenAI and Gemini, the higher-budget TieredQ single-query paths are slower than RaBitQ even with eight threads. These comparisons retain the different recall and storage levels shown above.

![Single-query p50 and p95 latency against recall, separated by thread count](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/latency-recall-all.png)

*Figure 2. Batch-1 latency for all measured points, with thread settings in separate rows. Solid lines show p50 and dashed lines p95, connecting all measured points in code-budget order. Supplemental filled stars show p50 and outlined stars p95; dotted vertical segments connect those two observed percentiles. [Vector PDF](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/latency-recall-all.pdf).*

## Batch throughput: batch 32

The QPS columns in the main table measure completed queries in batches, rather than single-query response time. The following ratios summarize the same data; values above one favor TieredQ. Neither column substitutes for a comparison at equal storage and recall.

| Dataset | Bits/dim | TieredQ / RaBitQ QPS · 1 thread | TieredQ / RaBitQ QPS · 8 threads |
| --- | --- | --- | --- |
| SIFT1M | 1 | 1.24× | 1.17× |
| SIFT1M | 2 | 1.38× | 1.15× |
| SIFT1M | 4 | 1.10× | 1.09× |
| GloVe-200 | 1 | 1.33× | 1.49× |
| GloVe-200 | 2 | 1.20× | 1.33× |
| GloVe-200 | 4 | 0.95× | 0.90× |
| OpenAI3-1536 | 1 | 0.91× | 0.93× |
| OpenAI3-1536 | 2 | 0.40× | 0.27× |
| OpenAI3-1536 | 4 | 0.40× | 0.28× |
| DBpedia Gemini-768 | 1 | 1.00× | 1.17× |
| DBpedia Gemini-768 | 2 | 0.49× | 0.30× |
| DBpedia Gemini-768 | 4 | 0.38× | 0.27× |

![Batch-32 throughput against recall, separated by thread count](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/qps-recall-all.png)

*Figure 3. Batch-32 throughput for all measured points, with thread settings in separate rows. Stars mark the six storage-selected TieredQ configurations. Lines connect all measured points in code-budget order as visual guides. Timing comparisons retain the separate shared-host-run limitation. [Vector PDF](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/qps-recall-all.pdf).*

## Comparisons under measured storage caps

Each storage cap is the actual size of a measured RaBitQ point. For each codec, select the **highest mean recall among all measured points fitting that cap**, including the six supplemental TieredQ points. Storage and recall use the eight-thread, batch-32 schedule. Entries give **nominal code-bit budget/vector**, actual serving bytes/vector, and mean recall. A **†** marks a supplemental point. Nominal code budgets exclude representation padding and serving overhead; actual stored bits are eight times the bytes shown. “None” means no measured point fits. No unmeasured operating point is estimated.

| Dataset | Cap · B/vector | TieredQ: code bits; bytes; recall | RaBitQ: code bits; bytes; recall |
| --- | --- | --- | --- |
| SIFT1M | 28.001 | 128; 27.574 B; 0.4976 | 128; 28.001 B; 0.2495 |
| SIFT1M | 52.001 | 256; 41.320 B; 0.7639 | 256; 52.001 B; 0.5484 |
| SIFT1M | 84.001 | 512; 73.343 B; 0.9338 | 512; 84.001 B; 0.8450 |
| GloVe-200 | 44.012 | 200; 42.878 B; 0.4044 | 200; 44.012 B; 0.4078 |
| GloVe-200 | 84.012 | 400; 67.939 B; 0.6665 | 400; 84.012 B; 0.6597 |
| GloVe-200 | 148.012 | 800; 117.042 B; 0.9015 | 800; 148.012 B; 0.8818 |
| OpenAI3-1536 | 204.070 | 833†; 202.410 B; 0.8236 | 1,536; 204.070 B; 0.8140 |
| OpenAI3-1536 | 404.070 | 2,411†; 401.961 B; 0.9326 | 3,072; 404.070 B; 0.9013 |
| OpenAI3-1536 | 788.070 | 5,312†; 783.800 B; 0.9780 | 6,144; 788.070 B; 0.9674 |
| DBpedia Gemini-768 | 108.044 | 493†; 107.642 B; 0.7498 | 768; 108.044 B; 0.7216 |
| DBpedia Gemini-768 | 212.044 | 1,114†; 210.208 B; 0.8864 | 1,536; 212.044 B; 0.8549 |
| DBpedia Gemini-768 | 404.044 | 2,880†; 401.245 B; 0.9742 | 3,072; 404.044 B; 0.9516 |

These comparisons maximize recall subject to each measured cap; they do not require equal recall or identical byte counts. In particular, “below the cap” does not by itself demonstrate better compression: the measured recall must also be considered. The supplemental selection protocol and paired results are given below.

### Minimum measured storage reaching Recall@10 ≥0.95

The threshold applies to the three-seed **mean** recall, not a confidence bound. Selection now includes the original main grid **and all six supplemental points**. These are minima among measured configurations, not the codec’s globally minimum required storage.

| Dataset | Codec | Code bits/vector | Minimum B/vector | Recall@10 | QPS · 8 threads, batch 32 |
| --- | --- | --- | --- | --- | --- |
| SIFT1M | TieredQ fast | Not reached | — | — | — |
| SIFT1M | RaBitQ | Not reached | — | — | — |
| GloVe-200 | TieredQ fast | Not reached | — | — | — |
| GloVe-200 | RaBitQ | Not reached | — | — | — |
| OpenAI3-1536 | TieredQ fast | 5,312† | 783.800 | 0.9780 | 1,067 |
| OpenAI3-1536 | RaBitQ | 6,144 | 788.070 | 0.9674 | 3,917 |
| DBpedia Gemini-768 | TieredQ fast | 2,880† | 401.245 | 0.9742 | 2,027 |
| DBpedia Gemini-768 | RaBitQ | 3,072 | 404.044 | 0.9516 | 8,284 |

Neither codec reaches the threshold on SIFT or GloVe within the measured budgets. OpenAI3-1536: TieredQ has the smaller measured point reaching the threshold (TieredQ 783.800 versus RaBitQ 788.070 B/vector). DBpedia Gemini-768: TieredQ has the smaller measured point reaching the threshold (TieredQ 401.245 versus RaBitQ 404.044 B/vector). These results replace the earlier main-grid-only threshold comparison. The selected points have different recall above 0.95 and do not establish matched-recall serving performance.

## Construction and encoding

| Dataset | Bits/dim | TieredQ build · s | RaBitQ build · s | TieredQ encode · vectors/s | RaBitQ encode · vectors/s |
| --- | --- | --- | --- | --- | --- |
| SIFT1M | 1 | 2.244 | 0.131 | 1,566,583 | 11,178,654 |
| SIFT1M | 2 | 1.934 | 1.525 | 1,118,550 | 674,236 |
| SIFT1M | 4 | 3.123 | 2.717 | 485,369 | 374,439 |
| GloVe-200 | 1 | 0.550 | 0.038 | 829,839 | 3,265,236 |
| GloVe-200 | 2 | 0.619 | 0.348 | 622,051 | 294,455 |
| GloVe-200 | 4 | 0.713 | 0.630 | 349,304 | 161,027 |
| OpenAI3-1536 | 1 | 14.180 | 0.115 | 105,238 | 1,451,067 |
| OpenAI3-1536 | 2 | 14.834 | 2.502 | 75,108 | 40,733 |
| OpenAI3-1536 | 4 | 16.043 | 4.146 | 42,953 | 24,405 |
| DBpedia Gemini-768 | 1 | 3.758 | 0.076 | 233,675 | 2,405,830 |
| DBpedia Gemini-768 | 2 | 3.930 | 1.151 | 183,197 | 87,972 |
| DBpedia Gemini-768 | 4 | 4.484 | 2.042 | 95,537 | 49,132 |

TieredQ’s training cost outweighs its faster higher-budget encoding in these full builds. Encoding throughput should therefore not be used as a proxy for total construction time. These are seed-0 adapter measurements with no construction-time variance estimate, and do not measure incremental update or concurrent ingestion performance.

## Range-bound checkpoint ablation

For the original main grid, all **48 paired ID sets matched exactly**, with identical recall and serving storage between the bounds-enabled run and the control. The control bypasses only the automatic checkpoint dispatch condition. For eligible one-thread, single-query points, bounds/control QPS ratios span **1.09–1.49×**.

| Dataset | Bits/dim | Bounds/control QPS · 1 thread, batch 1 |
| --- | --- | --- |
| SIFT1M | 2 | 1.35× |
| SIFT1M | 4 | 1.49× |
| OpenAI3-1536 | 2 | 1.30× |
| OpenAI3-1536 | 4 | 1.39× |
| DBpedia Gemini-768 | 2 | 1.24× |
| DBpedia Gemini-768 | 4 | 1.09× |

The dispatch requires AVX-512 and a supported code-lookup layout with more than four tiers. **Batch-32 searches do not activate this checkpoint.** Differences in batch-32 controls, or other inactive configurations, reflect between-run variation rather than a checkpoint speedup. The supplemental points were not included in this ablation. The ablation supports parity on these tested outputs; it is not a proof for arbitrary data. See [paired results](../benchmark/results/issue4-range-refresh-20261027/output/ablation-summary.json), [exact ID parity](../benchmark/results/issue4-range-refresh-20261027/ablation/parity.json), and [control variant](../benchmark/results/issue4-range-refresh-20261027/ablation/variant.json).

## Supplemental comparisons at actual RaBitQ storage caps

Six TieredQ fast points fill the missing actual-storage comparisons on OpenAI and Gemini. **M1, M2, and M4** mean that TieredQ was selected below the serving storage of the corresponding RaBitQ 1-, 2-, or 4-bit point. They are not TieredQ nominal budgets.

All supplemental measurements use the original measured TieredQ source, verified byte-identical inputs, the isolated exact-code-budget adapter, the audited compiler policy, CPUs 0–7, 1,000 queries, and recall seeds 0/1/2. Timing and serving storage are seed 0, with threads 1/8 and batches 1/32 and the original warmups/repeats. Before the five-point extension, one-bit controls on **both datasets** reproduced the archived seed-0 batch-32 top-10 IDs and serving storage exactly. Native quantizer and search kernels were unchanged.

**Selection uses storage alone, before candidate recall is measured.** The target is 99.0–99.5% of the baseline cap. Integer code budgets are selected using a storage-based initial estimate followed by bounded bisection. Fast-layout changes can make storage discrete; if the target band is not found, the protocol retains the largest observed serving size below the cap, without optimizing recall. Every attempt and any fallback are recorded. **5 of six** selected points fall in the target band. The original OpenAI M2 selection and the five-point extension have separate logs and metadata; the main audit and original plot artifacts are unchanged; the report figures have been regenerated to include all measured points.

| Dataset / cap | Codec | B/vector | Stored bits/vector | Recall@10 mean ± SD | Batch-32 QPS · 1 thread | Batch-32 QPS · 8 threads |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI3-1536 M1 | TieredQ fast | 202.410 | 1,619.3 | 0.8236 ± 0.0028 | 993 | 7,190 |
| OpenAI3-1536 M1 | RaBitQ | 204.070 | 1,632.6 | 0.8140 ± 0.0014 | 615 | 4,118 |
| OpenAI3-1536 M2 | TieredQ fast | 401.961 | 3,215.7 | 0.9326 ± 0.0011 | 289 | 1,396 |
| OpenAI3-1536 M2 | RaBitQ | 404.070 | 3,232.6 | 0.9013 ± 0.0022 | 598 | 4,589 |
| OpenAI3-1536 M4 | TieredQ fast | 783.800 | 6,270.4 | 0.9780 ± 0.0004 | 199 | 1,067 |
| OpenAI3-1536 M4 | RaBitQ | 788.070 | 6,304.6 | 0.9674 ± 0.0007 | 539 | 3,917 |
| DBpedia Gemini-768 M1 | TieredQ fast | 107.642 | 861.1 | 0.7498 ± 0.0006 | 1,811 | 12,957 |
| DBpedia Gemini-768 M1 | RaBitQ | 108.044 | 864.4 | 0.7216 ± 0.0034 | 1,209 | 7,675 |
| DBpedia Gemini-768 M2 | TieredQ fast | 210.208 | 1,681.7 | 0.8864 ± 0.0016 | 816 | 5,734 |
| DBpedia Gemini-768 M2 | RaBitQ | 212.044 | 1,696.4 | 0.8549 ± 0.0017 | 1,119 | 8,394 |
| DBpedia Gemini-768 M4 | TieredQ fast | 401.245 | 3,210.0 | 0.9742 ± 0.0015 | 435 | 2,027 |
| DBpedia Gemini-768 M4 | RaBitQ | 404.044 | 3,232.4 | 0.9516 ± 0.0006 | 1,152 | 8,284 |

| Dataset / cap | TieredQ code bits/vector | Storage saving | Recall gain · percentage points | TieredQ / RaBitQ QPS · 8 threads, batch 32 |
| --- | --- | --- | --- | --- |
| OpenAI3-1536 M1 | 833 | 0.81% | +0.96 | 1.75× |
| OpenAI3-1536 M2 | 2,411 | 0.52% | +3.13 | 0.30× |
| OpenAI3-1536 M4 | 5,312 | 0.54% | +1.06 | 0.27× |
| DBpedia Gemini-768 M1 | 493 | 0.37% | +2.82 | 1.69× |
| DBpedia Gemini-768 M2 | 1,114 | 0.87% | +3.14 | 0.68× |
| DBpedia Gemini-768 M4 | 2,880 | 0.69% | +2.26 | 0.24× |

![Measured OpenAI and Gemini recall against stored bits, including all six storage-selected points](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/recall-storage-matched.png)

*Figure 4. Discrete measured points; stars mark the supplemental configurations. M1/M2/M4 refer to RaBitQ storage caps. Error bars are population standard deviations over recall seeds. Storage is seed 0 from the eight-thread, batch-32 row. Lines connect all measured points in code-budget order as visual guides; no unmeasured results are inferred. A [vector PDF](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/recall-storage-matched.pdf) is available.*

All six comparisons show higher measured mean recall at slightly lower actual serving storage. This tests the six explicit caps; it does not establish dominance between them or at equal recall. The QPS ratios retain the separate shared-host timing runs, so small differences are not stable speed claims.

The following table reports each supplemental TieredQ point’s complete single-query latency and construction cost. RaBitQ’s matching nominal-budget latency and construction measurements remain in the main tables above.

| TieredQ point | Batch-1 p50 / p95 · ms, 1 thread | Batch-1 p50 / p95 · ms, 8 threads | Build · s | Encode · vectors/s |
| --- | --- | --- | --- | --- |
| OpenAI3-1536 M1 | 1.057 / 1.213 | 0.491 / 0.530 | 13.669 | 181,756 |
| OpenAI3-1536 M2 | 8.270 / 9.321 | 2.185 / 2.301 | 14.552 | 83,780 |
| OpenAI3-1536 M4 | 13.255 / 14.474 | 3.194 / 3.334 | 15.794 | 48,040 |
| DBpedia Gemini-768 M1 | 0.545 / 0.597 | 0.221 / 0.239 | 3.604 | 355,078 |
| DBpedia Gemini-768 M2 | 1.260 / 1.435 | 0.406 / 3.192 | 3.815 | 194,522 |
| DBpedia Gemini-768 M4 | 6.612 / 7.415 | 1.494 / 1.562 | 4.305 | 108,172 |

Gemini M2 records an eight-thread single-query p95 of **3.192 ms** alongside a p50 of **0.406 ms**. This tail result is retained as observed; its cause was not isolated.

The first OpenAI M2 point is retained in [matched-storage/results.csv](../benchmark/results/issue4-range-refresh-20261027/matched-storage/results.csv), with its [metadata](../benchmark/results/issue4-range-refresh-20261027/matched-storage/meta.json), [selection](../benchmark/results/issue4-range-refresh-20261027/matched-storage/selection.json), and [reproduction instructions](../benchmark/results/issue4-range-refresh-20261027/matched-storage/README.md). The remaining five points are in [storage-sweep/results.csv](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/results.csv), with [metadata](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/meta.json), [control parity](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/controls.json), [audit](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/audit.json), [per-point storage selections and reproduction instructions](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/README.md), and [job log](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/job.log). The extension contains 60 rows, including 20 timed seed-0 rows. Its audit is separate from the original 288-row audit. RaBitQ timings are still reused from 26 October; all new TieredQ runs are dated in their metadata.

## Scope, provenance, and reproduction

The measured TieredQ revision is **`575e167a0ad66920119bd0d0836b6c0c92c0a4e5`**. The checkout at report preparation, **`1363199`**, changes the public API: fast is now the default, `mode="full"` explicitly selects the compact full representation, and the boolean API has been removed. For current usage, `TieredQIndex(dimensions=1536, code_bits=3072)` selects fast mode by default; `mode="full"` selects full mode. See the [current README](../README.md). These usage forms describe the current API, not the syntax used by the measured adapter. RaBitQ’s pinned revision is `ced3763857e7e4e48553951bf66fabc8517bc4f0`.

The primary comparison excludes the separate **100-query normal-mode control**. Its sample size, seed coverage, and representation differ, so it cannot establish compact-mode performance for this main evaluation.

The benchmark covers one CPU, one implementation pairing, four corpora, three main budgets plus six supplemental OpenAI/Gemini points, two thread settings, and two batch sizes. It does not establish performance for ANN indexes, other architectures, larger embedding corpora, semantic search quality, cold starts, concurrent request traffic, peak RSS, or matched-recall operating points between measured budgets. Recall-seed standard deviations quantify seed variation only; three seeds do not support broad significance claims. No interpolated operating points are reported.

The [experiment overview](../benchmark/results/issue4-range-refresh-20261027/README.md) describes the retained source export, scripts, validation logs, and remote workspace. The [audit](../benchmark/results/issue4-range-refresh-20261027/output/audit.json) passes all 288 main rows, including 96 timed seed-0 rows, matching input hashes and reused-code hashes. [Row provenance](../benchmark/results/issue4-range-refresh-20261027/output/row-provenance.json) identifies the origin of every row:

- TieredQ main rows are fresh; Gemini RaBitQ recall seeds 1/2 are also fresh.
- SIFT, GloVe, and OpenAI RaBitQ seed-0 measurements come from [issue4-results.csv](../benchmark/results/issue4-range-refresh-20261027/reuse/issue4-results.csv); recall seeds 1/2 come from [curve-results.csv](../benchmark/results/issue4-range-refresh-20261027/reuse/curve-results.csv), with timing fields explicitly absent.
- Gemini RaBitQ seed-0 measurements come from [gemini-results.csv](../benchmark/results/issue4-range-refresh-20261027/reuse/gemini-results.csv). Baseline metadata and dataset verification are retained in the [reuse directory](../benchmark/results/issue4-range-refresh-20261027/reuse/).

The authoritative numerical source is [results.csv](../benchmark/results/issue4-range-refresh-20261027/output/results.csv). [Metadata](../benchmark/results/issue4-range-refresh-20261027/output/meta.json) records compiler commands, source hashes, revisions, CPU details, affinity, and timing dates. [Summary JSON](../benchmark/results/issue4-range-refresh-20261027/output/summary.json) retains derived operating points; its entries combine batch-32 throughput with batch-1 latency and use the maximum recorded storage across recall seeds within each batch-32 schedule. This report derives tables directly from the CSV and labels schedules explicitly, rather than treating those summary fields as a single timed call.

To reproduce the original experiment, restore the retained `source.tar.gz` into the recorded remote workspace, restore the pinned Python/native dependencies, verify cached input hashes against [inputs.json](../benchmark/results/issue4-range-refresh-20261027/output/inputs.json) and the Gemini manifest, and run [run_refresh.py](../benchmark/results/issue4-range-refresh-20261027/run_refresh.py) under CPU affinity 0–7. The script records existing host-specific cache paths, so reproductions elsewhere must map those paths to byte-identical inputs. [finalize.py](../benchmark/results/issue4-range-refresh-20261027/finalize.py) merges the explicitly reused RaBitQ baselines and audits the rows; [run_ablation.py](../benchmark/results/issue4-range-refresh-20261027/run_ablation.py) generates the independent no-range control. A fully fresh RaBitQ rerun would be a new experiment, not a reconstruction of the saved timings.

All numerical tables in this report are calculated from the retained CSV rows and checked independently by [verify_report.py](../benchmark/results/issue4-range-refresh-20261027/storage-sweep/verify_report.py); recall deviations use the population definition and all ratios use unrounded inputs. The linked figures were inspected for labels, datasets, schedules, and consistency with those rows. No original benchmark artifacts were overwritten.
