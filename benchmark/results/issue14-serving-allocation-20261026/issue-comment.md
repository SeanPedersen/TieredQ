## Results: lookup grouping and serving-aware allocation

**Keep the production defaults.** The tested 256-entry memory lookups preserve scores but are slower. Brandt's greedy `log2(sigma)` allocation rule provides no consistent recall improvement. A serving-aware candidate selector does improve SIFT's feasible recall under joint memory/latency ceilings, so that constrained option merits follow-up; a blanket four-bit cap does not merit adoption.

These are TieredQ experiments with fixed Gaussian codebooks, default tier rotations and the corrected SIFT adapter. They are **not a faithful Brandt implementation**. Production sources/public formats were not changed, and nothing was committed.

### 1. Lookup grouping

For narrow models, combine the two original nibble-table entries into one 256-entry int32 table indexed by the byte already stored. Preserve integer sums, scale boundaries, FP32 accumulation order, codes, padding and resident storage. For mixed/wide models, use 256-entry FP32 tables with the original codebook levels, retaining whole-coordinate groups, the same packed-code unpacking and accumulation order. This avoids attributing a precision change to layout.

Before timing: every document score and full ranking for one query per point, plus top-10 scores/IDs over all 1,000 held-out queries, matched exactly. Checks cover SIFT 64/80/128/256/512/1032 bits and GloVe/OpenAI four/eight bits per dimension. Append/load, document tails and zero-query ties also passed.

Eight-thread single-query p50 and batch-32 throughput, each averaging three repeat measurements:

| Dataset / budget | Register p50 | Byte p50 | Register QPS | Byte QPS |
| --- | ---: | ---: | ---: | ---: |
| SIFT 64 total bits | 0.326 ms | 1.112 ms | 3681 | 1018 |
| SIFT 128 total bits | 0.424 ms | 1.825 ms | 2729 | 602 |
| SIFT 256 total bits | 1.881 ms | 9.915 ms | 2135 | 1442 |
| SIFT 512 total bits | 3.170 ms | 13.902 ms | 1630 | 1080 |
| SIFT 1032 total bits | 5.776 ms | 11.877 ms | 1495 | 1132 |
| GloVe 4 bits/dim | 0.393 ms | 2.244 ms | 9149 | 5450 |
| GloVe 8 bits/dim | 1.026 ms | 2.387 ms | 7642 | 5580 |
| OpenAI 4 bits/dim | 3.923 ms | 17.879 ms | 1044 | 716 |
| OpenAI 8 bits/dim | 9.309 ms | 19.362 ms | 890 | 715 |

At SIFT 128, lookup count falls from 36 to 18, but query tables grow from 576 bytes to 19,008 bytes (the candidate retains the original tables). Stored codes remain 18 bytes/vector for 16 nominal code bytes; actual fast serving remains 27.574 B/vector. A separate one-thread diagnostic measures table construction at approximately 5.95/9.66 µs, scan at 2.98/13.17 ms, and selection at 0.115/0.100 ms for register/byte. Diagnostic counters are excluded from primary timing.

**Recommendation:** retain register lookup. This is a concrete negative result for these AVX-512 implementations on hartmut, not proof against every repacked/reordered byte layout or another CPU backend. No cache-miss hardware-counter claim is made.

### 2. Allocation and actual serving constraints

Main SIFT study: full 1M base, 1,000 queries, 32,768 base training rows, 256 centroids, seeds 0/1/2. Compare current marginal Gaussian-distortion allocation, greedy `log2(sigma) - allocated_bits` under the same eight-bit cap, and a four-bit-capped marginal candidate. The cap targets the measured narrow/wide representation transition. Seed 0 additionally covers 160/224/288/384 bits.

Three-seed Recall@10 means:

| Total bits | Current marginal | Log rule | Four-bit cap |
| ---: | ---: | ---: | ---: |
| 128 | 50.137% | 50.010% | 50.137% |
| 192 | 65.933% | 66.867% | 65.673% |
| 256 | 76.373% | 76.373% | 74.507% |
| 512 | 93.390% | 93.497% | 83.050% |
| 1032 = 8 × 129 | 98.770% | 98.770% | infeasible |

At 192 bits, current/log are nearly matched in actual memory (33.310/33.317 B/vector) and seed-0 eight-thread p50 latency (about 1.48 ms), with a 0.933-point recall gain for log. The gain is not consistent at other budgets.

At seed-0 160 bits, allocation changes routing: current has a five-bit tier, 29.306 B/vector and 1.337 ms p50; log stays narrow, 31.583 B/vector and 0.508 ms. Recall rises from 58.96% to 59.73%, but memory increases. At 256 bits, the four-bit cap uses **44.60 versus 41.32 B/vector** and loses about 1.87 recall points. Nominal bits conceal tier-group padding and layout costs.

A simple serving-aware selector minimizes disjoint-base reconstruction MSE among measured policy/budget candidates subject to actual memory and latency ceilings. It uses 2,048 base calibration rows disjoint from codec training and 32 **base-vector query proxies** for cost measurements. Held-out queries and exact truth are never used for fitting/selection.

Under the same approximately **37.315 B/vector and 1 ms ceilings**, the measured current marginal grid can use its narrow 128-bit point: 50.137% three-seed recall, 27.574 B/vector, seed-0 p50 about 0.436 ms. The selector chooses cap-4 at 192 bits: **65.673%** three-seed recall, **35.585–36.585 B/vector**, and seed-0 p50 about **0.601 ms**. Both fit the same actual resource ceilings; consumption is not exactly equal. Independent base-proxy cost calibration selects this candidate without labels. At 41.32 B/vector/1 ms, a seed-0-only cap-4 224-bit point reaches 70.28% at 40.592 B/vector and about 0.640 ms; that point needs more seeds.

Supplemental full-cache GloVe/OpenAI screening uses seed 0 and 8,192 training rows. At four bits/dim, log changes GloVe recall from 89.95% to 89.57% while increasing memory from 112.79 to 117.04 B/vector; OpenAI is essentially tied at 98.35/98.36%. Cap-4 lowers OpenAI to 97.28% and increases memory to 920.39 from 887.80 B/vector. All-eight-bit marginal/log recipes are **byte-identical** on both datasets, as they are for each SIFT seed at 1032 bits: allocation has no freedom.

**Recommendation:** keep the marginal allocator as default. A base-calibrated serving-constrained option merits further evaluation for a concrete latency/memory target; neither a global log-rule replacement nor a blanket four-bit cap is supported. This contributes the baseline and resource-cost evidence to #9 without claiming allocation optimality. For #13, retain precision-preserving register kernels; no tested byte-table candidate merits production dispatch.

### Integrity, timing limits and artifacts

- 64 allocation recipes checked for fixed Gaussian tables and exact budgets; default SIFT recipes match the prior tiny-budget bundle byte for byte. All five all-eight-bit paired settings produce identical recipes. Every SIFT recipe has disjoint-base reconstruction measurements.
- Uninstrumented seed-0 timing: three warmups/three repeats, single-query at one/eight threads, batch-32 at eight threads. Adapter work is inside SIFT timing. Main narrow/allocation cases use shuffled order; supplemental lookup variants run in separate processes. Small differences are not adoption evidence.
- Another unrelated calibration job started on CPUs 0–7 during later supplemental checks. Its overlapping measurements are preserved and labeled. Final supplemental eight-thread lookup comparisons and base-cost calibration moved to physical cores 10–17 with OpenMP binding, excluding that job and its SMT siblings. Background services remain; the host was not exclusive. No other job was stopped or changed.
- C++20/CMake/OpenMP and existing Python 3.12/uv environment; effective `-O3 -march=native -ffp-contract=off` audited. Source/binary hashes, source snapshots, recipes, raw timing/per-query results, group/table footprints, packing/padding/storage, compiler audits, logs, commands, source stability and exit markers are retained. Final focused validation and matrix completeness checks passed. AVX2/NEON performance was not tested.

Report: `benchmark/results/issue14-serving-allocation-20261026/README.md`; drivers: `benchmark/experiments/serving_allocation/`. These are working-tree artifacts, not committed GitHub links. Existing bundles and unrelated work were preserved.

Isolated workspace: `sean@hartmut:/home/sean/experiments/TieredQ-serving-allocation-20261026-XQJU1q`; full indexes and all-document validation arrays remain there. The report bundle includes `finish-study.sh`, `isolated-finish.sh`, complete raw results and three source variants. Final primary source archive SHA-256: `f872c52f1d061bdabac44376d72f548f2e4fbc7c973747bee76466957d11d6b3`.
