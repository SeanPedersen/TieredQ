# Issue #14: lookup grouping and serving-aware allocation

**Retain the current production defaults.** The tested 256-entry memory lookups preserve scores but lose speed. Brandt's log allocation rule has no consistent recall advantage. A simple serving-aware policy can extend SIFT's feasible recall frontier under joint memory/latency ceilings, but a universal four-bit cap sacrifices too much quality and can increase actual storage. It merits further work as a constrained, opt-in choice, not a default replacement.

This is TieredQ with fixed Gaussian codebooks, not a faithful Brandt implementation. No production behavior, public format or committed source was changed.

## Lookup grouping

The existing narrow cache already packs two register lookup groups into one byte. The controlled byte candidate indexes that same byte into a 256-entry int32 table containing the exact sum of the original two int8 entries. It preserves each original 32-group scale section and FP32 accumulation order. Thus codes, padding, actual vector storage and numerical approximation are identical; only query-table construction and scanning change. The added tables remain query scratch, counted separately from resident serving memory.

The wide candidate uses 256-entry FP32 tables of repeated original codebook levels. It retains whole-coordinate groups, the same period/phase unpacking and coordinate accumulation order; combining FP32 coordinates would reassociate sums and confound this first precision-controlled test. Models with mixed narrow/wide tiers and all-eight-bit tiers are included. No additional numerical approximation was tested.

Before timing, both candidates checked every document score and full ranking for one query per operating point, plus top-10 scores/IDs for all 1,000 held-out queries. Narrow checks cover SIFT 64/80/128 bits; wide checks cover SIFT 256/512/1032 and GloVe/OpenAI four/eight bits per dimension. Append/save/load and tails at 1/15/16/17/255/256/257/4095/4096/4097 documents, including zero-query ties, passed. Existing SIFT top-10 IDs match the prior bundle.

SIFT narrow results below average the three repeat p50 values/QPS, with unchanged resident bytes within every pair:

| Total bits | Register single p50, 1 thread | Byte single p50, 1 thread | Register single p50, 8 threads | Byte single p50, 8 threads | Register batch-32 QPS | Byte batch-32 QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.003 ms | 7.504 ms | 0.326 ms | 1.112 ms | 3681 | 1018 |
| 80 | 2.259 ms | 8.762 ms | 0.342 ms | 1.243 ms | 3406 | 893 |
| 128 | 2.681 ms | 12.986 ms | 0.424 ms | 1.825 ms | 2729 | 602 |

At 128 bits the 36 register lookups become 18 byte lookups. Original tables occupy 576 bytes/query; the byte variant retains those and adds 18,432 bytes, totaling 19,008 table bytes/query. The codes still occupy 18 bytes/vector for 16 nominal bytes, plus nine metadata bytes and document padding. Thus the candidate halves lookup count while greatly increasing table footprint and losing speed. This does not rule out differently repacked/reordered byte groups or another backend.

Measured lookup timings and allocation tables are in [summary.json](summary.json), with per-call samples in the raw JSONL files. All measurements use top-10 selection; the all-document validation calls are outside timing.

Final supplemental comparisons use eight threads on separate physical cores 10–17. Values average three repeat p50/QPS measurements; both variants consume identical resident storage:

| Dataset / budget | Register single p50 | Byte single p50 | Register batch-32 QPS | Byte batch-32 QPS |
| --- | ---: | ---: | ---: | ---: |
| SIFT 256 total bits | 1.881 ms | 9.915 ms | 2135 | 1442 |
| SIFT 512 total bits | 3.170 ms | 13.902 ms | 1630 | 1080 |
| SIFT 1032 total bits | 5.776 ms | 11.877 ms | 1495 | 1132 |
| GloVe 4 bits/dim | 0.393 ms | 2.244 ms | 9149 | 5450 |
| GloVe 8 bits/dim | 1.026 ms | 2.387 ms | 7642 | 5580 |
| OpenAI 4 bits/dim | 3.923 ms | 17.879 ms | 1044 | 716 |
| OpenAI 8 bits/dim | 9.309 ms | 19.362 ms | 890 | 715 |

See `wide-isolated-timing.jsonl` and `cross-isolated-timing.jsonl`. Earlier contended supplemental runs remain separate; an apparent GloVe-eight-bit improvement there disappeared in the isolated-core comparison. No byte variant improved either final measured latency or batch throughput.

A separate one-thread diagnostic at SIFT 128 bits measures register/byte table construction at about 5.95/9.66 microseconds, scan at 2.98/13.17 ms, and selection at 0.115/0.100 ms. These are single instrumented calls, not primary latency estimates; table construction is nested within preparation. The slowdown remains in scanning rather than selection.

## Allocation

SIFT1M uses the frozen corrected 129-dimensional query-metric adapter, 32,768 sampled base training rows, 256 centroids, default tier rotations, seeds 0/1/2 and 1,000 held-out queries. The main three-seed grid covers 128/192/256/512/1032 bits; seed 0 additionally covers 160/224/288/384 bits. Gaussian tables are identical in every recipe. Retrained default recipes match the prior tiny-budget recipes exactly.

The log rule greedily increments the PCA coordinate with largest `log2(sigma) - bits`, under the same eight-bit cap and exact total budget. The serving prototype adds a four-bit-capped marginal candidate, motivated by the measured representation discontinuity, and chooses among candidate policies/budgets using measured storage/latency constraints and disjoint-base reconstruction MSE. This changes which PCA ranks share rotations, as normal TieredQ training does; it is not merely rewriting bit counts on an already trained basis.

| SIFT total bits | Marginal Recall@10 | Log Recall@10 | Cap-4 Recall@10 | Marginal B/vector | Log B/vector | Cap-4 B/vector |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 50.137% | 50.010% | 50.137% | 27.574 | 27.577 | 27.574 |
| 192 | 65.933% | 66.867% | 65.673% | 33.310 | 33.317 | 35.918 |
| 256 | 76.373% | 76.373% | 74.507% | 41.320 | 41.328 | 44.600 |
| 512 | 93.390% | 93.497% | 83.050% | 73.343 | 73.343 | 74.630 |
| 1032 | 98.770% | 98.770% | infeasible | 141.347 | 141.347 | — |

Values are three-seed means, including bytes/vector; each seed's layout and quality are retained. At 1032 = 8 × 129 bits, marginal/log recipes are byte-identical for each seed. There is no allocation freedom. The cap-4 candidate is excluded above its feasible budget.

At 192 bits, marginal/log use almost the same measured memory (difference about 0.007 B/vector) and single-query latency (about 1.48 ms with eight threads); log gains 0.933 percentage points. At 128 bits it loses 0.127 points; at 256 it ties. These three seeds do not establish a general allocation improvement.

The transition is itself policy-dependent: at seed-0 160 bits, marginal has a five-bit tier, uses 29.306 B/vector and takes about 1.337 ms; log stays narrow, uses 31.583 B/vector and takes about 0.508 ms. It improves Recall@10 from 58.96% to 59.73% but consumes more memory. This is a representation tradeoff, not a same-memory speedup.

A capped allocator also incurs tier-group padding. At 256 bits it uses about 44.60 B/vector versus 41.32 for marginal, while losing about 1.87 recall points. At 512 bits it loses about 10.34 points. Nominal code bits conceal those costs.

## Matched serving constraints

The serving-aware selector uses 2,048 base rows disjoint from codec training for reconstruction MSE and 32 base-vector query proxies for measured single-query cost. It never fits on held-out queries or exact truth. The actual memory ceilings come from measured baseline layouts; the latency ceiling grid is 0.75/1/1.5/2/3 ms. The full selections, including cases that retain the current allocator, are preserved in [summary.json](summary.json).

A useful SIFT operating region is a roughly 37.3 B/vector ceiling with a 1 ms single-query ceiling. In the measured grid, current marginal allocation can use its 128-bit narrow point (50.137% three-seed recall; 27.574 B/vector; seed-0 latency about 0.436 ms), while its 160-bit and larger tested points exceed 1 ms. The cap-4 192-bit point reaches 65.673% three-seed recall, uses 35.585–36.585 B/vector across seeds, and takes about 0.601 ms at seed 0 in the primary held-out timing. Independent base-proxy calibration on the final core set measures about 0.502 ms for this candidate versus 0.428 ms for marginal 128, and selects this same cap-4 candidate without evaluation labels. Both fit the same measured resource ceilings. This demonstrates a useful constrained tradeoff; their resource consumption is not exactly equal.

At about 41.32 B/vector and 1 ms, the seed-0 cap-4 224-bit point reaches 70.28% recall at 40.592 B/vector and about 0.640 ms. This point has only one seed and is screening evidence. It does not justify a default switch.

## GloVe/OpenAI screening

Full existing 100,000-vector caches, 1,000 held-out queries, seed 0, 8,192 base training rows, default rotations and Gaussian tables. This smaller training protocol checks generalization of the direction, not a replacement for the established three-seed 32,768-row benchmark.

| Dataset / budget | Marginal Recall@10 | Log Recall@10 | Cap-4 Recall@10 | Marginal B/vector | Log B/vector | Cap-4 B/vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GloVe 4 bits/dim | 89.95% | 89.57% | 89.57% | 112.79 | 117.04 | 117.04 |
| OpenAI 4 bits/dim | 98.35% | 98.36% | 97.28% | 887.80 | 887.80 | 920.39 |
| GloVe 8 bits/dim | 99.16% | 99.16% | infeasible | 212.80 | 212.80 | — |
| OpenAI 8 bits/dim | 99.82% | 99.82% | infeasible | 1655.92 | 1655.92 | — |

Both eight-bit dataset controls have one all-eight-bit tier and byte-identical marginal/log recipes. The log rule's small SIFT gains do not carry over consistently. Cap-4 increases actual storage and hurts OpenAI recall in this screening.

## Protocol, limits and issue coordination

- Primary latency is uninstrumented seed 0: one/eight threads for single queries (128 queries), eight threads for batch 32 (all 1,000), three warmups, three repeats. SIFT narrow and allocation case orders are seeded/shuffled. Wide/cross lookup variants run in separate sequential processes, with cases shuffled within each variant; do not interpret small differences as a speed claim.
- SIFT timing includes the existing adapter transformation. Cross-dataset input is already the benchmark's cached metric representation. Batch response time and throughput are separate metrics. Query/score output allocations are inside the timed Python call.
- The host was not exclusive. Workload snapshots retain background services, including CPU-consuming uvicorn processes. CPU affinity avoids SMT sibling pairs within the experiment; background processes are uncontrolled. Initial base-proxy calibration has a conspicuous first-case outlier and is retained separately. A later unrelated `run_calibrated.py calibrate` process started on CPUs 0–7 during supplemental measurements; those wide/cross timing rows and the second base-cost run are retained as contended evidence. Final eight-thread wide/cross comparisons and base-cost calibration use physical cores 10–17, `OMP_PROC_BIND=close`, and `OMP_PLACES=cores`, excluding that process and its SMT siblings. Final files contain `isolated` in their names. The unrelated process is preserved. No close timing difference supports a default adoption claim.
- Separate instrumented diagnostics report preparation, table construction, scan, selection and merge. Table time is nested inside preparation; worker counters are comparable with wall time only in these one-thread diagnostics. The wide path constructs shared codebook tables within scan, so its preparation/table counters do not isolate that work. Never substitute these counters for primary timings.
- Packing/add wall time, fixed stride, nominal/stored code bytes, group padding, document padding, lookup counts, register/byte table footprint, serialized index sizes and actual allocated storage are preserved. Byte grouping deliberately preserves the original payload; measured vector storage is unchanged within each lookup pair. Actual storage includes metadata, padded vector arrays, shared serving codec/cache state and (SIFT) adapter; it excludes caller-owned originals, transient query/selection scratch, allocator bookkeeping and RSS.
- Every SIFT allocation recipe has disjoint-base reconstruction MSE. This proxy measures the corrected SIFT objective but does not guarantee ranking quality. The selection costs use base proxies, independently of held-out evaluation labels.
- Effective compiler flags are audited for `-O3 -march=native -ffp-contract=off`; primary, byte-wide and diagnostic binary/source hashes are retained. Only AVX-512 on hartmut was measured; no AVX2/NEON crossover or cache-miss hardware-counter conclusion is claimed. Query table footprint grows substantially, and the CPU's gather mitigation makes memory lookup particularly costly. No hardware cache-counter collection was performed.
- #9 owns the broader allocation-objective/optimality question. These results contribute a log-rule baseline and measured resource-constrained candidates; they do not prove optimality. #13 owns production serving kernels. Retain its precision/storage constraints and register dispatch; the tested memory-gather alternative does not warrant adopting a new default kernel.

## Recommendation

Keep Gaussian codebooks, current marginal allocation and current lookup kernels as production defaults. Do not adopt this byte-table implementation. Pursue a base-calibrated, serving-constrained allocation option only if a concrete workload values the demonstrated SIFT latency/recall tradeoff; validate a larger seed/budget grid and full GloVe/OpenAI training before production adoption. A blanket four-bit cap and a global log-rule replacement are unsupported.

## Artifacts and reproduction

Isolated workspace: `sean@hartmut:/home/sean/experiments/TieredQ-serving-allocation-20261026-XQJU1q`. Remote datasets and previous tiny-budget indexes were reused without changes. Full indexes and all-document validation arrays remain remote. Existing result bundles were preserved; no commits were made.

The [experiment README](../../experiments/serving_allocation/README.md) describes build/variant contracts. Exact measurement commands are in `finish-study.sh` and the final separate-core `isolated-finish.sh`; main lookup/allocation/timing commands use the same paths and `run.py --mode lookup|allocation|timing`. Frozen adapter: `/home/sean/experiments/TieredQ-tiny-budgets-20261026-9NAZyQ/results/adapter.npz`; Python/data: `/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/.venv/bin/python` and `benchmark/data` under that workspace.

Raw files include `lookup-quality.jsonl`, `lookup-timing.jsonl`, `wide-timing.jsonl`, `allocation-quality.jsonl`, `allocation-timing.jsonl`, `reconstruction.jsonl`, `cross-quality.jsonl`, `cross-timing.jsonl`, `costs.jsonl`, `costs-controlled.jsonl`, `phases.jsonl`, `phases-isolated.jsonl`, final isolated timing/cost rows, per-query `.npz` arrays, all recipes, manifests, `layouts.json`, `validation.json`, `compiler-audit.json`, `provenance.json`, `cpu.json`, source snapshots, build/run logs and exit markers. `summary.json` contains aggregates and constrained choices.

Findings posted to [issue #14](https://github.com/SeanPedersen/TieredQ/issues/14#issuecomment-6419394817). The exact comment is retained in `issue-comment.md`.
