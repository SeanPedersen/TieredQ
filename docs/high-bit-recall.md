# High-bit recall investigation (#12)

The main recall loss was in TieredQ's approximate serving representation, rather than an inability of its trained high-bit codec to represent good rankings. A second, independent problem affected the benchmark's SIFT L2-to-inner-product adapter.

## Controlled scorer comparison

Full original datasets and 1,000 held-out queries, seed 0, original L2 lift. Each pair trains the same model and changes only the index representation/scorer. These are nominal 8-bit budget points, not equal-memory comparisons.

| Dataset | Original fast scorer | TieredQ FP32 scorer | RaBitQ original native baseline |
| --- | ---: | ---: | ---: |
| SIFT1M | 0.9570 | 0.9816 | 0.9890 |
| GloVe-200 | 0.9744 | 0.9921 | 0.9878 |
| OpenAI3-1536 | 0.9930 | 0.9981 | 0.9967 |

[Raw controlled comparison](../benchmark/results/issue12-recall/recall-baseline.json). The RaBitQ column is the preserved [original native-serving run](../benchmark/results/hartmut-full-single-run-20261023/README.md), with native bounds and HACC enabled. It is not a full-score diagnostic.

At a full `8*D` budget, GloVe and OpenAI both learn one all-8-bit tier. Their allocation has no freedom left under the default 8-bit cap. Their high-bit fast-path recall deficit therefore cannot be explained by choosing the wrong per-coordinate bit allocation. SIFT has 129 lifted coordinates but a budget of `8*128`: its original allocation is 124 learned PCA/tier coordinates at 8 bits, four at 7 bits and one at 4 bits. These are learned axes, not direct bit assignments to raw input coordinates.

## Precision lost after encoding

The old x86 fast representation rounded every learned reconstruction level into the integer range −63…63. At most 127 distinct numerical values remained for a codebook containing 256 codewords. It then rounded query weights into int8 using one global query scale. SIMD executed these approximated dot products correctly; scalar fallback was not the cause of the measured high-bit score loss.

The FP32 scorer already retained codebook values. The new fast representation keeps the original code IDs ([packed at tier width](#packed-code-storage-and-serving-memory)) whenever a model contains a tier wider than 4 bits, looks up its original FP32 codebook values, and multiplies by unrounded query weights. Each 16-document block is decoded once and shared by up to 32 queries ([scan optimization](#scan-throughput-optimization)). NEON and a portable scorer consume the same representation. Query projection is separated from LUT construction so these weight-based scans do not allocate unused per-coordinate LUTs.

The regression checks compare every score against the codec's independent scalar row scorer, then compare top-k. They cover mixed learned tiers, 5/6/7/8-bit budgets, 4,109 documents, coordinate padding, partial blocks, query tiles, save/load and append. Existing legacy numerical-level files remain readable; their discarded precision cannot be recovered without rebuilding the index. Code-ID files use distinct fast payload kinds (byte-per-coordinate kind 3, packed kind 4), which older readers reject; kind 3 files are repacked on load.

This change applies to fast-only indexes and the auxiliary fast cache on normal indexes. Normal FP32 scoring retains its existing numerical behavior.

## SIFT adapter objective

For the original lift, database vectors are `[x/s, -||x/s||²/2]`, while queries are `[q/s, 1]`. Quantization minimizes reconstruction error in the lifted vector space. The last query coordinate is always one, so its error has disproportionate influence on rankings despite its low database variance. Merely preserving FP32 scoring recovers substantial recall but leaves this objective mismatch.

The revised benchmark adapter fits the second moment `M = E[z zᵀ]` of lifted query proxies `[x/s, 1]`, using at most 32,768 **base** vectors and no held-out queries or ground truth. With `M = L Lᵀ`, it transforms database vectors by `L` and queries by `L⁻ᵀ`. Their exact inner products are preserved; expected squared score error is approximately reconstruction error in the transformed space. A numerical ridge handles rank-deficient calibration samples.

This is a correction to the benchmark adapter, not a new L2 metric in the public TieredQ API. Both FP32 and fast scoring can index the transformed vectors. The normal and fast public APIs continue to estimate inner products.

The benchmark includes adapter fitting/database transformation in construction time, query lifting/matrix multiplication in timed serving calls, and both stored maps plus scale in effective storage. `--legacy-l2-lift` retains the original adapter for controlled reproduction. Results identify the revised adapter as `fast-native-query-metric-lift`.

This finding relates to #9's allocation-objective question: weighting reconstruction error by expected query sensitivity matters. It does not establish that the generic greedy allocator is defective, and does not explain the all-8-bit GloVe/OpenAI loss.

## Experiments excluded from the fix

Rescaling only the added norm coordinate improved SIFT but did not clearly resolve its remaining deficit. A bounded alternating scale/code refinement improved some seed-0 points and worsened another. It was removed from production code; [the experimental patch](../benchmark/results/issue12-recall/experiments/refinement.patch) and raw diagnostic results are retained. No query labels were used to fit the final transform.

## Verification and remaining tradeoffs

The precision-preserving scan costs more arithmetic and lookup work than the old approximate int8 dot product. Improved recall does not establish a throughput win over RaBitQ or by itself reduce the byte-per-coordinate expansion; the subsequent packed-storage change below addresses that expansion. Comparisons must include measured storage and serving throughput. The completed primary run uses seeds 0/1/2 for recall, with timing at seed 0 only, three warmups and three repeats. It measures 8 threads and batch 32 at budgets 4/8. The original full scheduling matrix and an incomplete exploratory serial-timing run remain separate; the final primary run makes no single-query latency claim.

Final measurements and reproduction instructions are in [the result bundle](../benchmark/results/issue12-recall/README.md). The existing native RaBitQ bounds remain enabled in primary comparisons. Exact equal-recall and equal-memory points are not implied by a nominal-budget grid.

## Source-level throughput and storage findings

The original wide x86 path uses 16-query tiles on AVX-512. `search_fast_tiles` assigns one OpenMP task per query tile, so batch 32 provides only two scan workers even when eight threads are requested. The preserved 8-bit measurements scale by about 1.92× (SIFT), 1.98× (GloVe) and 2.21× (OpenAI) from one to eight threads at batch 32; RaBitQ scales by 7.43×, 6.78× and 7.62×. This is a scheduling limitation, not scalar fallback. A single-query fast call has one query tile and therefore one scan worker. The scan optimization below splits documents across workers when query tiles cannot occupy them.

The legacy x86 representation selection is model-wide: **every** tier must fit within four bits to use byte LUTs. A single 5/6/7/8-bit tier selects expanded numerical levels for all coded coordinates. Level storage then takes `4*ceil(coded_dimensions/4)` bytes per vector regardless of the nominal allocation budget, plus nine bytes for two FP32 per-vector factors and a centroid byte. Document blocks are padded to 16 rows. Narrow models instead pack two lookup groups per byte, grouping at most four code bits per lookup, and pad document blocks to 256 rows. These two layouts explain discontinuities when increasing a total code budget crosses the widest-tier threshold.

For example, the original 4-bit-budget SIFT model contains 5-bit tiers and codes 128 dimensions: 128 level bytes plus nine metadata bytes yield 137 bytes per vector before shared model state, despite a packed code budget of only 64 bytes. The 8-bit SIFT model codes 129 dimensions, padded to 132 byte levels, giving 141 bytes before shared state. The original OpenAI 4/8-bit models code 1,534/1,536 dimensions; both round to 1,536 byte levels plus nine metadata bytes. Their nearly identical fast-data sizes are expected, while precision and encoding work differ.

Shared state is also substantial at 1,536 dimensions: the model retains the dense FP32 basis and centroids, derived fused encoder panels, and FP64 centroid projections/half norms. The original benchmark accounts for `codec.memory_bytes()` in addition to vector storage, even though some encoder state is unnecessary for immutable serving. At 100,000 vectors, approximately 25.38 MB of accounted model memory adds about 253.8 bytes per vector at the all-8-bit OpenAI point. This explains much of the difference between its 1,545-byte fast data and 1,798.804-byte reported total. Storage figures follow the harness's allocated-component accounting, not process RSS or exact equal-memory matching.

`trace_layouts.py` reproduces learned tier widths, group counts, padding, legacy vector-data components and shared derived encoder array sizes for all four budgets, using the original adapter. The changed scorer does not change generic model training or its allocation.


## Completed results

Mean Recall@10 across seeds 0/1/2 at the nominal 8-bit budget:

| Dataset | TieredQ | RaBitQ native |
| --- | ---: | ---: |
| SIFT1M | 0.98853 | 0.98820 |
| GloVe-200 | 0.99257 | 0.98850 |
| OpenAI3-1536 | 0.99827 | 0.99670 |

SIFT's seed ranges overlap: TieredQ 0.9866–0.9896, RaBitQ 0.9874–0.9890. Three seeds do not establish statistical superiority. The GloVe and OpenAI improvements persist across all three measured seeds. A full 1,000-query seed-0 normal-mode check reproduces fast-mode recall exactly on all three datasets: 0.9894, 0.9921 and 0.9981.

At delivery of the recall fix, primary/repeated 8-thread batch-32 TieredQ QPS was 266–276 on SIFT, 1,620–1,779 on GloVe and 233–238 on OpenAI, approximately 2–2.6× below the original approximate fast path. The [scan optimization](#scan-throughput-optimization) supersedes these throughput figures without changing recall or storage. Native RaBitQ remains much faster. The result bundle supplies measured storage and discrete budget/recall comparisons; nominal-budget improvements do not imply equal-memory superiority.

The calibration proxy uses the same float32 division as actual query lifting, then accumulates its second moment in float64 and adds a minimal double-precision ridge for the full-rank benchmark data. Rank-deficient samples use an output-precision numerical fallback; the delivered guard leaves the full SIFT calibration maps byte-identical to the measured version. A preliminary variant divided in float64 and used a larger float32-derived ridge; its SIFT seed-0 8-bit recall was 0.9868. That incomplete run was stopped and preserved as an experiment. This sensitivity is another reason to report seed variation rather than claim a robust SIFT win.

## Pruning diagnostic — separate from primary native serving

Disabling only RaBitQ's extra-bit bound pruning, retaining HACC and seed 0, gives:

| Dataset | Native bounds QPS | Full-score QPS | Native recall | Full-score recall |
| --- | ---: | ---: | ---: | ---: |
| SIFT1M | 1,641 | 289 | 0.9890 | 0.9890 |
| GloVe-200 | 11,043 | 1,447 | 0.9878 | 0.9915 |
| OpenAI3-1536 | 4,052 | 281 | 0.9967 | 0.9972 |

Native pruning provides approximately 5.7×/7.6×/14.4× throughput in this diagnostic. Its measured recall impact is dataset dependent. These full-score measurements explain an execution-path advantage; they do not replace the native-serving comparison or establish a general speed advantage for TieredQ.

## Phase profiling

A separately compiled diagnostic extension instruments query projection/weights, centroid terms, LUT preparation, fused scan/accumulation, and selection/finalization. Counters sum worker time; they are not additive parallel wall-clock latencies. The profiling protocol includes three warmup single queries plus three repeats over 32 queries. Python L2 transformation remains outside native counters and inside primary end-to-end timing.

At the high-bit points, scan/accumulation consumes more than 99.5% of scan-plus-selection worker time. Selection costs about 0.13/0.02/0.03 worker milliseconds per query on SIFT/GloVe/OpenAI, versus about 33.3/5.4/35.2 for scanning. Native preprocessing fractions are approximately 0.4%/1.5%/3.7% in this mixed warmup/batch diagnostic. The high-bit path has no LUT-building phase after projection was separated from table materialization. Narrow byte-LUT models retain a fused projection containing centroid outputs, so their separate centroid counter is zero by construction. Accumulation is part of each scan kernel rather than a separate execution pass.

The remaining throughput deficit is therefore primarily full scanning and its FP32 lookup/arithmetic cost, alongside RaBitQ's effective pruning. It is not explained by top-k selection or a hidden high-bit scalar fallback.

## Scan throughput optimization

A follow-up optimized the precision-preserving scan without changing its arithmetic: every fast score remains bit-identical to the scalar codec's row scorer, so recall and measured storage are unchanged in all 36 primary rows. [Result bundle](../benchmark/results/issue12-recall/scan-opt/README.md).

### Measured causes

- **Microcoded gathers.** The benchmark CPU reports `gather_data_sampling: Mitigation: Microcode`. One 16-lane `vgatherdps` cost about 9 ns (~30 cycles), independent or not, so the scan was gather-throughput bound even with query sharing. Affected Intel generations (Skylake through Ice/Tiger Lake) share this cost.
- **Spilled accumulators.** GCC 16 kept the runtime-sized four-query accumulator array on the stack. Each coordinate added a store-forwarding round trip to every accumulation chain and reloaded codebook/weight pointers.
- **Scheduling.** One OpenMP task per query tile left batch-32 and single-query calls with too few workers.

### Changes

1. **Exact register-table decode (AVX-512).** A tier codebook occupies at most 16 registers; `vpermt2ps` resolves the low five code bits within a 32-entry register pair and mask blends on the remaining bits select the pair. Each 16-document block is decoded once per group of up to 32 queries.
2. **Register query tiles.** Eight compile-time query rows accumulate decoded levels with separate multiply and add in coordinate order, which is the scalar codec's rounding sequence. FMA would be about 20% cheaper in accumulation but changes rounding, so it is not used.
3. **Centroid terms without gathers.** Per-tile centroid tables are interleaved by centroid; one 8-row vector load per document feeds `center + weight*correction`, followed by an in-register 16×8 transpose.
4. **Single queries** consume register lookups directly, skipping the decoded block.
5. **Document segments.** `search_fast_tiles` splits documents into whole 4,096-document tiles when query tiles cannot occupy every worker; per-segment top-k lists merge with the same score/id order. This also affects narrow models, without changing their kernels or scores.
6. **Batched query projection.** `query_weights` runs on tiles of eight queries, sharing basis rows and vectorizing centroid dot products across queries with four interleaved chains. Each output keeps its sequential summation order; outputs are bitwise identical to the previous implementation on ARM and x86. Normal-mode `query()` uses the same projection.
7. AVX2 uses the same decode-once structure with scalar decoding; NEON shares decoded blocks across four-query tiles and uses a lane-insert single-query path.

### Results

Same harness and seeds as the primary run, 8 threads, batch 32; recall/storage identical to the delivered recall fix. "Rerun" is the unchanged corrected implementation measured in the same session.

| Point | Published | Rerun | Optimized | RaBitQ native (same run) |
| --- | ---: | ---: | ---: | ---: |
| SIFT 8-bit | 266 | 202 | 1,569 | 1,644 |
| GloVe 8-bit | 1,779 | 1,806 | 7,026 | 10,996 |
| OpenAI 8-bit | 233 | 201 | 868 | 4,021 |
| SIFT 4-bit | 277 | 282 | 1,771 | 1,763 |
| OpenAI 4-bit | 233 | 235 | 1,038 | 4,048 |

Single-query 8-bit QPS (batch 1) improved from 10.6/70/9.1 to 25/169/21 at one thread and to 184/959/112 at eight threads on SIFT/GloVe/OpenAI. Native RaBitQ single queries do not use extra threads: 226/1,628/545.

Stage profiling shows scan worker time per query falling from 33.3/5.4/35.2 ms to 7.5/1.3/7.9 ms. OpenAI centroid preparation fell from 0.80 to 0.03 worker ms per query and projection from 0.65 to 0.35.

### Remaining tradeoffs

- RaBitQ's bound pruning still leads by about 1.6× on GloVe and 4.6× on OpenAI at 8 bits. SIFT now reaches throughput parity at 8 bits and slightly exceeds RaBitQ at the 4-bit point, where TieredQ also has higher recall.
- Machine noise is material: two unrelated processes floated across the measured cores, and RaBitQ itself varied by up to ±10% between runs. Consecutive runs of the scan-only and final code differ within that noise.
- Accumulation runs at about 1.1–1.25 cycles per query·coordinate·16 documents against a 1.0 floor for exact mul+add; decode adds about 13.5 cycles per coordinate per block, amortized over the query group. Further large gains need fewer exact operations or pruning, not lookup tuning.
- Per-segment top-k adds a small selection cost (SIFT 0.13 → 0.20 worker ms per query). The AVX2 epilogue still gathers centroid terms.

## Packed code storage and serving memory

A third step reduced bytes per vector without changing a score: recall is identical to the recall fix in all 36 primary rows, and fast scores stay bit-identical to the scalar codec. [Result bundle](../benchmark/results/issue12-recall/pack-opt/README.md).

### Changes

1. **Serving-only model memory.** Encoder state (fused basis/centroid GEMM panels and FP64 centroid projections) is used only by `encode()`. It is now built on the first encode behind a once-guard and kept; copies of a model share it. Loaded and serving models hold basis, mean, sigma, centroids, codebooks and float half norms. `memory_bytes()` reports encoder bytes only once built. For OpenAI at 100k vectors this removes about 143 B/vector of resident model state (25.4 → 11.1 MB); SIFT 0.70 → 0.21 MB, GloVe 1.25 → 0.38 MB. The benchmark now measures the serving footprint of the reloaded index. Encoder bytes appear separately in diagnostics.
2. **Packed code-lookup caches.** Fast caches for models with tiers wider than 4 bits previously stored one byte per coded coordinate. They now store each document's original packed code row (tier widths, `code_bits` per vector) as 32-bit words interleaved across the 16 documents of a block. The byte layout was the special case of an all-8-bit model, which therefore keeps identical bytes.
3. **AVX-512 unpacking inside the decode.** Each tier segment walks periods of `lcm(width, 32)` bits whose code positions are compile-time immediates (`vpsrld`/`vpslld` plus `vpord` for straddling codes). A segment's bit offsets share a residue modulo `gcd(width, 32)`, its phase, so even widths at odd offsets also get immediate shifts. Ranks before the first period and after the last use variable shifts. Codebooks narrower than 32 entries are replicated across the `vpermt2ps` register pair, so neighbouring code bits above the tier width never need masking. AVX-512BW has no VBMI byte permutes; none are needed. AVX2, NEON and the portable scorer extract codes with scalar 32/64-bit shifts.
4. **Persistence.** New files use fast payload kind 4. Validation rejects any set bit past `code_bits` in a row and any nonzero padding document. Kind 3 byte-per-coordinate files still load: each code is checked against its tier width, padding documents are dropped, and the cache is repacked. They resave as canonical kind 4 files, byte-identical to a freshly built index.

### Results

Same harness and seeds as the primary run, 8 threads, batch 32. "Baseline" is the delivered scan optimization in a separate workspace, measured in the same session.

| Point | Bytes/vector baseline → packed | QPS baseline → packed | RaBitQ bytes / QPS (recall) |
| --- | ---: | ---: | ---: |
| SIFT 4-bit | 137.8 → **73.3** | 1,717 → 1,666 | 84.0 / 1,723 (0.84503) |
| OpenAI 4-bit | 1,798.7 → **887.7** | 1,052 → 1,098 | 788.1 / 4,439 (0.96737) |
| GloVe 4-bit | 121.6 → 112.9 | 11,067 → 10,946 | 148.0 / 12,607 (0.88183) |
| SIFT 8-bit | 141.9 → 137.3 | 1,537 → 1,415 | 148.0 / 1,644 (0.98820) |
| GloVe 8-bit | 221.5 → 212.8 | 5,261 → 7,391 | 276.0 / 10,822 (0.98850) |
| OpenAI 8-bit | 1,798.8 → 1,655.8 | 908 → 898 | 1,556.1 / 3,997 (0.99670) |

4-bit packing roughly halves storage (SIFT −47%, OpenAI −51%). 8-bit points change only by the model-memory trim. GloVe 4-bit uses the narrow lookup cache, so it gains only the model trim. RaBitQ varied by up to 8% between the two same-session runs, and the GloVe 8-bit baseline is a low outlier; 8-bit layouts are byte-identical, and their differences are noise.

Single queries (batch 1, seed 0, QPS at 1 / 8 threads):

| Point | Baseline | Packed |
| --- | ---: | ---: |
| SIFT 4-bit | 60.1 / 390 | 43.6 / 307 |
| OpenAI 4-bit | 53.8 / 252 | 63.5 / 286 |
| SIFT 8-bit | 25.1 / 184 | 24.6 / 180 |
| GloVe 8-bit | 170 / 950 | 163 / 911 |
| OpenAI 8-bit | 21.3 / 112.7 | 21.0 / 111.2 |

The prior estimate was right about storage: packing halves 4-bit storage and leaves 8-bit storage unchanged. It overstated the batch-32 cost (−3% SIFT, +4% OpenAI, both within noise). It was right about SIFT single queries (−21 to −27%) but wrong in sign for OpenAI single queries (+14 to +18%): halving the bytes read outweighs the extra shifts there. The stage profile agrees: SIFT 4-bit scan worker time rose from 4.85 to 5.48 ms per query against the previous session, while 8-bit points were 5–10% faster, and OpenAI 4-bit fell from 5.78 to 5.60.

### Where the single-query cost comes from

Single-core kernels (`microbench/kbench.cpp`, 65,536 documents, median of three interleaved runs on a core whose SMT sibling is idle; ns per document·dimension·query):

| Tiers | Baseline 1 / 32 queries | Packed 1 / 32 queries |
| --- | ---: | ---: |
| SIFT 4-bit learned tiers (8:2, 7:12, 6:16, 5:20, 4:22, 3:28, 2:18, 1:8) | 0.110 / 0.027 | 0.140 / 0.028 |
| 128 × 5 bits | 0.091 / 0.026 | 0.090 / 0.027 |
| 128 × 7 bits | 0.167 / 0.030 | 0.186 / 0.030 |
| 128 × 8 bits | 0.283 / 0.032 | 0.290 / 0.033 |
| OpenAI 4-bit-like (7:13, 6:129, 5:316, 4:536, 3:477, 2:61, 1:2) | 0.112 / 0.030 | 0.092 / 0.024 |

Long segments run the immediate-shift periods at byte-layout speed. Straddling 7-bit codes add a shift and an OR to a single query's chain. Every SIFT 4-bit segment is shorter than its period (32 ranks for odd widths), so those ranks use variable shifts whose count broadcast competes with `vpermt2ps` for port 5. Single queries pay this per query; batch 32 amortizes it. Two alternatives were measured and rejected, both worse than the delivered kernel on the mixed shape and on long periods (0.108 vs 0.090 at 5 bits):
- Per-rank precomputed slots with an unconditional second shift: 0.151 on the mixed shape.
- Memory-broadcast slots with a straddle branch: 0.147 on the mixed shape.

The OpenAI layout exceeds the cache, so halving its bytes wins outright. An earlier pass on an SMT-shared core is kept as `kbench-smt-contended.log` and is not used for these figures.
Aligning every tier to a word boundary would remove segment heads but not short-segment tails, at up to 4 bytes per tier per vector (SIFT 4-bit has eight tiers).

### Budget curve and equal memory

`run.py --bits` now accepts every budget from 1 to 8. Historical threshold-7 curve (seeds 0/1/2, batch 32, 8 threads; [curve](../benchmark/results/issue12-recall/pack-opt/curve/summary.json)). TieredQ bytes / recall@10:

| Bits | SIFT | GloVe | OpenAI |
| ---: | ---: | ---: | ---: |
| 1 | 27.3 / 0.498 | 38.7 / 0.404 | 287.5 / 0.895 |
| 2 | 101.3 / 0.760 | 63.8 / 0.667 | 1,489.8 / 0.938 |
| 3 | 57.3 / 0.877 | 112.9 / 0.817 | 1,651.5 / 0.959 |
| 4 | 73.3 / 0.934 | 112.9 / 0.902 | 887.7 / 0.984 |
| 5 | 89.3 / 0.965 | 212.8 / 0.945 | 1,079.7 / 0.992 |
| 6 | 105.3 / 0.981 | 212.8 / 0.965 | 1,271.7 / 0.996 |
| 7 | 121.3 / 0.986 | 188.8 / 0.986 | 1,463.7 / 0.997 |
| 8 | 137.3 / 0.989 | 212.8 / 0.993 | 1,655.8 / 0.998 |

Historical equal-memory comparison from `summary.json` (best TieredQ recall within each RaBitQ point's earlier measured bytes). The [corrected memory table](../benchmark/results/issue12-recall/storage-refresh/README.md) supersedes storage conclusions for the eleven remeasured configurations; notably, GloVe 4-bit no longer fits within the RaBitQ 3-bit footprint:

- **SIFT:** TieredQ has higher recall within every RaBitQ point's bytes except 2 bits, where the int8-level layout below inflates TieredQ's 2-bit point. RaBitQ 4-bit (84.0 B, 0.845) vs TieredQ 4-bit (73.3 B, 0.934, 1,697 vs 1,626 QPS). RaBitQ 6-bit (116.0 B, 0.958) vs TieredQ 6-bit (105.3 B, 0.981).
- **GloVe:** RaBitQ 3/4/5-bit (116/148/180 B, 0.808/0.882/0.937) vs TieredQ 4-bit (112.9 B, 0.902). RaBitQ 6-bit (212 B, 0.965) vs TieredQ 7-bit (188.8 B, 0.986). RaBitQ is 1.1–1.4× faster at these points.
- **OpenAI:** RaBitQ 5-bit (980 B, 0.982) vs TieredQ 4-bit (887.7 B, 0.984). RaBitQ 6/7/8-bit vs TieredQ 5/6/7-bit at slightly fewer bytes and higher recall. RaBitQ is about 4× faster. Below 788 B only TieredQ 1-bit (287.5 B, 0.895) fits.

The historical curve is not monotonic in bytes wherever a model's widest tier has 5–6 bits (SIFT 2-bit, GloVe 3–6-bit, OpenAI 2–3-bit). Those measured points used their earlier layout: x86 int8 numerical levels, one byte per coordinate (or the narrow lookup cache when all tiers fit 4 bits). Those levels are also requantized, unlike code-lookup scans. Routing every model with a tier wider than 4 bits through packed code lookup removes this numerical-level expansion and preserves scalar-codec scores (measured GloVe 5-bit: 212.8 → 140.8 B). Narrow lookup padding can still create plateaus. Throughput would drop at those points (GloVe 5-bit int8 levels: 13,679 QPS; 7-bit code lookup: 8,465). The approved follow-up now makes this policy change; the historical curve above remains the pre-switch measurement. See [the same-session A/B](../benchmark/results/issue12-recall/code5/README.md) for the five affected points, including the substantial throughput tradeoff.

### Remaining tradeoffs

- SIFT 4-bit single queries are 21–27% slower: its tiers are short segments of odd widths. Batched throughput is unchanged within noise.
- After an append, the resident model includes the encoder; for OpenAI it is about 2.3× the serving model. Releasing it after bulk loading would need an explicit API; it was not added.
- Legacy kind 3 files pay a one-time repack on load.


## Exact mid-width routing and tier-prefix pruning follow-up

Every newly built model containing a tier wider than four bits now uses packed original codes (payload kind 4). This threshold change from 7 to 5 is deliberate: it trades lower bytes and higher recall for substantial throughput losses at several mid-width points. Old kind 2 numerical-level files remain readable and retain their representation through search, resave and append; their discarded precision is not reconstructed. The [same-session A/B](../benchmark/results/issue12-recall/code5/README.md) records bytes, three-seed recall, batch-32 QPS and batch-1 QPS at one/eight threads for SIFT 2-bit, GloVe 5/6-bit and OpenAI 2/3-bit. Historical curves above retain the old routing.

The [Python pruning study](../benchmark/results/issue12-recall/code5/pruning.md) evaluates each widest-tier prefix at 4/8 bits with sign-aware codebook-range bounds, exact metadata terms, a conservative floating-point allowance and an optional residual-norm bound. Survivor scores continue the sequential FP32 accumulator and are checked bit for bit; no C++ pruning code was added.

Whole-tier range pruning does not support a general 2–4× gain: its optimistic best coordinate-work estimates are about 1.76× for SIFT 4-bit, 1.06× for SIFT 8-bit and 1.38× for OpenAI 4-bit. GloVe 4/8-bit and OpenAI 8-bit each contain one tier, so there is no tier boundary to split. Optional per-document residual norms raise SIFT 4-bit to about 2.22× and OpenAI 4-bit to about 1.43× before bound, heap, storage and compaction costs. Batched shared-decode survival unions weaken these estimates further. This leaves a limited SIFT 4-bit experiment plausible, without evidence for a broad 2–4× serving win or parity with RaBitQ's pruning. Splitting within tiers would be a separate study.

## Final-review accounting correction

The archived measurements above predate the final memory-counter review. The
current counter additionally includes fast-cache geometry (`code_offset`,
`tier_of`, `level_scale`), cache objects, narrow-lookup projection panels/groups/
codebooks, and the lazy encoder object. Previously those allocations were
omitted from the resident serving estimate. The correction is largest for narrow
x86 lookup models, which own an additional dense query projection. Historical
CSVs, figures and tables remain unchanged and must not be treated as exact
current-counter storage totals. A [fresh storage-only run](../benchmark/results/issue12-recall/storage-refresh/README.md)
remeasured all eleven reported configurations on the full original base sets
for seeds 0/1/2. The recovered scan-stage source archive matches its recorded
digest and all 43 stage source hashes. Recall, persisted bytes, payload layout
and historical measured QPS are unaffected. Allocator bookkeeping and shared-pointer control blocks remain
excluded; shared encoder allocations are counted per model view, not deduplicated
across copies.

The corrected maximum serving footprints at 4/8 bits are 73.343/137.347 B/vector
for SIFT, 118.706/212.797 for GloVe, and 887.805/1655.925 for OpenAI. GloVe
4-bit exceeds the historical RaBitQ 3-bit footprint (116.012 B/vector), although
it remains below RaBitQ 4-bit (148.012). The approved mid-width policy is
unchanged; OpenAI 2/3-bit now measure 493.905/695.553 B/vector. See the refreshed
table for per-seed diagnostics and all eleven configurations. Other points in
the historical 1–8-bit curves were not remeasured, and their equal-memory
summaries remain historical.
