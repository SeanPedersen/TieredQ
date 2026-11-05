# Issue 26 continuation

Unmodified main 5509f27 baseline: `fast-vs-full-8.md`. These are real-codec results. Search times are provisional because compilation and the pre-existing simulation ran concurrently.

| Dataset | full misses@10 | full+cache misses@10 | fast misses@10 |
|---|---:|---:|---:|
| GloVe | 82 | 82 | 159 |
| Gemini | 25 | 25 | 68 |
| OpenAI 100k | 21 | 21 | 38 |

Experimental changes remain uncommitted in `/Users/Studio-MacMini/projects/TieredQ-issue26`:
- Full-mode cap override no longer affects fast prefixes; explicit configuration takes precedence.
- Refinement cap override accepts 1–12, with combined widths up to 12 and individual fields at most 8. Default cap remains 8.
- Finer allocation uses full mode; suffix-only ranks exceeding the field width fail explicitly.
- Narrow models preserve the original compact tables; persistence accepts old byte tables and new wide tables and validates tier widths against available tables.
- Additive allocation tests retain their byte-codebook scope. Refinement tests cover cap isolation, serialization and scalar/AVX-512 parity at caps 8/10/12.

Validation on hartmut: CTest passed 2/2, including the full codec suite (156.14 seconds).

Remote sweep: `~/experiments/TieredQ-issue26-real-20261104/benchmark/experiments/issue26_bound/{nested_recall.py,run-nested.sh,nested-build.log,nested-run.log,nested-8.md}`. Guarded runner starts only after successful tests and Python package installation. Actual payload bits are reported; requested total_bits is not assumed to equal stored bits. Prefix 6/7 bits per dimension, caps 8/10/12, R=64/100/200, lists=1, suffix rows, three timing repetitions. No equal-budget claim is valid for rows where actual payload differs.

The user agreed to retain explicit caps during validation; automatic activation policy is undecided. Wide tiers may be useful at or below 8 bits/dimension because the cap is per coordinate and the budget is an average. No commit or issue post has been made. The pre-existing recall_sim process was left untouched.

## Follow-up finding

The original automatic-prefix-cap sweep is saved as `nested-8-control-partial.md`. GloVe completed, Gemini completed, and OpenAI stopped at prefix 6/cap 10 because the finer model codes fewer ranks than the prefix. With a uniform prefix tier, averaging finer widths over that tier erases wide-cap redistribution; these are control results, not evidence about the selected wide-tier design.

The experimental nested Python builder now has `prefix_max_bits=0` (existing automatic default), allowing an explicit 8-bit prefix cap independently of the prefix's average budget. The new sweep uses 8, allowing spectral allocation at 6/7 bits per dimension. Refinement now permits fewer finer ranks: existing trailing prefix ranks remain unchanged; common-rank span validation checks only the available finer basis. This latest change is under renewed CTest validation, not covered by the earlier pass yet.

Active remote driver: `benchmark/experiments/issue26_bound/validate-sweep.sh`, log `validate-sweep.log`, result `nested-prefix-cap8-8.md`. It sequentially rebuilds, tests, reinstalls Python, then runs the sweep. The earlier `run-prefix-cap.sh` waiter will time out without starting because the control run failed; it does not run compute work. The original control runner is finished.

Storage caveat: nested suffix rows include 4 bytes for their own scale/correction, beyond the prefix metadata. Even matching code-bit payloads do not imply matching total bytes per vector with a single full model. Later versions of the sweep script report index-plus-tier resident bytes/vector as well; early control rows only report code bits and suffix row bytes.

Latest validation passed CTest 2/2 (136.51 seconds) and Python installation succeeded. The prefix-cap8 sweep is running. Gemini prefix 6/cap10 and cap12 both achieve 18 misses versus full-cap8 25 and fast 68, with exactly 6144 code bits. R=64/100/200 are identical on completed cells. Prefix7/cap12 gets 18 misses with only 5889 bits; report its actual payload rather than claiming exact budget use. GloVe prefix7/cap10/12 gets 75 misses with 1598 code bits (requested 1600).

The extended sweep script supports 7 bits/dimension (only prefixes below total budget) and optional --sample-seed for additional random training subsets. This varies the training subset; codec and IVF seeds remain 0, so it must not be described as a codec-seed sweep.

## 8-bit 100k sweep complete

All 54 combinations completed. R=64/100/200 produced identical recall in every configuration. Best observed misses: GloVe 75 (prefix7/cap10 or12; 1598 bits), Gemini 18 (prefix6/cap10 or12; 6144 bits), OpenAI 11 (prefix7/cap10 or12; 12285 bits). Compare fast baseline misses 159/68/38 and full cap8 82/25/21. Full cap10 previously had 72/16/3 misses; nested does not recover all of its improvement. Source: `nested-prefix-cap8-8.md`.

Extended sequential runs are active under remote `benchmark/experiments/issue26_bound/run-extended.sh`, log `extended-run.log`: 7-bit 100k; 8-bit 1M; 7-bit 1M; additional training subsets seeds 1 and 2 at 8-bit 100k. No multiple codec benchmark jobs overlap in this driver. The pre-existing simulation remains active independently, so timing is still preliminary. The additional subset runs are not codec-seed variations.

## Latest checkpoint

7-bit 100k sweep finished; results: `nested-prefix-cap8-7.md`. Gemini prefix6/cap10/12: 32 misses at exactly 5376 code bits. OpenAI prefix6/cap10/12: 20 misses at 10748 bits (requested 10752). Matching 7-bit full/fast baselines still need to be measured before making relative claims.

The 8-bit 1M sweep has started on OpenAI (999000 rows, D=1536, 1000 queries); no completed 1M cells at this checkpoint. Extended driver will then run MS MARCO, 7-bit 1M, and training-subset seeds 1/2. The rejected recall_sim process PID 1568642 exited on its own; it was not killed.

Remaining work: collect 1M and training-subset results, obtain corresponding real-codec full/fast baselines, decide automatic cap policy, and assess piecewise-uniform quality before implementing the later exact wide scorer. Extra training subsets do not replace varying the internal codec RNG seed, which the Python builder currently fixes to 0.

## Two-stage fast-mode proxy (issue's chosen design), 8 bits/dim, 100k — `two-stage-proxy-8.md`

The 1M nested sweep was stopped on request (partial `nested-1m-8.md` on hartmut). Proxy: one full-mode model per cap at 8·D bits; stage 1 reads each stored code's top 8 bits (wider codes → Gaussian coarse-cell centroid), scored in float32; stage 2 rescores top R exactly from the same row. Same bytes/vector for every cap, no extra storage. Experimental hook `TieredQ::experimental_decode_ranks` + binding `TieredQIndex.experimental_decode` (TieredQ-issue26 and hartmut copy only; not for commit). Script: `benchmark/experiments/issue26_bound/two_stage_proxy.py`.

Missed top-10 hits at R=64 (R=100/200 identical; stage 1 never dropped a true hit by R=64 in any row):

| Dataset | fast cap 8 today | two-stage cap 8 | cap 10 | cap 12 | bytes/vector |
|---|---:|---:|---:|---:|---:|
| GloVe | 159 | 84 | 80 | 80 | 205 |
| Gemini | 68 | 30 | 21 | 20 | 773 |
| OpenAI 100k | 38 | 25 | 8 | 4 | 1541 |

Caveats: the proxy decode omits the stored centroid-correction term, so stage 2 is ~2–5 misses worse than the real full scorer (full cap 8 = 82/25/21). Stage 1 does not model the int8 nibble-table rounding; fast cap 8's own top-64 also lost no hits, so this likely matters little after rerank, but it is unverified. No latency. Next: real kernel (hierarchical prefix codebook + fine bits in rows, exact rescoring of R=64), then latency, 7 bits/dim and 1M.

## Two-stage proxy at k=100 — `two-stage-proxy-8-k100.md`

Missed top-100 hits (of 100000) after exact rescoring; R=150..1000 identical, and the top 150 stage-1 candidates contained every true top-100 hit for every cap and dataset:

| Dataset | fast cap 8 today | cap 8 | cap 10 | cap 12 |
|---|---:|---:|---:|---:|
| GloVe | 1166 | 608 | 614 | 614 |
| Gemini | 641 | 309 | 213 | 194 |
| OpenAI 100k | 389 | 190 | 112 | 87 |

Measured real fast-scan cost of a deeper shortlist (depth 100 → 1000, ms/query): Gemini 0.66 → 0.74, OpenAI 1.32 → 1.39, GloVe 0.20 → 0.26. Candidate rule supported so far: R = max(64, 1.5·k) (k=10 tested at R≥64, k=100 at R≥150; smaller R untested). Float32-rerank rows are a storage-heavy reference only (issue's scan+rerank comparison), not a candidate.

## Real kernels via existing TieredQRefinement, prefix matched to the wide allocation — `nested-matched.md`, `nested-latency.md`

Flat nested (lists=1, tier_rows=suffix), prefix bits = proxy stage-1 bits (cap 8), refinement cap 10/12, total 8·D − 32 (suffix metadata). Misses / ms per query (batch of 1000, 8 threads):

| | k=10 R=64 | k=100 R=150 |
|---|---|---|
| Gemini fast cap 8 today | 68 / 0.69 | 641 / 0.68 |
| Gemini nested cap 10 | 13 / 1.09 | 193 / 1.25 |
| Gemini nested cap 12 | 14 / 1.02 | 196 / 1.26 |
| OpenAI fast cap 8 today | 38 / 1.33 | 389 / 1.34 |
| OpenAI nested cap 10 (over budget: 12341 bits) | 13 / 2.01 | 131 / 2.61 |
| OpenAI nested cap 12 | 11 / 2.05 | 139 / 2.60 |

Gemini beats the proxy; OpenAI misses its proxy (4 / 87) — refinement gains are whole bits per prefix tier, flattening OpenAI's per-rank 9–12-bit allocation. Resident bytes ~+13/vector (IVF id map etc.).

Latency breakdown, k=10 R=64 cap 12 (ms/query): Gemini flat 0.657 (depth 64: 0.657), IVF lists=1 0.880, nested stage 1 0.827, nested total 0.966. OpenAI flat 1.319 (depth 64: 1.340), IVF 1.731, stage 1 1.586, total 2.036. So: IVF scan path +30% over the flat path; rescoring 64 rows +0.14 / +0.45 ms — far above the decode arithmetic, unprofiled. Deeper shortlist in the flat scan is ~free.

## Flat fast mode with wide tiers — real implementation, `fast-wide-8.md`

Experimental tree TieredQ-issue26 (uncommitted): DV_TIEREDQ_FAST_WIDE_BITS lets fast-target training allocate 9-12-bit tiers whose books refine the canonical 8-bit additive book (Gaussian Lloyd within each cell); TieredQIndex fast keeps a prefix view (tiers clipped to 8 bits) for the unchanged scan plus a packed fine row per vector; search scans to max(64, 1.5k) and rescores exactly. Bug found+fixed: detail::read_code reads ≤2 bytes, 11-bit codes need 3 (shared read_wide_code in tieredq_layout.h). CTest 2/2 passed before the latency changes (not re-run since).

| Dataset | cap | bytes/vec | resident | misses@10 | ms/q@10 | misses@100 | ms/q@100 |
|---|---|---|---|---|---|---|---|
| Gemini | 8 | 773 | 776.5 | 68 | 0.664 | 641 | 0.675 |
| Gemini | 10 | 773 | 776.3 | 15 | 0.774 | 134 | 1.027 |
| Gemini | 12 | 773 | 776.3 | 16 | 0.777 | 138 | 1.014 |
| OpenAI 100k | 8 | 1541 | 1548.2 | 38 | 1.343 | 389 | 1.341 |
| OpenAI 100k | 10 | 1541 | 1548.0 | 16 | 1.657 | 98 | 2.178 |
| OpenAI 100k | 12 | 1541 | 1547.9 | 10 | 1.636 | 92 | 2.167 |
| GloVe | 8 | 205 | 206.0 | 159 | 0.184 | 1166 | 0.195 |
| GloVe | 10/12 | 205 | 206.0 | 84 | ~0.22 | 653 | ~0.26 |

Latency overhead remains (+17-23% at k=10, +32-62% at k=100); rescoring is batch-parallel now but its per-candidate cost (~30 ns/coordinate) is unexplained; perf cannot open events on hartmut (needs root sysctl).

## Flat fast wide tiers — latency work, final state (`fast-wide-8.md`)

Fixes: (1) batch rescoring parallel across queries and each query chunk's distinct candidates decoded once in document order (lookup blocks interleave 256 documents, one cache line per pair byte); (2) FastRowReader gathers pair-major with prefetch; (3) single queries read+score candidate slices on all workers; (4) weights from the scan's own fast projection (project_fast_queries), passed into the single-query scan and reused for rescoring in both paths (batch/single results identical); (5) reader built once per index; (6) R = max(32, 1.5k) from a depth sweep (k=10 recall flat from R=24). Temporary timers removed. CTest 2/2 passed incl. new tests/test_tieredq_fast_wide.cpp.

| Dataset | cap | recall@10 | batch ms@10 | single ms@10 | recall@100 | batch ms@100 | single ms@100 |
|---|---|---|---|---|---|---|---|
| Gemini | 8 | 0.9932 | 0.668 | 1.313 | 0.9936 | 0.666 | 1.402 |
| Gemini | 12 | 0.9984 | 0.657 | 1.246 | 0.9986 | 0.742 | 1.600 |
| OpenAI | 8 | 0.9962 | 1.355 | 2.731 | 0.9961 | 1.336 | 2.969 |
| OpenAI | 12 | 0.9990 | 1.419 | 2.970 | 0.9991 | 1.544 | 3.670 |
| GloVe | 8 | 0.9841 | 0.184 | 0.248 | 0.9883 | 0.191 | 0.331 |
| GloVe | 12 | 0.9916 | 0.199 | 0.315 | 0.9935 | 0.241 | 0.427 |

Same bytes/vector and resident bytes for every cap. Open: k=100 overhead (+11-26%), wide fast save/load format, Python config for max_bits (benchmarks use DV_TIEREDQ_FAST_WIDE_BITS), IVF/TieredQFastBlocks with wide models untested, AVX2/NEON paths untested, 1M datasets.

## 6 and 7 bits/dim (`fast-wide-6.md`, `fast-wide-7.md`); auto = today's cap max(6, ceil(bits/dim))

| bits/dim | Dataset | auto misses@10 / batch ms | cap 8 | cap 10 | cap 12 |
|---|---|---|---|---|---|
| 7 | Gemini | 93 / 0.583 | 100 / 0.581 | 23 / 0.626 | 23 / 0.612 |
| 7 | OpenAI | 52 / 1.167 | 66 / 1.162 | 14 / 1.300 | 20 / 1.318 |
| 7 | GloVe | 220 / 0.166 | 213 / 0.169 | 213 / 0.165 | 213 / 0.169 |
| 6 | Gemini | 126 / 0.304 | 95 / 0.501 | 48 / 0.556 | 48 / 0.561 |
| 6 | OpenAI | 76 / 0.616 | 79 / 1.000 | 28 / 1.167 | 28 / 1.184 |
| 6 | GloVe | 336 / 0.090 | 290 / 0.143 | 290 / 0.138 | 290 / 0.138 |

At 6 bits/dim any tier above 6 bits slows the scan 55-90% (#13's cap-6 effect): an 8-bit prefix is too expensive there. Wide tiers only switch rescoring on when allocated (GloVe 7 bits: none, no gain). Decision (user): policy C — allocator-predicted gain decides wide tiers; prefix width should follow the automatic fast cap (6/7/8) instead of always 8. Pending user decisions: full mode stays at cap 8 for this issue? exact rescoring without wide tiers as a separate issue?

## Policy C + prefix at the automatic cap — `policyc-{6,7,8}.md` (2026-11-05)

Implementation (TieredQ-issue26 tree): `TieredQConfig::fine_tiers` {automatic (default), off, on} + `max_fine_bits` (default 12); `max_bits` is again the scan cap (≤ 8) and the fine-tier prefix width (automatic: max(6, ceil(B))). Fine books refine the prefix-width additive book; training allocates with and without fine tiers under the same stored bytes and keeps fine tiers when forced or predicted ≥ 1 dB lower weighted error (`src/tieredq_train_fine.h`). Model records `fast_prefix_bits` (13 tables + prefix width; old models unchanged). Env vars DV_TIEREDQ_FAST_WIDE_BITS / DV_TIEREDQ_MAX_BITS / DV_TIEREDQ_REFINE_MAX_BITS removed; full mode, L2 and refinement stay ≤ 8 (on → error, automatic → off). IVF, TieredQFastBlocks and refinement reject fine-tier models; Python IVF/nested bindings pass fine_tiers off. Python TieredQIndex: max_bits, fine_tiers='auto'|'off'|'on', max_fine_bits, fast_prefix_bits property. Fine-tier indexes persist (fast payload version 3 = prefix storage + fine rows; version-2 files load unchanged). experimental_decode and the nested/refinement cap-12 experiment were reverted (backup patch in session scratchpad).

Predicted gain (dB): Gemini 8.2-8.4, OpenAI 9.0-9.2 at 6/7/8 bits/dim; GloVe -0.19 at all three → threshold 1 dB separates them; auto = on:12 everywhere except GloVe (auto = off).

auto vs off (misses, batch ms/q, single ms; same bytes/vector, resident within 1 B):

| bits/dim | Dataset | @10 off → auto | batch @10 | single @10 | @100 off → auto | batch @100 | single @100 |
|---|---|---|---|---|---|---|---|
| 8 | Gemini | 68 → 16 | 0.657 → 0.675 (+3%) | 1.344 → 1.262 | 641 → 137 | 0.673 → 0.750 (+11%) | 1.415 → 1.571 (+11%) |
| 8 | OpenAI | 38 → 10 | 1.335 → 1.431 (+7%) | 2.839 → 2.948 (+4%) | 389 → 92 | 1.340 → 1.547 (+15%) | 3.058 → 3.483 (+14%) |
| 7 | Gemini | 93 → 35 | 0.586 → 0.604 (+3%) | 1.135 → 1.083 | 746 → 250 | 0.588 → 0.669 (+14%) | 1.223 → 1.399 (+14%) |
| 7 | OpenAI | 52 → 18 | 1.180 → 1.291 (+9%) | 2.597 → 2.526 | 463 → 163 | 1.194 → 1.397 (+17%) | 2.630 → 3.205 (+22%) |
| 6 | Gemini | 126 → 53 | 0.315 → 0.343 (+9%) | 0.855 → 0.811 | 1102 → 425 | 0.311 → 0.413 (+33%) | 0.950 → 1.083 (+14%) |
| 6 | OpenAI | 76 → 29 | 0.622 → 0.734 (+18%) | 2.057 → 1.988 | 715 → 279 | 0.626 → 0.840 (+34%) | 2.035 → 2.556 (+26%) |
| 6-8 | GloVe | unchanged (auto = off) | | | | | |

vs the 8-bit prefix (`fast-wide-6/7.md`): 6 bits/dim keeps nearly all of the recall (Gemini 53 vs 48, OpenAI 29 vs 28) at +9/+18% batch instead of +77/+88%. 7 bits/dim: OpenAI 18 vs 14-20 at +9% (was +10-13%); Gemini 35 vs 23 at +3% (was +5-7%) — the 7-bit prefix costs Gemini recall, unexplained (refined 8-bit book vs additive 8-bit tier? stage-1 depth?). Forced fine tiers on GloVe (allocation predicted worse) still gain 159 → 84 / 220 → 172 / 336 → 298: exact rescoring alone, i.e. #30's effect, which auto deliberately does not take. Open: k=100 batch overhead +11-34%; single seed; 1M sets.

Verification: CTest 2/2 on Mac ARM/Clang (NEON), hartmut (GCC/AVX-512), frieda (GCC/AVX2); pytest 77+3 skipped (Mac), 80 (frieda), incl. new tests/python/test_fine_tiers.py.

Cleanup (2026-11-05): fine-tier index code renamed wide → fine (FastFine, src/tieredq_fast_fine.h, tests/test_tieredq_fast_fine.cpp). Experiment scripts that needed the removed hooks (two_stage_proxy.py, nested_matched.py, nested_latency.py, real_recall.py, nested_recall.py) were deleted; their tables above stay as recorded (scripts archived outside the repo). fast_vs_full.py (unmodified codec) and fast_wide.py remain; fast_wide.py now also reports add() time, the fine prefix width, seeds (base permutations) and `/pW` scan-cap variants.

## TODO 6 validation — `validation-1m-{6,7,8}.md`, `validation-seeds-{6,7,8}.md`, `prefix7-vs-8.md` (2026-11-05)

Flat fast TieredQIndex, off vs auto, 1000 queries, 8 threads on cores 8-15, same bytes/vector (resident within 0.1 B). add() times are from the build before the encode-grid fix (see below).

1M, seed 0 (misses, batch ms/q, single ms):

| bits/dim | Dataset | @10 off → auto | batch @10 | single @10 | @100 off → auto | batch @100 | single @100 | add() s |
|---|---|---|---|---|---|---|---|---|
| 6 | OpenAI-1M | 66 → 24 | 6.04 → 5.45 | 16.8 → 13.3 | 743 → 323 | 6.09 → 5.68 | 16.8 → 14.7 | 53 → 131 |
| 6 | MS MARCO-1M | 116 → 30 | 3.97 → 3.49 | 10.9 → 8.3 | 930 → 333 | 4.03 → 3.62 | 11.1 → 8.9 | 33 → 87 |
| 7 | OpenAI-1M | 49 → 16 | 11.38 → 10.67 | 18.3 → 17.2 | 471 → 178 | 11.39 → 10.83 | 19.9 → 18.8 | 75 → 153 |
| 7 | MS MARCO-1M | 73 → 26 | 7.61 → 6.82 | 12.2 → 11.2 | 628 → 197 | 7.71 → 6.97 | 13.1 → 12.4 | 48 → 103 |
| 8 | OpenAI-1M | 39 → 8 | 13.38 → 12.94 | 22.1 → 20.7 | 388 → 127 | 13.55 → 12.83 | 25.4 → 22.1 | 95 → 176 |
| 8 | MS MARCO-1M | 52 → 28 | 8.71 → 8.31 | 14.4 → 13.3 | 474 → 133 | 8.74 → 8.51 | 15.6 → 14.5 | 61 → 115 |

At 1M fine tiers are faster (batch -3 to -12%, single -6 to -24%); at 100k they cost +3-18% batch. Probe (OpenAI-1M, 6 bits/dim, `~/experiments/TieredQ-issue26-gridfix-20261105/probe`): still faster single-threaded (1 core: single p50 95.5 → 80.1 ms, batch 48.9 → 44.8 ms/q), so not threading. Off codes all 1536 ranks at 6 bits (9216 scanned bits); auto skews: 549 ranks at 7-10 bits (scanned at 6), 648 at 2-5, i.e. 8286 scanned bits + 928 fine bits. Likely fewer nibble lookups and tighter single-query pruning; not isolated.

100k seeds (base permutations; seed 0 from `policyc-*.md`), misses@10 off → auto: 6 bits Gemini 126/122/120 → 53/52/52, OpenAI 76/81/72 → 29/24/29; 7 bits Gemini 93/77/88 → 35/27/21, OpenAI 52/61/43 → 18/23/17; 8 bits Gemini 68/64/66 → 16/13/16, OpenAI 38/38/47 → 10/3/5. GloVe auto = off on every seed. Latency deltas match seed 0 within noise.

Gemini 7 bits/dim prefix 7 (auto) vs 8 (`on:N/p8`), seed 0: @10 35 vs 23, @100 250 vs 245, batch +3-5% for prefix 8; OpenAI 18 vs 14/20. With seeds 1/2 at 27/21, the seed-0 gap is mostly seed noise; prefix = automatic cap stays. Stage-1 depth is not the limit (local ARM sweep: R = 16, 32, 64 … 512 all equal; R = k = 10 loses 168).

Encode: fine tiers forced the scalar scale grid (`scale_grid_terms` fallback). Removed; AVX-512 grid got cases 9-12; AVX2/NEON grids were already width-generic; grid parity test extended to 1-12 bits incl. NEON (passes ARM, AVX2 frieda, AVX-512 hartmut). OpenAI-1M 6 bits/dim first add (train + encode), cores 0-7 next to the running benchmark: off 56.0 s, auto 78.1 s (was 130.5 s). Remaining ~30% presumably wider codes' deeper binary search and gathers; to check on a quiet machine against an 8 bits/dim off model.

add() with the encode-grid fix, quiet hartmut (cores 8-15, 8 threads), OpenAI-1M — `add-gridfix-1m.md`: first add (train + encode) off → auto 52.6 → 70.4 s (6 bits/dim, +34%), 75.5 → 90.3 s (7, +20%), 93.0 → 103.5 s (8, +11%); before the fix 53 → 131, 75 → 153, 95 → 176 s. Encode-only rows/s match first-add rates (training adds < 1 s). 8 bits/dim off (all 8-bit codes) encodes at 10.9k rows/s vs 14.4k for 6 bits/dim auto: the remaining cost is that of wider codes (deeper threshold search, gathers past 7-bit levels), not a fallback.
