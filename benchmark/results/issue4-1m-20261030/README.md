# Issue #4 at 1M: DBpedia OpenAI-1536 1M (Hartmut, 2026-10-30)

TieredQ `072ea48` (fast mode) against RaBitQ and Qdrant TQ+ on DBpedia OpenAI-1536 with 999,000 base vectors, plus
the equal-recall budgets and timing behind [docs/benchmark-report.md](../../../docs/benchmark-report.md).
Harness `d89d27d` and its successors in [`benchmark/experiments/issue4_1m/`](../../experiments/issue4_1m/).
Workspace: `~/experiments/TieredQ-issue4-1m-20261030` (tree `source`, SOURCE_REVISION recorded). Host: i9-7980XE,
36 CPUs, 62 GB, shared.

## Data (`data/`)
- `openai1m` = the `openai` 100K set's 1,000 held-out queries against all 999,000 other rows of
  `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M@4a97312`, normalized FP32, exact FP32 IP top-10.
- Checks (`prepare-1m.log`, `verify_1m.py`): queries bit-identical to the 100K set, the 100K base rows equal their
  rows in the 1M base, truth spot check against a fresh matmul, unit norms. Preparation took 48 s from the HF Arrow
  cache. The manifest records base, query, truth and query-row hashes.

## α check (`alpha/`, `job_alpha.sh`)
Fast mode, allocation power 1 / 1.25 / 1.5 (shipped default, arm `base`) / 1.75 / 2, at 1-4 bits/dim, seeds 0-2,
`ab_eval` with `AB_POWER`. Chosen on even query ids, reported on odd ones (`heldout.py`, `alpha/heldout.txt`):

| α | 1 bit/dim | 2 | 3 | 4 | Δ vs 1.5, even ids (selection) | Δ vs 1.5, odd ids (held out) |
|---|---:|---:|---:|---:|---:|---:|
| 1 (MSE) | 0.8928 | 0.9481 | 0.9717 | 0.9855 | −0.49 pt | −0.37 pt |
| 1.25 | **0.8953** | 0.9527 | 0.9750 | 0.9867 | −0.15 | −0.12 |
| **1.5 (default)** | 0.8930 | **0.9546** | **0.9790** | 0.9887 | – | – |
| 1.75 | 0.8937 | 0.9534 | 0.9776 | **0.9892** | −0.03 | −0.04 |
| 2 | 0.8893 | 0.9522 | 0.9754 | 0.9886 | −0.36 | −0.13 |

R@10 over all 1,000 queries, mean of seeds 0-2. No alternative beats 1.5 on the selection half; the optimum is flat
over 1.5-1.75 as at 100K, and 1.25 is ahead only at 1 bit/dim (+0.23 pt). The default stays.

## Equal-memory match (`match-b1/`, `match-b2/`, `match-b4/`)
`match_memory.py --quick` per bits/dim lane, RaBitQ then Qdrant TQ+ in one output directory so both share TieredQ
probes. All reference cells here are new (no 1M reference cells existed); everything else in the report reuses
earlier reference cells.

| Reference | bits/dim | Matched on | Target bits/vector | TieredQ code bits | TieredQ max bits/vector | TieredQ R@10 | Reference R@10 | Δ pt | Win |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| RaBitQ | 1 | disk | 1,632 | 1,557 | 1,631 | 0.8955 ± 0.0022 | 0.8296 | +6.60 | yes |
| RaBitQ | 1 | ram | 1,633 | 1,549 | 1,632 | 0.8942 ± 0.0015 | 0.8296 | +6.46 | yes |
| RaBitQ | 1 | peak | 1,639 | 1,549 | 1,637 | 0.8942 ± 0.0015 | 0.8296 | +6.46 | yes |
| RaBitQ | 2 | disk | 3,232 | 3,136 | 3,218 | 0.9575 ± 0.0015 | 0.9101 | +4.73 | yes |
| RaBitQ | 2 | ram | 3,233 | 3,136 | 3,226 | 0.9575 ± 0.0015 | 0.9101 | +4.73 | yes |
| RaBitQ | 2 | peak | 3,238 | 3,100 | 3,230 | 0.9565 ± 0.0007 | 0.9101 | +4.64 | yes |
| RaBitQ | 4 | disk | 6,304 | 6,202 | 6,298 | 0.9888 ± 0.0008 | 0.9701 | +1.87 | yes |
| RaBitQ | 4 | ram | 6,305 | 6,173 | 6,275 | 0.9887 ± 0.0002 | 0.9701 | +1.86 | yes |
| RaBitQ | 4 | peak | 6,310 | 6,139 | 6,283 | 0.9893 ± 0.0007 | 0.9701 | +1.92 | yes |
| Qdrant TQ+ | 1 | disk | 1,600 | 1,526 | 1,598 | 0.8935 ± 0.0006 | 0.8270 | +6.65 | yes |
| Qdrant TQ+ | 1 | ram | 1,602 | 1,518 | 1,600 | 0.8925 ± 0.0018 | 0.8270 | +6.55 | yes |
| Qdrant TQ+ | 1 | peak | 1,608 | 1,518 | 1,606 | 0.8925 ± 0.0018 | 0.8270 | +6.55 | yes |
| Qdrant TQ+ | 2 | disk | 3,136 | 3,038 | 3,121 | 0.9531 ± 0.0012 | 0.9104 | +4.27 | yes |
| Qdrant TQ+ | 2 | ram | 3,139 | 3,038 | 3,129 | 0.9531 ± 0.0012 | 0.9104 | +4.27 | yes |
| Qdrant TQ+ | 2 | peak | 3,144 | 3,005 | 3,129 | 0.9531 ± 0.0014 | 0.9104 | +4.27 | yes |
| Qdrant TQ+ | 4 | disk | 6,208 | 6,112 | 6,202 | 0.9882 ± 0.0005 | 0.9723 | +1.59 | yes |
| Qdrant TQ+ | 4 | ram | 6,210 | 6,080 | 6,180 | 0.9879 ± 0.0001 | 0.9723 | +1.56 | yes |
| Qdrant TQ+ | 4 | peak | 6,216 | 6,048 | 6,187 | 0.9888 ± 0.0008 | 0.9723 | +1.65 | yes |

All 18 targets are wins (+1.6 to +6.7 pt). TieredQ's seed std is ≤ 0.0022; TieredQ memory is the max over seeds 0-2.

## Equal recall (`equal-recall/`)
`equal_recall.py`: for each reference cell, the smallest TieredQ code budget (bisected to 8 code bits on seed-0
R@10) reaching the reference's mean R@10, then that budget's R@10 over seeds 0-2. 100K sets and SIFT1M use the
issue #9 ship matches' reference recall; OpenAI-1M (`b1/`, `b2/`, `b4/`) this bundle's.

## Timing (`timing/`)
`run.py`, seed 0, threads 1/8 × batch 1/32, 3 warmups + 3 repeats, alone on CPUs 0-7 after all matching: OpenAI-1M
RaBitQ and TQ+ at 1/2/4 bits, TieredQ at every matched budget and every equal-recall budget; TieredQ at the 100K
and SIFT equal-recall budgets (their reference timing is reused, see the report).

## Cross-check of reused reference timing (`dbpedia-speed-100k/`)
`issue9_empirical/job_dbpedia_speed.sh` (29-30 October) timed RaBitQ next to TieredQ on OpenAI/Gemini 100K in one
session. RaBitQ QPS and p50 there are within -10% to +22% of the reused 26-27 October cells. Its TQ+ cells failed
(the tree had no TQ+ shim build) and are not used.

## TQ+ speed caveats
- Hartmut's i9-7980XE has no AVX-512 VNNI, so Qdrant's TQ+ kernels (runtime dispatch AVX-512 VNNI → AVX2 → SSE)
  run their AVX2 path here, while TieredQ and RaBitQ use AVX-512. The shim was built with `-C target-cpu=native
  -C opt-level=3` and fat LTO (cargo fingerprint) and calls the production `score_precomputed_batch`. TQ+ timings
  in `timing/` are therefore host-specific.
- The shim parallelises across queries only: TQ+ 8-thread batch-1 rows are single-threaded scans. Use TQ+ 1-thread
  and batch-32 rows for comparisons.

## Run notes
- The first match launch used one lane per reference (`run-first-two-lanes/`); it was stopped after ~6 minutes and
  restarted as one lane per bits/dim. Its completed bits-1 cells were moved into `match-b1/`.
- Starting three lanes at once raced on the RaBitQ include overlay (`setup_native.prepare` copies and patches it
  per run): the bits-2 lane failed at start (`run-first-two-lanes/match-b2-race.log`) and ran from a copy of the
  tree (`lane_b2.sh`, tree `source-b2`). The final overlay hash matches `patches.json` and the shared native build
  postdates it, so the bits-1/4 cells use the seeded rotator.
- `job_match_time.sh` was stopped after it launched the lanes; `job_equal_recall_time.sh` took over equal recall
  and timing.

## Files
- `job_*.sh`, `lane_b2.sh`, `*.log`, `*-utc.txt`: jobs as run, logs, timestamps.
- `match-b*/`: `cells.csv`, `calibration.csv`, `matches-{rabitq,qdrant}.{json,md}`, `meta-*.json`, per-cell
  `build.json` / `serve.json`.
- `report-tables.md`: `python benchmark/experiments/issue4_1m/report_tables.py benchmark/results/issue4-1m-20261030`.
