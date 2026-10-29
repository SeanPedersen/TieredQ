# Issue #4: equal-memory TieredQ vs RaBitQ (OpenAI, Gemini)

Question: at RaBitQ's exact measured disk, steady-RAM or peak-RAM budget, does TieredQ reach higher recall@10? This bundle runs the comparison twice. **Before** uses the library from before the RAM work (`f73d394` + streamed loader `9b7ded4`). **After** adds the folded lookup projection, the parallel query tile, the lean training residual and the FP16 basis (model format 3). See issue #18.

Hartmut (Intel i9-7980XE, AVX-512), GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. OpenAI: 100K × 1536, Gemini: 99K × 768 (the issue #4 splits and ground truth), 1,000 queries, seeds 0/1/2. The before run used CPUs 8–15 and the after run used CPUs 0–7; both record memory and recall only, so the host's other work affects neither.

## Method

`benchmark/match_memory.py --quick`:

1. Measure RaBitQ (official flat adapter) at 1/2/3/4 bits per dimension, seeds 0–2, with `measure_memory.py` cells: build in one fresh process, reload and serve in another.
   - **Disk:** saved index size.
   - **RAM:** steady RSS growth after reload.
   - **Peak:** VmHWM growth over reload and a query workload of one untimed pass at each of threads 1/8 × batch 1/32.
2. Each (RaBitQ bits, metric) pair is a target: the minimum of that metric over RaBitQ's three seeds.
3. For each target, find the largest TieredQ code-bit budget whose seed-0 measurement is at or below the target. The search brackets the target, then bisects with secant steps to within 8 code bits. **Selection reads storage only, never recall.** Every probe is shared across targets and logged in `calibration.csv`.
4. Confirm the chosen budget at seeds 1 and 2. If any seed exceeds the target, the search steps down to the next smaller budget that fits.
5. A **win** requires TieredQ's maximum memory over seeds ≤ the target *and* a higher mean recall.

`--quick` replaces the timed workload (3 warmups + 3 repeats) with one untimed pass per schedule. Allocations, steady RAM and recall are unchanged. Quick mode lowers peak RSS by about 160 bits/vector for *both* codecs equally (TieredQ −161, RaBitQ −157 on OpenAI 2-bit), which is Python timing-harness memory. Both runs use quick mode for both codecs.

## Result

[comparison.md](comparison.md) has all 24 targets per run. Full per-target rows (disk/RAM/peak of both codecs, recall std) are in [baseline/matches.md](baseline/matches.md) and [after/matches.md](after/matches.md).

**TieredQ wins all 24 targets in both runs.** At equal **peak RAM**, the RAM work widened the margin:

| Peak-RAM match | RaBitQ R@10 | OpenAI before → after | Gemini RaBitQ R@10 | Gemini before → after |
|---|---:|---|---:|---|
| RaBitQ 1-bit | 0.8140 | +0.6 → **+4.5** pts | 0.7216 | +1.0 → **+5.7** pts |
| RaBitQ 2-bit | 0.9013 | +1.4 → **+2.7** | 0.8549 | +3.0 → **+4.7** |
| RaBitQ 3-bit | 0.9435 | +0.8 → **+1.7** | 0.9173 | +1.2 → **+2.7** |
| RaBitQ 4-bit | 0.9674 | +0.5 → **+0.9** | 0.9516 | +0.8 → **+1.6** |

On disk and steady RAM the after margins are +2.1 to +5.0 points on OpenAI and +2.4 to +7.2 points on Gemini. Seed standard deviations are 0.0002–0.0042, well below every margin.

This corrects #4's earlier readout ("RaBitQ has slightly higher recall at its exact budgets"). That comparison only placed TieredQ at nominal 1/2/4 bits × d budgets, which never coincide with RaBitQ's sizes. With the code budget calibrated to RaBitQ's measured memory, TieredQ already won before the RAM work, though narrowly on peak RAM.

[after-memory/](after-memory/) holds the after library's disk/RAM/peak matrix at the nominal 1/2/4-bit budgets on all four datasets ([memory-table.md](after-memory/memory-table.md)), also in quick mode.

## Files

- `baseline/`, `after/`: `cells.csv` (every measured cell), `calibration.csv` (every probe and confirmation), `matches.{json,md}`, `meta.json`, and per-cell `build.json`/`serve.json`.
- `baseline-run/`, `after-run/`: job scripts, logs, host CPU/memory/process snapshots and UTC timestamps. `after-run/source-commit.txt` records the library.
- `baseline-full-partial/`: the first before run with the full timed workload, stopped during Gemini 3-bit when the faster quick mode replaced it. Across its 20 completed targets, the chosen budgets match `baseline/` exactly on disk (storage is deterministic), within ±15 code bits on steady RAM and within ±13 on peak, with one exception. At OpenAI 4-bit peak, budget 4,672 overshot the target by 0.66 bits/vector on seed 1 (a few pages of RSS noise). The confirmation step then fell back to the next smaller budget in its cache, 3,954, against 4,665 in quick mode.

## Limitations

- **Resumed runs.** Both runs were interrupted once by a VmHWM reset race on the busy host, now fixed with a post-reset high-water-mark baseline, and resumed from `cells.csv`. On resume, probes served from the cell cache are logged again, so `calibration.csv` contains duplicate rows with identical values.
- **Calibration seed.** Calibration uses seed 0, and seeds 1/2 only confirm. TieredQ storage is fixed-stride per configuration, so seeds differ only in RSS noise.
- **Conservative step-down.** When a confirmation seed exceeds the target, the search falls back to the next smaller budget already measured, which can be far below the boundary (see `baseline-full-partial/`). This under-allocates TieredQ, never RaBitQ, so wins are not overstated, but some margins may be understated.
- **Fixed costs at 100K.** These are fixed per-process costs divided by N = 100K. At larger N, both TieredQ's overhead and the savings shrink per vector.
- **No timing.** Speed is not measured here; see `query-ram-20261028/` and `fp16-basis-20261028/` for the A/Bs.
