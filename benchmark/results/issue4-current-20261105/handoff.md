Finish issue #4 (benchmark report) on Hartmut, repo ~/projects/TieredQ, branch main, so TieredQ can be released.

## Task: re-run the #4 matrix for TieredQ on current main, then rewrite docs/benchmark-report.md and close #4
The user chose a full re-run (2026-11-04). Every TieredQ number in the report must come from current main; references
are re-timed in the same session.
1. **Equal memory:** re-match TieredQ against RaBitQ and Qdrant TQ+ at 1/2/4 bits/dim, on disk / steady RAM / peak
   RAM, seeds 0-2. Use `match_memory.py --quick` for SIFT1M, GloVe-200 100K, OpenAI-1536 100K, Gemini-768 99K and
   OpenAI-1536 1M.
2. **Equal recall:** run `experiments/issue4_1m/equal_recall.py` per dataset and reference cell, then the 3-seed-mean
   check below.
3. **Timing:** time everything through `run.py` in one session, alone on CPUs 0-7, all cells interleaved on one tree:
   - every reference cell (RaBitQ, TQ+ at 1/2/4) on all five datasets, re-timed;
   - TieredQ at every matched and equal-recall budget.
   MS MARCO-1M was already measured on current main: reuse its match, equal recall and recall cells, and re-time only
   its real-query cells in this session so the report is one timing session.
4. **Report:** rewrite `docs/benchmark-report.md`, the summary, the four answers and every table, for the current
   library.
   - Add MS MARCO-1M (real queries) as a sixth dataset, plus a short real-vs-doc paragraph.
   - Drop the "reference timings are from earlier runs" limitation.
   - Update the build-cost section; the old "heaviest build" claim is stale.
   - Mention `TieredQNestedIndex` as the 4-bit single-query path, keeping the existing #13 numbers.
   - Results bundle: `benchmark/results/issue4-current-<date>/`. Commit, then ask before pushing, before posting the
     closing comment on #4, and before closing #4.

Out of scope (the user said "#4 report only for now"): release packaging (PyPI, wheels, public repo, README install
text), the nested-index re-timing, TurboQuant beyond the existing shim, and rerank.

## Why a re-run (state as of 2026-11-04, origin/main 1d913a6)
- The report says library `072ea48`, harness `d89d27d`. 19 library commits since then change the numbers:
  - #13: nibble tables for wide tiers (c0910d6), range-checkpoint pruning (08173a5), and a 6-bit default cap for fast
    IP/cosine (178ef93).
  - #19.3: in-place training and sampling by id, build peak −72 to −84% (825e35f); fast training by default plus a
    **budget-dependent allocation power** (07b5d3f). The power change can move recall, so the recall matches are re-run
    too, not only the timing.
- Expected direction, from MS MARCO-1M on current main (`benchmark/results/issue22-msmarco1m-flat-20261103`):
  - At equal recall, TieredQ stores 0.53-0.63× the bits of RaBitQ and TQ+.
  - QPS 1t·b32 against RaBitQ: 3.0× / 1.9× / 1.0× at its 1/2/4-bit recall. The report still says 0.41-0.47× at 4 bits.
  - p50 against RaBitQ at 4-bit recall: 1.8×.
  - Build peak at 1M: 1.6-4.8K bits/vector. The report says 5.0-14.4K.

## Reuse reference recall and memory cells (no re-measurement)
- **Reference code is unchanged since the last #4 run:**
  - RaBitQ is pinned at `ced3763`;
  - `benchmark/native/module.cpp` changes since d89d27d only touch TieredQ paths;
  - the TQ+ shim is the same build;
  - the data and seeds are the same.
- Seed each new match output dir with the old reference cells, as `benchmark/results/issue4-final-20261029/seed_refs.py`
  does (`seed_refs.py OUTPUT dataset,... SOURCE_CELLS_CSV...`). Sources:
  - SIFT: `results/issue9-ship-20261030/match-sift/cells.csv`.
  - GloVe: `results/issue9-ship-20261030/match-a/cells.csv`.
  - OpenAI 100K and Gemini: `results/issue9-ship-20261030/match-b/cells.csv`.
  - OpenAI-1M: `results/issue4-1m-20261030/match-b{1,2,4}/cells.csv`.
- **Pitfall: the column layout changed on 2026-11-03.** `measure_memory.FIELDS` gained `companion_recall10` (1d913a6),
  and `match_memory.Cells.measure` appends rows with `FIELDS + ["requested_code_bits"]`. `seed_refs.py` writes the
  header from the *old* rows' keys, so the appended rows would misalign with that header.
  - Write the seeded file with the current `FIELDS + ["requested_code_bits"]` header (empty `companion_recall10`).
  - Before starting the lanes, check that a reload with `Cells(...)` gives the old reference recall.
- Check that the seeded cells are the ones the old report used: compare the reference R@10 / disk against the report
  tables.

## Data (all cached, nothing to download)
- SIFT1M, GloVe, OpenAI 100K and Gemini: `~/experiments/TieredQ-issue4-equal-memory-after-20261028/data`
  (`sift-1000000-1000-v2.npz`, `glove-…`, `openai-…`, `gemini-prepared.npz`). TieredQ serves SIFT with native L2;
  `dimensions()` in the old job maps sift 128 / glove 200 / gemini 768 / openai 1536.
- OpenAI-1M: `~/experiments/TieredQ-issue4-1m-20261030/data` (`openai1m-999000-1000-v2.npz`, 5.9 GB).
- MS MARCO-1M: `~/experiments/TieredQ-issue22-msmarco-20261031/data` (`msmarco1m` real, `msmarco1m-doc`).

## Template and tools
- **Most recent full job:** `benchmark/results/issue22-msmarco1m-flat-20261103/job_flat.sh` (match → equal recall →
  timing in one script, one lane per bits/dim) and `job_extra_b4.sh`.
  - `equal_recall.py` bisects on seed-0 recall. Its budget can miss the reference on the 3-seed mean (MS MARCO RaBitQ
    4-bit: 2,362 → 2,378 code bits).
  - Run the step-up check for every equal-recall row whose mean is below the reference, *before* timing.
- **Older templates:** `~/experiments/TieredQ-issue4-1m-20261030/job_match_time.sh` and `job_equal_recall_time.sh`,
  for the OpenAI-1M + 100K layout and the `budgets()` / `cell()` helpers.
- **Python:** `~/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542/source/benchmark/.venv/bin/python` (no
  pytest; use `-m unittest`).
- **Workspace snapshot:**
  - `mkdir ~/experiments/TieredQ-issue4-current-<date>/source && git archive HEAD | tar -x -C …/source`, plus
    `git rev-parse HEAD > source/SOURCE_REVISION`.
  - Build once: `cd source/benchmark && taskset -c 8-23 $P -c "from native_build import build_native; build_native()"`.
    `native_build` now holds a file lock, so parallel lanes no longer race on the overlay.
  - Copy the TQ+ shim into `source/benchmark/turboquant/native/target/release/` from
    `~/experiments/TieredQ-issue4-final-20261029/source-tq/benchmark/turboquant/native/target/release/libqdrant_bench.so`.
- **Tables:**
  - `experiments/issue4_1m/report_tables.py` hard-codes the old bundle paths (SHIP, FINAL, `match_directories`,
    `timing_rows` reading reference timing from FINAL provenance).
  - Point it at the new bundle (one match dir per bits/dim, all datasets, timing from the new bundle only) or write
    `experiments/issue4_current/report_tables.py` reusing its table functions.
  - `NAMES` / `DIMENSIONS` already include msmarco1m. `experiments/issue22_flat/report_tables.py` has the
    "vs fastest reference config" equal-recall column; reuse it.
- **Timing cost on current main:** MS MARCO-1M cells take 1-3 min (TQ+ 4-bit about 5 min). 100K cells take well under
  a minute. OpenAI-1M at 1,536-d is about 1.5× MS MARCO. The OpenAI-1M match took about 2 h 50 min in three lanes
  (2026-10-30); MS MARCO took about 1 h 20 min.

## Constraints
- Home has ~6.3 GB free (/home 100%). The match lanes write index files of at most about 0.8 GB each at 1M, and cells
  delete them. /tmp is tmpfs (RAM).
- **Shared host:** another Claude session may work in the same tree. Commit only your own files with explicit
  `git add <path>`. Never reset, `checkout -- <file>` or rebase. Ask before pushing, posting or closing.
- **CPU lanes:**
  - Timing runs only on CPUs 0-7 and only after all lanes finish. CPUs 18-25 are the SMT siblings of 0-7.
  - Match/recall lanes run on 8-31 (at most 3).
  - Builds and tests use `taskset -c 8-31` (or 16-31).
  - Serialize jobs. Give an ETA before each run. Run job scripts with `bash`.
- **Reporting:**
  - Use measured RAM / disk / QPS from this one session's interleaved runs.
  - At each equal-recall level, compare against each reference's fastest config reaching it, not only the matching
    bits/dim.
  - Report losses plainly. At equal memory TieredQ is slower than RaBitQ from 2 bits/dim up on d ≥ 768, and its 1-bit
    build is about 10× RaBitQ's (8 s vs 0.8 s at 1M).
- The user's goal is a public release: finish and ship, and skip optional extras unless asked.
