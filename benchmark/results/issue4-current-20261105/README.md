# Issue #4: current-main benchmark review bundle

Library revision: `c74b4e354b92d848a004e0f50199c3f7db7e0ca4`. Source snapshot: `measured-source.tar.gz`.
Benchmark workspace: `/home/sean/experiments/TieredQ-issue4-current-20261105`. Total benchmark wall time: 11.98 hours.

Six datasets; TieredQ default flat fast index with fine tiers enabled; native L2 for SIFT.
All TieredQ cells are new. Cached reference recalls/storage are audited in `reference-audit.json`,
then independently checked against fresh seed-0 reference builds. Every selected codec/budget is
retimed in the interleaved session described by `timing/plan.json` and `timing/run.log`.
The timing process disappeared after 99 complete cells without an exit marker. `timing/resume.json`
records continuation with unchanged native binaries and settings; the interrupted cell was rerun.
The primary GloVe dataset has 1M vectors. `glove1m/` contains fresh references, matching and equal-recall
measurements; the root matching files retain the original 100K scaling appendix. The superseded timing
directory is retained for provenance only and contributes no final timing rows.

Read `benchmark-report.md` for the report, `tables.md` for complete matched targets and schedule
comparisons, and the `all-*.csv`, `equal-recall.csv` and `preparation.csv` files for raw tabular metrics.
`configurations.md` puts codecs in rows and combines recall, storage, serving RAM, isolated build cost,
encoding throughput, QPS for all four schedules and single-query p50/p95 for every timed budget.
`match-b*/` retains every calibration and seed plus fresh-process build/serve JSON and index hashes.
`equal-recall/` retains bisection probes, three-seed confirmation and 0.95 searches. `reference-build-b*/`
contains fresh reference build peaks, build times, encoding rates and recall/disk checks.

`input-files.json` hashes the prepared inputs. `recovered-data-provenance.json` documents byte-identical
SIFT/GloVe recovery after stale symlinks were discovered. `reference-sources.json` hashes the reused CSVs.
`source-hashes.json`, `host.json`, compiler commands and per-phase metadata retain run provenance.
`harness-instrumentation.patch` adds absolute RSS and encoding diagnostics to JSON without changing
the library or CSV schema; `network-guard.patch` forbids accidental input downloads.

The preparation diagnostic is separate from headline timing: `preparation-diagnostic/` records its
additional API and build provenance. Every timed native cell checks identical search IDs on 32 queries
before preparation-only probes. The production reference shim is preserved. Diagnostic source for the
pinned Qdrant revision was fetched into an isolated directory; Cargo used its existing lock offline.

`excluded-attempts/` records interrupted attempts, which contribute no measurement rows here. A diagnostic
command initially rewrote a loaded adapter binary in place; the interrupted checks were discarded, the
adapter was rebuilt byte-identically, validation passed again, and reference checks restarted.
Temporary serialized indexes created by the harness were cleaned by its existing cell lifecycle;
original datasets, environments and reference artifacts were retained.

## Reproduction

Use the archived source with the Python environment recorded in the logs. Verify/link the prepared
input files from `input-files.json` into a fresh workspace. `experiments/issue4_current/prepare.py WORKSPACE RESULTS_BUNDLE`
seeds current-schema reference cells from this bundle and refuses missing prepared caches; `recover_data.py` recovers
the missing SIFT/GloVe containers from the documented retained inputs. Run `preflight.py`, build the
native adapter once, run native CTest and `validate.py`, then build both preparation diagnostics.
`job.sh` runs matching/recall lanes, serialized interleaved timing and report generation. Set
`WORKSPACE` to the new workspace; paths and the shared Python interpreter are explicit in the scripts.
`bundle.py` packages completed measurements; it never pushes, posts, closes issues or commits.

## Review status

Prepared locally for review. Pushing the report commit, posting the #4 closing comment and closing #4
still require user approval. Separate investigation #38 was created at the user's explicit request.
All headline timing cells must have four schedules and 1,000 queries, and all reported accuracy
points must have seeds 0–2. SHA256 checksums of every bundle file are in `SHA256SUMS.json`.
