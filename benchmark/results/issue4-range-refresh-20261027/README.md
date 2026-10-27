# Issue #4 range-bound refresh and normal-mode control

Measured on Hartmut (Intel i9-7980XE), CPUs 0–7, source `575e167`, GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. Four full corpora: SIFT1M, GloVe-200 (100K), OpenAI3-1536 (100K), and DBpedia Gemini-768 (99K base / 1K held-out documents). Average code budgets are 1, 2 and 4 bits/dimension.

- [Main results and plots](output/README.md): three recall seeds, 1,000 queries, threads 1/8, batches 1/32. Fresh TieredQ measurements plus explicitly sourced saved RaBitQ baselines. The fresh matrix, including Gemini RaBitQ recall seeds 1/2, took 683.13 seconds, excluding builds/input verification.
- [Quick normal-mode control](normal-control/README.md): freshly measured fast and normal modes on the same first 100 queries, seed 0, eight threads, both batch sizes, all corpora and budgets. This subset does not replace the full benchmark.
- [Main CSV](output/results.csv), [audit](output/audit.json), [row provenance](output/row-provenance.json), [bounds comparison](output/ablation-summary.json), and [exact ID parity](ablation/parity.json).

All 48 bounds-enabled/control comparisons returned identical top-10 IDs, recall and serving storage. Native correctness tests and all 16 benchmark adapter tests passed. On eligible one-thread single-query points, measured bounds/control QPS ratios range from 1.09× to 1.49×. Batch-32 searches do not activate this checkpoint; their differences measure between-run variation. The ablation took 608.89 seconds excluding its build.

Saved RaBitQ timings are from October 26; fresh timings are from October 27. Both use Hartmut with the same CPU affinity. Base/query/truth hashes match exactly for all four datasets, and native RaBitQ adapter and timing code hashes match. These are separate timing runs on a shared host; small timing differences should not be treated as stable wins. See the per-row provenance and `reuse/` source artifacts.

The source export is retained in `source.tar.gz`; `run_refresh.py`, `run_ablation.py`, `finalize.py`, and `run_normal_control.py` capture the experiment. The scripts use the existing verified data-cache paths recorded in them; `remote-workspace.txt` identifies the isolated remote workspace. `job.log`, `ablation.log`, and `normal-control.log` retain validation and progress. The no-range control is a copy of the source export with only the checkpoint dispatch condition bypassed, recorded in `ablation/variant.json`.

The normal-mode control completed all 48 timed cells in 190.41 seconds. In its matched 100-query batch-32 comparison, fast mode was 1.32–5.40× faster; normal mode used less storage. The largest absolute Recall@10 difference was 0.004 on this subset. See the per-point table rather than treating this small single-seed control as a general recall estimate.
