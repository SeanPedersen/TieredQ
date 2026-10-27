# OpenAI/Gemini extension at actual RaBitQ serving-storage caps

Five new TieredQ fast points complete the three RaBitQ storage anchors per embedding dataset. The previously measured OpenAI M2 point remains in [../matched-storage/](../matched-storage/README.md). The original main grid is unchanged. M1/M2/M4 label the **RaBitQ** nominal budget defining a cap; TieredQ uses a separately selected exact integer code budget.

| Point | Storage-selection record | Measured rows |
| --- | --- | --- |
| OpenAI M1 | [selection](openai-rabitq1/selection.json) | [CSV](openai-rabitq1/results.csv) |
| OpenAI M4 | [selection](openai-rabitq4/selection.json) | [CSV](openai-rabitq4/results.csv) |
| Gemini M1 | [selection](gemini-rabitq1/selection.json) | [CSV](gemini-rabitq1/results.csv) |
| Gemini M2 | [selection](gemini-rabitq2/selection.json) | [CSV](gemini-rabitq2/results.csv) |
| Gemini M4 | [selection](gemini-rabitq4/selection.json) | [CSV](gemini-rabitq4/results.csv) |

Each code budget is selected using **seed-0 serving storage alone, before evaluating candidate recall**, targeting 99.0–99.5% of its saved RaBitQ cap. The initial code-bit estimate uses the storage slope between the first two main-grid TieredQ budgets, followed by integer bisection in the positive-code-budget interval up to the 4-bit main budget. Up to 16 attempts are allowed. If the search does not find a point in the target band, retain the largest observed size below the cap. No recall optimization is performed. All attempts, selected budgets, and target-band status are recorded.

The isolated adapter in `../matched-storage/` exposes exact code budgets. The archived quantizer and search sources are unchanged. Environment checks verify all original recorded source hashes, the adapter’s complete permitted patch, loaded binary path/hash, audited compile flags, affinity CPUs 0–7, and input/truth hashes. Before selection, one-bit OpenAI and Gemini controls must reproduce archived seed-0 batch-32 top-10 IDs and serving storage exactly. See [controls.json](controls.json).

Each selected point uses the audited procedure: 1,000 queries; recall seeds 0/1/2; threads 1/8; batches 1/32; seed-0 timing with three single-query warmups and three repeats. Recall-only seeds use eight threads and batch 32; their recall is copied to scheduling rows. Storage and timing comparisons use seed 0. RaBitQ’s saved timings remain from 26 October; the additional TieredQ measurements are dated in [metadata](meta.json). Shared-host between-run timing limitations still apply.

[results.csv](results.csv) contains 60 rows, 20 timed, covering five points. [summary.json](summary.json) summarizes all six pairs, including the earlier OpenAI M2 point. [audit.json](audit.json) verifies the extension; [row-provenance.json](row-provenance.json) maps aggregate rows to per-point files. The original experiment’s audit is independent. Candidate storage inspections overwrite only temporary indexes in this extension’s own directory; temporary indexes/build outputs are not included in the retained local artifact set.

## Reproduction

Use the original recorded Hartmut workspace and restored source/dependency/data caches. The existing adapter must first be built following [the earlier supplemental instructions](../matched-storage/README.md). From the workspace recorded in `../remote-workspace.txt`:

```sh
taskset -c 0-7 source/benchmark/.venv/bin/python -u storage-sweep/run.py
source/benchmark/.venv/bin/python storage-sweep/finalize.py
source/benchmark/.venv/bin/python storage-sweep/plot.py
```

The run’s native progress and storage selection are retained in [job.log](job.log). Input and workspace paths in `runtime.py` retain the measured host’s exact locations. Adapt paths to byte-identical archived inputs when reproducing elsewhere. No core-source edits or native rebuild are required after the earlier exact-budget adapter has been built.

Within the local repository, update and verify the report using Python 3.12:

```sh
python3 benchmark/results/issue4-range-refresh-20261027/storage-sweep/update_report.py
python3 benchmark/results/issue4-range-refresh-20261027/storage-sweep/verify_report.py
```

`plot.py` regenerates the focused OpenAI/Gemini storage comparison and all three complete four-dataset figures via `plot_all.py`. Figures connect all measured points, including the six supplemental points, in code-budget order. Lines are visual guides and do not estimate unmeasured results. Thread settings remain separate; figures are exported as PNG and vector PDF. `plot_style.py` holds their shared visual styles. Original figure artifacts in `../output/` remain unchanged.

`report_data.py` loads per-schedule points for report generation and plotting. `verify_report.py` independently recalculates numerical table entries directly from CSV rows and checks relative links. The previous verifier in `../matched-storage/` belongs to the earlier one-point report snapshot; the extension’s verifier validates the current report.
