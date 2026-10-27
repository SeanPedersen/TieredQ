# OpenAI fast-mode point below RaBitQ’s 2-bit serving storage

Supplemental measurement on Hartmut, 27 October 2026. The original audited grid is unchanged.

The storage cap is the original RaBitQ seed-0, eight-thread, batch-32 OpenAI point: 404.0696 bytes/vector. Selection used only seed-0 serving storage, targeting 99.0–99.5% of that cap, before evaluating candidate recall. See `selection.json` for every attempted budget. Only the selected point was evaluated for recall; three seeds were used. Seed 0 supplies timing and storage.

The added native adapter exposes an exact integer code-bit budget. Quantizer and search kernels remain the original measured source. All original metadata source hashes were verified before measurement. A 1,536-code-bit control reproduced the original seed-0 OpenAI batch-32 top-10 IDs and serving storage exactly. The selected budget is 2,411 code bits/vector; accounted serving storage is 401.96076 bytes/vector.

Files:

- `results.csv`: 12 scheduling/seed rows, including four seed-0 timing rows.
- `meta.json`: source verification, input hashes, compile commands, affinity, and UTC date.
- `selection.json`, `control.json`, `diagnostics-seed*.json`: budget selection and validation.
- `adapter.patch`: complete adapter change relative to measured source.
- `run.py`, `prepare.py`, `plot.py`: reproduction scripts; paths retain the original workspace.
- `job.log`: validation, storage selection, and measurement progress.
- `verify_report.py`: independent report table and link verification; run from any working directory with Python 3.12.

## Reproduction

Restore the original retained source and its verified cached datasets in the workspace recorded in `../remote-workspace.txt`. Use the pinned uv environment under `source/benchmark/.venv`. From that workspace:

```sh
python3 matched-storage/prepare.py
source/benchmark/.venv/bin/python --version
cmake -S matched-storage/native -B matched-storage/build \
  -DRABITQ_SOURCE="$PWD/source/benchmark/.deps/rabitq-native" \
  -DPython_EXECUTABLE="$PWD/source/benchmark/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build matched-storage/build -j8
taskset -c 0-7 source/benchmark/.venv/bin/python -u matched-storage/run.py
source/benchmark/.venv/bin/python matched-storage/plot.py
```

`prepare.py` creates an isolated native adapter and CMake project referring to the original source tree; it does not edit the original tree. Adapt the recorded workspace and cache paths if reproducing elsewhere. Saved RaBitQ timings remain from 26 October, so timing comparisons retain shared-host, between-run uncertainty. The report’s original figures exclude the added point; `openai-matched-storage.png` includes it explicitly.
