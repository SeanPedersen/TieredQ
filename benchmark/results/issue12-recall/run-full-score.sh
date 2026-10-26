#!/usr/bin/env bash
# Run from benchmark/ after validate.sh; keep this diagnostic separate from native-serving results.
set -euo pipefail
output=output-rabitq-full-score
if [[ -e "$output" ]]; then
  echo "Use a fresh diagnostic output directory" >&2
  exit 1
fi
uv run --locked --python 3.12 python - <<'PY'
import hashlib
import json
from pathlib import Path
from native_build import build_native
from setup_native import REVISION
build = build_native(full_score=True)
output = Path("output-rabitq-full-score")
output.mkdir()
metadata = dict(status="built", upstream_revision=REVISION, native_bounds=False, high_accuracy=True,
                compiler_commands=build.compiler_commands, patches=json.loads((build.overlay/"patches.json").read_text()),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in (Path("../src"),Path("../include"),Path("native"))
                               for p in folder.rglob("*") if p.is_file()})
(output/"meta.json").write_text(json.dumps(metadata,indent=2))
PY
for dataset in sift glove openai; do
  PYTHONPATH="$PWD/build-full-score" BENCH_SEED=0 taskset -c 0-7 \
    uv run --locked --python 3.12 python run.py --codec rabitq --full-score \
      --datasets "$dataset" --bits 8 --seeds 0 --threads 8 --batches 32 \
      --output "$output" --cell "$output/$dataset.json"
done
uv run --locked --python 3.12 python - <<'PY'
import csv
import json
from pathlib import Path
from run import FIELDS
output=Path("output-rabitq-full-score")
rows=[row for dataset in ("sift","glove","openai") for row in json.loads((output/f"{dataset}.json").read_text())]
with (output/"results.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=FIELDS)
    writer.writeheader();writer.writerows(rows)
path=output/"meta.json"
meta=json.loads(path.read_text());meta["status"]="complete";path.write_text(json.dumps(meta,indent=2))
PY
