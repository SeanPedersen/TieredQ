#!/bin/bash
set -euo pipefail
ROOT=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
SIFT=/home/sean/experiments/TieredQ-issue44-sift-sq8-20261107
cd "$ROOT"
trap 'code=$?; printf "%s\n" "$code" >resume.exitcode' EXIT
# Inputs are already prepared, SHA256 verified and copied to NVMe.
# Release, sanitizer, and remote end-to-end checks passed before launching.
python3 - <<'STATUS'
import datetime,json
from pathlib import Path
Path('status.json').write_text(json.dumps(dict(stage='queued_after_sift_native_int8_controls',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
STATUS
touch native-int8-trainer.ready
cp sift_sq8_study.py "$SIFT/study.py"
if [ -d "$SIFT/results" ]; then mv "$SIFT/results" "$SIFT/results-fp32-training"; fi
cd "$SIFT"
python3 study.py >native-int8-study.log 2>&1
cd "$ROOT"
if [ -d results ]; then mv results results-initial-diagnostic-stopped; fi
rm -f job.exitcode
python3 study.py >native-int8-study.log 2>&1
