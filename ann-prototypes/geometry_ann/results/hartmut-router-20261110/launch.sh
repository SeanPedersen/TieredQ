#!/bin/bash
# Starts study.py once the assignfast and finelists studies, queued earlier on the same lock, have finished.
cd "$(dirname "$0")"
done_stage() { python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["stage"] in ("complete","failed") else 1)' "$1"; }
until done_stage ../TieredQ-issue44-assignfast-20261110/status.json && done_stage ../TieredQ-issue44-finelists-20261110/status.json; do sleep 60; done
exec nice -n 0 python3 study.py
