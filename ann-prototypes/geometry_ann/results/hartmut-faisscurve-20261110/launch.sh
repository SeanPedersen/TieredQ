#!/bin/bash
# Starts study.py once the router tuning study (queued before it on the same lock) has finished.
cd "$(dirname "$0")"
stage() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$1" 2>/dev/null; }
until s=$(stage ../TieredQ-issue44-routertune-20261110/status.json); [ "$s" = complete ] || [ "$s" = failed ]; do sleep 60; done
exec python3 study.py
