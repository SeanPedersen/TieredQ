#!/bin/bash
# Starts study.py once the resumed Faiss curve study has finished.
cd "$(dirname "$0")"
stage() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$1" 2>/dev/null; }
until grep -q 'faisscurve resumed' ../reorder-nested.log 2>/dev/null; do sleep 60; done
sleep 30
until s=$(stage ../TieredQ-issue44-faisscurve-20261110/status.json); [ "$s" = complete ] || [ "$s" = failed ]; do sleep 30; done
exec python3 study.py
