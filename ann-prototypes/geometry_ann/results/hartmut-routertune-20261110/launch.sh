#!/bin/bash
# Starts study.py once the router study, whose saved indexes and driver build it uses, has finished.
cd "$(dirname "$0")"
stage() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "$1" 2>/dev/null; }
until s=$(stage ../TieredQ-issue44-router-20261110/status.json); [ "$s" = complete ] || [ "$s" = failed ]; do sleep 30; done
exec python3 study.py
