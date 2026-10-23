#!/usr/bin/env bash
# Build pinned native dependencies and validate without downloading vector datasets.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec uv run --locked --python 3.12 --project . python validate.py
