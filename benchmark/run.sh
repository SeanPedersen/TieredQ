#!/usr/bin/env bash
# Isolated, locked Python environment; codec compilation happens in run.py.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec uv run --locked --python 3.12 --project . python run.py "$@"
