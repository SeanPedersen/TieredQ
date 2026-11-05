#!/bin/sh
# Build, test and publish TieredQ using the maintainer's local uv credentials.
set -eu

release_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --no-project --python 3.12 "$release_root/scripts/release.py" --publish "$@"
