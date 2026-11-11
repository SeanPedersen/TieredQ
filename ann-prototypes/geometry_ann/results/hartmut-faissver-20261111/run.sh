#!/bin/bash
# Runs faissver.py with Faiss 1.15.1 and 1.16.0 under the shared benchmark lock.
cd "$(dirname "$0")"
exec 9>>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
for v in 1.15.1 1.16.0; do
  [ -f faiss-$v.json ] || uv run -q --no-project --python 3.12 --with faiss-cpu==$v --with numpy python faissver.py faiss-$v.json > faiss-$v.log 2>&1
done
echo complete > status
