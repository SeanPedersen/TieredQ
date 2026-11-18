#!/bin/bash
# Runs source/run.sh under the shared benchmark lock; exit code to job.exitcode.
cd "$(dirname "$0")/source"
flock /home/sean/experiments/.tieredq-issue44-benchmark.lock bash "${1:-run.sh}" > ../"${2:-run.log}" 2>&1
echo $? > ../job.exitcode
