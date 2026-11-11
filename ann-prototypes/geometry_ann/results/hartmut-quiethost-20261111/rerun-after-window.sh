#!/bin/zsh
# Reruns the #52 quiet-host study once the quiet window (#41, #53) is done: the first attempt failed at the driver build.
cd ~/experiments/TieredQ-issue52-quiethost-20261111
while [[ ! -f quiet-window.done ]]; do sleep 30; done
mkdir -p results-failed && mv results/driver results-failed/driver-parse-error
rm -f job.exitcode && python3 study.py > launcher-rerun.log 2>&1
echo $? > rerun.done
