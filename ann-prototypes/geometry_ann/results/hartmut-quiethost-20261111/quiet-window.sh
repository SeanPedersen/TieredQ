#!/bin/zsh
# Quiet-host window (other hartmut services paused): each study takes the shared lock in turn.
# 1. #52 item 4 + 5c: Python vs C++ ABBA on one saved 8.84M nested index, then 32-thread batch QPS.
# 2. #41: the nested arxiv serves on the heap fix (the earlier ones overlapped host load).
# 3. #53: the packed-list ABBA serves again, arxiv and Qwen (the saved A/B indexes are reused).
E=~/experiments
cd $E/TieredQ-issue52-quiethost-20261111 && python3 study.py > launcher.log 2>&1
cd $E/TieredQ-issue41-nestedram-20261111 && mkdir -p results-loaded-host && mv results/nested-fix results/nested-fix-s1 results-loaded-host/ \
  && rm -f job.exitcode && python3 study.py > launcher-quiet.log 2>&1
cd $E/TieredQ-issue53-packedlists-20261111 && mkdir -p results-loaded-host && for s in results/*-serve-*; do mv $s results-loaded-host/; done \
  && rm -f job.exitcode && python3 study.py > launcher-quiet.log 2>&1
echo done > $E/TieredQ-issue52-quiethost-20261111/quiet-window.done
