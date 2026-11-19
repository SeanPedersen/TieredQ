#!/bin/sh
# #58: RSS over time (0.5 s samples, epoch seconds and KB) of one seed-0 full build per version, to find where the
# +96 MB peak build RSS of steps 1+2 appears. base = 7761b5f, new = steps 1+2. Takes the benchmark lock.
E=/home/sean/experiments/TieredQ-issue58-scan-20261119
Q=/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107
SSD=/home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/ssd
export UV_CACHE_DIR=/home/sean/experiments/uv-cache
exec 9>/home/sean/experiments/.tieredq-issue44-benchmark.lock; flock 9
mkdir -p $E/rss
for mode in base new; do
  if [ $mode = base ]; then src=/home/sean/experiments/TieredQ-issue45-bound-20261119/source; else src=$E/source; fi
  index=$SSD/issue58-rss-$mode.tq; tier=$SSD/issue58-rss-$mode.rows
  OMP_NUM_THREADS=32 uv run --project $src --no-dev --with threadpoolctl python $E/build.py $Q/base.i8 $Q/scales.f32 \
    $Q/offsets.f32 1024 $index $tier $E/rss/$mode-build.json 0 > $E/rss/$mode-run.log 2>&1 &
  sleep 3
  pid=$(pgrep -f "bin/python $E/build.py" | head -1)
  : > $E/rss/$mode.tsv
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(awk '/^(VmRSS|RssAnon|RssFile):/{printf "%s ", $2}' /proc/$pid/status 2>/dev/null)
    echo "$(date +%s.%N) $rss" >> $E/rss/$mode.tsv
    sleep 0.5
  done
  wait
  rm -f $index $tier
  echo "done $mode"
done
echo ALLDONE
