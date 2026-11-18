# MS MARCO 1M after DBpedia 100K: the trainers that kept every list nonempty there, plus Lloyd on 256 rows per list at
# about the default budget (3 iterations). Waits for the baseline worker of the first launch to finish.
set -e
cd "$(dirname "$0")"
while pgrep -f "worker lloyd-s64-i10" > /dev/null; do sleep 5; done
T=lloyd-s256-i3,emeans-s256-e2.5,lloyd-s256-i10,emeans-s256-e10,emeans-s640-e1,smeans-s640-e1,smeans-s256-e10
uv run python partition_study.py ../data/dbpedia100k.npy ../results/dbpedia100k --lists 256 --probes 8,16,32 \
  --seeds 0,1 --emeans-path vendor --only lloyd-s256-i3
uv run python partition_study.py ../data/msmarco1m.npy ../results/msmarco1m --lists 1024 --probes 8,16,32 \
  --seeds 0 --truth-rows 10000 --emeans-path vendor --only lloyd-s64-i10,$T
echo DONE
