"""Query-time peak RSS growth of one saved TieredQ index under one schedule, in a fresh process."""
import json, sys
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
from measure_memory import reset_peak, memory_status
data, path, threads, batch = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dataset = Path(path).stem.split("-")[0]
_, queries, _ = prepare(data, dataset)
index = codec_native.Index.load(path, True)
codec_native.set_threads(threads)
baseline = reset_peak()
ids = np.empty((len(queries), 10), dtype=np.uint32)
for start in range(0, len(queries), batch):
    index.search(queries[start:start+batch], ids[start:start+batch], threads)
rss, peak = memory_status()
print(json.dumps(dict(index=Path(path).stem, threads=threads, batch=batch, peak_mib=round((peak-baseline)/2**20, 2),
                      retained_mib=round((rss-baseline)/2**20, 2))))
