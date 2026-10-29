"""QPS/latency per schedule and a top-10 ID hash for one saved TieredQ index, in a fresh process."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
from measure import measure
data, path, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
schedules = [tuple(map(int, s.split("x"))) for s in sys.argv[4].split(",")]
dataset = Path(path).stem.split("-")[0]
_, queries, truth = prepare(data, dataset)
index = codec_native.Index.load(path, True)
result = dict(label=label, index=Path(path).stem)
for threads, batch in schedules:
    codec_native.set_threads(threads)
    result[f"t{threads}-b{batch}"] = measure(index, queries, truth, threads, batch)
ids = np.empty((len(queries), 10), dtype=np.uint32)
index.search(queries, ids, 8)
result["ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
print(json.dumps(result))
