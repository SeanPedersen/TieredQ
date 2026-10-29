"""Disk, accounted RAM and recall@10 (seed 0, 8 threads) of saved TieredQ indexes, one JSON line each."""
import json, sys
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
data, label, folder = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for path in sorted(folder.glob("*.bin")):
    dataset = path.stem.split("-")[0]
    _, queries, truth = prepare(data, dataset)
    index = codec_native.Index.load(str(path), True)
    ids = np.empty((len(queries), 10), dtype=np.uint32)
    index.search(queries, ids, 8)
    n = {"openai": 100000, "gemini": 99000, "glove": 100000}[dataset]
    print(json.dumps(dict(label=label, index=path.stem, disk_bits=round(path.stat().st_size * 8 / n, 1),
                          accounted_bits=round(index.resident_bytes() * 8 / n, 1),
                          recall10=round(float(np.mean([len(set(a) & set(t[:10])) / 10 for a, t in zip(ids, truth)])), 4))), flush=True)
