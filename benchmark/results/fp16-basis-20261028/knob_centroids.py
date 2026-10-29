"""Exploration: recall@10 and accounted storage vs TieredQ centroid count and code budget (seed 0)."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
data = Path(sys.argv[1])
grid = {"openai": [900, 1100, 1400, 1800, 2400], "gemini": [500, 700, 900, 1200, 1600]}
for dataset, budgets in grid.items():
    base, queries, truth = prepare(data, dataset)
    codec_native.set_threads(8)
    for k in (32, 64, 128, 256):
        os.environ["KNOB_CENTROIDS"] = str(k)
        for code_bits in budgets:
            index = codec_native.Index(base, -(-code_bits // base.shape[1]), 0, True, False, True, code_bits)
            with tempfile.TemporaryDirectory() as directory:
                disk = index.save(str(Path(directory) / "i.bin"))
                loaded = codec_native.Index.load(str(Path(directory) / "i.bin"), True)
            ids = np.empty((len(queries), 10), dtype=np.uint32)
            loaded.search(queries, ids, 8)
            recall = float(np.mean([len(set(a) & set(t[:10])) / 10 for a, t in zip(ids, truth)]))
            print(json.dumps(dict(dataset=dataset, centroids=k, code_bits=code_bits, recall10=round(recall, 4),
                                  disk_bits=round(disk * 8 / len(base), 1),
                                  resident_bits=round(loaded.resident_bytes() * 8 / len(base), 1))), flush=True)
