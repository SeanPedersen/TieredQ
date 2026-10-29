"""Exploration: recall@10 with the trained basis rounded to FP16 vs FP32, seeds 0-2."""
import json, os, sys
from pathlib import Path
import numpy as np
import codec_native
from data import prepare
data = Path(sys.argv[1])
for dataset, budgets in {"openai": [1100, 1536, 3072], "gemini": [700, 900, 1536], "glove": [200, 400]}.items():
    base, queries, truth = prepare(data, dataset)
    codec_native.set_threads(8)
    for code_bits in budgets:
        for seed in (0, 1, 2):
            os.environ["BENCH_SEED"] = str(seed)
            row = dict(dataset=dataset, code_bits=code_bits, seed=seed)
            for label in ("fp32", "fp16"):
                if label == "fp16": os.environ["KNOB_FP16_BASIS"] = "1"
                else: os.environ.pop("KNOB_FP16_BASIS", None)
                index = codec_native.Index(base, -(-code_bits // base.shape[1]), seed, True, False, True, code_bits)
                ids = np.empty((len(queries), 10), dtype=np.uint32)
                index.search(queries, ids, 8)
                row[label] = round(float(np.mean([len(set(a) & set(t[:10])) / 10 for a, t in zip(ids, truth)])), 4)
            print(json.dumps(row), flush=True)
